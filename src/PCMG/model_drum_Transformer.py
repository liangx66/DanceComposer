'''
encoder
 1. bar/beat position
 2. binary
decoder
 1. type
 2. bar/beat position
 3. density
 4. pitch
 5. duration
 6. velocity
 7. strength (onset_density)
 8. i_beat
 9. n_beat 
'''
import numpy as np
import torch
import torch.cuda
from torch import nn

from utils import Embeddings, BeatPositionalEncoding, PositionalEncoding

from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import TriangularCausalMask

ATTN_DECODER = "causal-linear"

################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    logits -= np.max(logits)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= (sum(probs) + 1e-10)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    try:
        word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    except:
        word = sorted_index[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0, greedy = False):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if greedy:  # greedy decoding
        probs /= (sum(probs) + 1e-10)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cur_word = sorted_index[0]
    else:       # top-p sampling
        if p is not None:
            cur_word = nucleus(probs, p=p)
        else:
            cur_word = weighted_sampling(probs)
    return cur_word


'''
last dimension of input data | attribute:
0: type 
1: bar/beat
2: density
3: pitch
4: duration
5: velocity
6: strength (onset_density)
7: i_beat
'''
class DrumTransformer(nn.Module):
    def __init__(self, n_token, 
                        emb_sizes, 
                        d_model=512,
                        num_encoder_layers=8,
                        num_decoder_layers=8,
                        num_heads=8,
                        dim_feedforward=2048,
                        dropout=0.1,
                        is_training=True):
        super(DrumTransformer, self).__init__()
        # --- params config --- #
        self.n_token = n_token
        self.d_model = d_model
        self.n_layer_encoder = num_encoder_layers
        self.n_layer_decoder = num_decoder_layers
        self.dropout = 0.1
        self.n_head = num_heads
        self.d_head = self.d_model // self.n_head
        self.d_inner = dim_feedforward
        self.d_ff = dim_feedforward
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.emb_sizes =emb_sizes

        self.time_encoding_size = 256
        self.binary_encoding_size = 256
        self.max_len = 512
        # --- modules config --- #
        print('>>>>>:', self.n_token)

        self.encoder_binary_emb = Embeddings(3, self.binary_encoding_size)
        self.encoder_pos_emb = PositionalEncoding(self.d_model, self.dropout)
        self.trainable_pos_emb = Embeddings(self.max_len, self.d_model)
        self.encoder_binary_linear = nn.Linear(int(self.binary_encoding_size), self.d_model)

        self.decoder_emb_type = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.encoder_emb_barbeat = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.decoder_emb_beat_density = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.decoder_emb_pitch = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.decoder_emb_duration = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.decoder_emb_velocity = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.decoder_emb_onset_density = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.decoder_pos_emb = BeatPositionalEncoding(self.d_model, self.dropout)

        # linear
        self.encoder_in_linear = nn.Linear(int(np.sum(self.emb_sizes[1])), self.d_model)
        self.decoder_in_linear = nn.Linear(int(np.sum(self.emb_sizes)), self.d_model)

        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layer_encoder,
            n_heads=self.n_head,
            query_dimensions=self.d_model // self.n_head,
            value_dimensions=self.d_model // self.n_head,
            feed_forward_dimensions=self.d_ff,
            activation='gelu',
            dropout=self.dropout,
            attention_type="linear",
        ).get()

        self.transformer_decoder = TransformerDecoderBuilder.from_kwargs(
            n_layers = self.n_layer_decoder,
            n_heads=self.n_head,
            query_dimensions=self.d_model // self.n_head,
            value_dimensions=self.d_model // self.n_head,
            feed_forward_dimensions=self.d_ff,
            activation='gelu',
            dropout=0.1,
            self_attention_type ="causal-linear",
            cross_attention_type = "linear",
        ).get()

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + self.emb_sizes[0], self.d_model)

        # individual output
        self.proj_type = nn.Linear(self.d_model, self.n_token[0])
        self.proj_barbeat = nn.Linear(self.d_model, self.n_token[1])
        self.proj_beat_density = nn.Linear(self.d_model, self.n_token[2])
        self.proj_pitch = nn.Linear(self.d_model, self.n_token[3])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[4])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[5])
        self.proj_onset_density = nn.Linear(self.d_model, self.n_token[6])

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def forward_hidden(self, en_x, de_x, disable_BE, disable_PE, is_training=True):
        '''
        en_x: music beats tokens
        de_x: drum track tokens
        '''
        # Beat embedding
        encoder_emb_barbeat = self.encoder_emb_barbeat(en_x[..., 0])
        encoder_emb_barbeat = self.encoder_in_linear(encoder_emb_barbeat)
        # Token embedding
        encoder_emb_binary = self.encoder_binary_emb(en_x[..., 1])
        encoder_emb_binary = self.encoder_binary_linear(encoder_emb_binary)
        if disable_BE:  # w/o Beat embedding
            encoder_emb_linear = encoder_emb_binary 
        else:
            encoder_emb_linear = encoder_emb_barbeat + encoder_emb_binary 
        if disable_PE:  # w/o Positional Encoding
            encoder_pos_emb = encoder_emb_linear
        else:
            pos = torch.arange(encoder_emb_linear.size(1), dtype=torch.long).cuda()
            pos = pos.unsqueeze(0)                          # [seq_len] -> [1, seq_len]
            pos = pos.repeat(encoder_emb_linear.size(0), 1) # [1, seq_len] -> [batch_size, seq_len]
            pos_emb = self.trainable_pos_emb(pos)           # [batch_size, seq_len, d_model]
            encoder_pos_emb = pos_emb + encoder_emb_linear  

            # encoder_pos_emb = self.encoder_pos_emb(encoder_emb_linear)  

        # transformer encoder
        encoder_hidden = self.transformer_encoder(encoder_pos_emb) 

        # decoder embeddings
        emb_type = self.decoder_emb_type(de_x[..., 0])
        emb_barbeat = self.encoder_emb_barbeat(de_x[..., 1])
        emb_beat_density = self.decoder_emb_beat_density(de_x[..., 2])
        emb_pitch = self.decoder_emb_pitch(de_x[..., 3])
        emb_duration = self.decoder_emb_duration(de_x[..., 4])
        emb_velocity = self.decoder_emb_velocity(de_x[..., 5])
        emb_onset_density = self.decoder_emb_onset_density(de_x[..., 6])

        embs = torch.cat(
            [
                emb_type,
                emb_barbeat,
                emb_beat_density,
                emb_pitch,
                emb_duration,
                emb_velocity,
                emb_onset_density
            ], dim=-1)

        decoder_emb_linear = self.decoder_in_linear(embs)
        decoder_pos_emb = self.decoder_pos_emb(decoder_emb_linear, de_x[:, :, 7])

        if is_training:
            attn_mask = TriangularCausalMask(decoder_pos_emb.size(1), device=de_x.device)
            decoder_hidden = self.transformer_decoder(decoder_pos_emb, encoder_hidden, x_mask=attn_mask)
            y_type = self.proj_type(decoder_hidden)
            return decoder_hidden, y_type
        else:
            decoder_mask = TriangularCausalMask(decoder_pos_emb.size(1), device=de_x.device)
            h = self.transformer_decoder(decoder_pos_emb, encoder_hidden, decoder_mask)
            h = h[:, -1:, :]
            h = h.squeeze(0)
            y_type = self.proj_type(h)

            return h, y_type

    def forward_output(self, h, y):
        '''
        h: forward hidden
        y: target (Ground Truth)
        '''
        # for training
        tf_skip_type = self.decoder_emb_type(y[..., 0])
        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        y_barbeat = self.proj_barbeat(y_)
        y_beat_density = self.proj_beat_density(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        y_onset_density = self.proj_onset_density(y_)

        return y_barbeat, y_pitch, y_duration, y_velocity, y_onset_density, y_beat_density

    def forward_output_sampling(self, h, y_type, recurrent=True):
        '''
        for inference
        '''
        y_type_logit = y_type[0, :]
        cur_word_type = sampling(y_type_logit, p=0.90, greedy=True) # greedy decoding

        type_word_t = torch.from_numpy(
            np.array([cur_word_type])).long().unsqueeze(0)

        if torch.cuda.is_available():
            type_word_t = type_word_t.cuda()

        tf_skip_type = self.decoder_emb_type(type_word_t).squeeze(0)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        # project other
        y_barbeat = self.proj_barbeat(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        y_onset_density = self.proj_onset_density(y_)
        y_beat_density = self.proj_beat_density(y_)

        # sampling gen_cond
        cur_word_barbeat = sampling(y_barbeat, t=1.2)
        cur_word_pitch = sampling(y_pitch, p=0.9)
        cur_word_duration = sampling(y_duration, t=2, p=0.9)
        cur_word_velocity = sampling(y_velocity, p=0.9)
        cur_word_onset_density = sampling(y_onset_density, p=0.90)
        cur_word_beat_density = sampling(y_beat_density, p=0.90)

        # collect
        next_arr = np.array([
            cur_word_type,
            cur_word_barbeat,
            cur_word_beat_density,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity,
            cur_word_onset_density,
        ])

        return next_arr
    
    def forward_encoder(self, en_x, disable_BE, disable_PE):
        '''
        en_x: music beats token
        '''
        # Beat embedding
        encoder_emb_barbeat = self.encoder_emb_barbeat(en_x[..., 0])
        encoder_emb_barbeat = self.encoder_in_linear(encoder_emb_barbeat)
        # Token embedding
        encoder_emb_binary = self.encoder_binary_emb(en_x[..., 1])
        encoder_emb_binary = self.encoder_binary_linear(encoder_emb_binary)
        # Beat embedding + Token embedding
        if disable_BE:  # w/o Beat embedding
            encoder_emb_linear = encoder_emb_binary 
        else:
            encoder_emb_linear = encoder_emb_barbeat + encoder_emb_binary 
        # Beat embedding + Token embedding + Positional Encoding
        if disable_PE:  # w/o Positional Encoding
            encoder_pos_emb = encoder_emb_linear
        else:
            pos = torch.arange(encoder_emb_linear.size(1), dtype=torch.long).cuda()
            pos = pos.unsqueeze(0)                          # [seq_len] -> [1, seq_len]
            pos = pos.repeat(encoder_emb_linear.size(0), 1) # [1, seq_len] -> [batch_size, seq_len]
            pos_emb = self.trainable_pos_emb(pos)           # [batch_size, seq_len, d_model]
            encoder_pos_emb = pos_emb + encoder_emb_linear  

            # encoder_pos_emb = self.encoder_pos_emb(encoder_emb_linear)

        # transformer encoder
        encoder_hidden = self.transformer_encoder(encoder_pos_emb)
        return encoder_hidden

    def forward_decoder(self, en_hidden, de_x, is_training=True):
        '''
        en_hidden: encoder hidden
        de_x: decoder input
        '''
        # decoder embeddings
        emb_type = self.decoder_emb_type(de_x[..., 0])
        emb_barbeat = self.encoder_emb_barbeat(de_x[..., 1]) 
        emb_beat_density = self.decoder_emb_beat_density(de_x[..., 2])
        emb_pitch = self.decoder_emb_pitch(de_x[..., 3])
        emb_duration = self.decoder_emb_duration(de_x[..., 4])
        emb_velocity = self.decoder_emb_velocity(de_x[..., 5])
        emb_onset_density = self.decoder_emb_onset_density(de_x[..., 6])

        embs = torch.cat(
            [
                emb_type,
                emb_barbeat,
                emb_beat_density,
                emb_pitch,
                emb_duration,
                emb_velocity,
                emb_onset_density
            ], dim=-1)

        decoder_emb_linear = self.decoder_in_linear(embs)
        decoder_pos_emb = self.decoder_pos_emb(decoder_emb_linear, de_x[:, :, 7])

        # transformer
        if is_training:
            attn_mask = TriangularCausalMask(decoder_pos_emb.size(1), device=de_x.device)
            decoder_hidden = self.transformer_decoder(decoder_pos_emb, en_hidden, x_mask=attn_mask) 
            y_type = self.proj_type(decoder_hidden)
            return decoder_hidden, y_type
        else:
            decoder_mask = TriangularCausalMask(decoder_pos_emb.size(1), device=de_x.device)
            h = self.transformer_decoder(decoder_pos_emb, en_hidden, decoder_mask)
            h = h[:, -1:, :]
            h = h.squeeze(0)
            y_type = self.proj_type(h)

            return h, y_type

    def inference_from_scratch(self, **kwargs):
        '''
        en_x: music beats tokens
        n_beat: time step threshold
        init_density: initial density value

        return: 
        final_res : music token list
        '''
        en_x =  kwargs['en_x']
        n_beat =  kwargs['n_beat']
        init_density = kwargs['init_density']
        disable_BE = kwargs['disable_BE']
        disable_PE = kwargs['disable_PE']

        dictionary = {'bar': 17}
        init = np.array([[1, 17, init_density, 0, 0, 0, 0, 0], ])   # initial input for decoder
        count = 1
        with torch.no_grad():
            final_res = []
            h = None
            y_type = None
            init_t = torch.from_numpy(init).long()
            if torch.cuda.is_available():
                init_t = init_t.cuda()
                en_x = en_x.cuda()
            en_hidden = self.forward_encoder(en_x, disable_BE, disable_PE)
            for step in range(init.shape[0]):
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                h, y_type = self.forward_decoder(en_hidden, input_, is_training=False)

            p_beat = 1
            cur_bar = 0
            cur_beat = 1
            all_beat = 1 
            while (True):
                next_arr = self.forward_output_sampling(h, y_type)
                if next_arr[0] == 2 and next_arr[3] == 0:  # type==Note and pitch==None
                    next_arr[3] = 1
                    print("warning note with instrument 0 detected, replaced by drum###################")
                if next_arr[1] == dictionary['bar']:        # beat pos==bar
                    cur_bar += 1
                if next_arr[0] == 1:                        # type=='M'
                    if next_arr[1] == 17:                   # beat pos==bar
                        cur_beat = 1
                    else:
                        cur_beat = next_arr[1]
                    all_beat = cur_bar * 16 + cur_beat - 1
                if all_beat >= n_beat:
                    break
                next_arr = np.concatenate([next_arr, [cur_bar * 16 + cur_beat - 1]])
                final_res.append(next_arr[None, ...])
                # forward
                input_cur = torch.from_numpy(next_arr).long().unsqueeze(0).unsqueeze(0)
                if torch.cuda.is_available():
                    input_cur = input_cur.cuda()
                input_ = torch.cat((input_, input_cur), dim=1)
                h, y_type = self.forward_decoder(en_hidden, input_, is_training=False)
                if next_arr[0] == 0:
                    print("EOS predicted")
                    break
                if len(final_res) >= 128:
                    break

        final_res = np.concatenate(final_res)
        return final_res


    def train_forward(self, **kwargs):
        en_x = kwargs['en_x']               # drum beats
        de_x = kwargs['de_x']               # input drum track tokens
        target = kwargs['target']           # target drum track tokens
        loss_mask = kwargs['loss_mask']     # loss mask
        disable_BE = kwargs['disable_BE']   # Whether to use Beat Embedding for drum beats
        disable_PE = kwargs['disable_PE']   # Whether to use Positional Encoding for drum beats

        h, y_type = self.forward_hidden(en_x, de_x, disable_BE, disable_PE, is_training=True)
        y_barbeat, y_pitch, y_duration, y_velocity, y_onset_density, y_beat_density = self.forward_output(h, target)

        # reshape (b, s, f) -> (b, f, s)
        y_barbeat = y_barbeat[:, ...].permute(0, 2, 1)
        y_type = y_type[:, ...].permute(0, 2, 1)
        y_pitch = y_pitch[:, ...].permute(0, 2, 1)
        y_duration = y_duration[:, ...].permute(0, 2, 1)
        y_velocity = y_velocity[:, ...].permute(0, 2, 1)
        y_onset_density = y_onset_density[:, ...].permute(0, 2, 1)
        y_beat_density = y_beat_density[:, ...].permute(0, 2, 1) 

        # loss
        loss_type = self.compute_loss(y_type, target[..., 0], loss_mask)
        loss_barbeat = self.compute_loss(y_barbeat, target[..., 1], loss_mask)
        loss_beat_density = self.compute_loss(y_beat_density, target[..., 2], loss_mask)
        loss_pitch = self.compute_loss(y_pitch, target[..., 3], loss_mask)
        loss_duration = self.compute_loss(y_duration, target[..., 4], loss_mask)
        loss_velocity = self.compute_loss(y_velocity, target[..., 5], loss_mask)
        loss_onset_density = self.compute_loss(y_onset_density, target[..., 6], loss_mask)

        return loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity, loss_onset_density, loss_beat_density

    def forward(self, **kwargs):
        if kwargs['is_train']:
            return self.train_forward(**kwargs)
        return self.inference_from_scratch(**kwargs)
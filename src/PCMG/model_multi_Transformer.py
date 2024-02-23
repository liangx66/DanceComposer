import numpy as np
import torch
import torch.cuda
from torch import nn

from utils import Embeddings, BeatPositionalEncoding

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


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


'''
last dimension of input data | attribute:
0: bar/beat
1: type
2: density (beat_density)
3: pitch
4: duration
5: instr_type
6: velocity
7: strength (onset_density)
8: p_beat
9: i_beat
'''
class MultitrackTransformer(nn.Module):
    def __init__(self, 
                 n_token, 
                 emb_sizes,
                 d_model=512,
                 num_encoder_layers=8,
                 num_decoder_layers=8,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 is_training=True):
        super(MultitrackTransformer, self).__init__()
        # params config
        self.n_token = n_token
        self.d_model = d_model
        self.n_layer_encoder = num_encoder_layers
        self.n_layer_decoder = num_decoder_layers 
        self.dropout = 0.1
        self.n_head = num_heads
        self.d_head = self.d_model // self.n_head
        self.d_ff = dim_feedforward
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.dropout = dropout
        self.emb_sizes = emb_sizes 
        self.time_encoding_size = 256

        # embedding
        self.encoder_emb_barbeat = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.encoder_emb_type = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.encoder_emb_beat_density = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.encoder_emb_pitch = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.encoder_emb_duration = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.encoder_emb_instr = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.encoder_emb_velocity = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.encoder_emb_onset_density = Embeddings(self.n_token[7], self.emb_sizes[7])
        self.encoder_emb_time_encoding = Embeddings(self.n_token[8], self.time_encoding_size)
        self.encoder_pos_emb = BeatPositionalEncoding(self.d_model, self.dropout)

        # linear
        self.encoder_in_linear = nn.Linear(int(np.sum(self.emb_sizes))+32, self.d_model)
        self.encoder_in_linear_wostyle = nn.Linear(int(np.sum(self.emb_sizes)), self.d_model)
        self.decoder_in_linear = nn.Linear(int(np.sum(self.emb_sizes)), self.d_model)
        self.encoder_time_linear = nn.Linear(int(self.time_encoding_size), self.d_model)
        self.style_linear = nn.Linear(32, self.d_model)

        # Transformer
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
            dropout=self.dropout,
            self_attention_type ="causal-linear",
            cross_attention_type = "linear",
        ).get()

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + self.emb_sizes[1], self.d_model)

        # individual output
        self.proj_barbeat = nn.Linear(self.d_model, self.n_token[0])
        self.proj_type = nn.Linear(self.d_model, self.n_token[1])
        self.proj_beat_density = nn.Linear(self.d_model, self.n_token[2])
        self.proj_pitch = nn.Linear(self.d_model, self.n_token[3])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[4])
        self.proj_instr = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])
        self.proj_onset_density = nn.Linear(self.d_model, self.n_token[7])

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def forward_hidden(self, en_x, de_x, fusion, style_embedding, is_training=True):
        '''
        en_x: drum track compound word tokens
        de_x: multi-track compound word tokens
        fusion: fusion embedding operation, must be one of the follow candidates
            'sum':    style embedding is added to the embedding of en_x
            'concat': style embedding is connected with the embedding of en_x
            'tile':   style embedding is tiled across every dimension of time in the embedding of en_x
        style_embedding: music style feature of multi-track
        '''
        # encoder embeddings
        en_emb_barbeat = self.encoder_emb_barbeat(en_x[..., 0])
        en_emb_type = self.encoder_emb_type(en_x[..., 1])
        en_emb_beat_density = self.encoder_emb_beat_density(en_x[..., 2])
        en_emb_pitch = self.encoder_emb_pitch(en_x[..., 3])
        en_emb_duration = self.encoder_emb_duration(en_x[..., 4])
        en_emb_instr = self.encoder_emb_instr(en_x[..., 5])
        en_emb_velocity = self.encoder_emb_velocity(en_x[..., 6])   
        en_emb_onset_density = self.encoder_emb_onset_density(en_x[..., 7])
        en_emb_time_encoding = self.encoder_emb_time_encoding(en_x[..., 8]) # embedding for p_beat, Beat-Timing Encoding(BTE)
        en_embs = torch.cat(
            [
                en_emb_barbeat,
                en_emb_type,
                en_emb_beat_density,
                en_emb_pitch,
                en_emb_duration,
                en_emb_instr,
                en_emb_velocity,
                en_emb_onset_density,
            ], dim=-1)
        if style_embedding is not None and fusion =='tile':
            style_embedding = style_embedding.unsqueeze(1)
            style_embedding = style_embedding.repeat(1, en_emb_barbeat.size(1), 1) 
            en_embs = torch.cat(
                [
                    en_embs,
                    style_embedding,
                ], dim=-1)
            encoder_emb_linear = self.encoder_in_linear(en_embs)
        else:
            encoder_emb_linear = self.encoder_in_linear_wostyle(en_embs)

        encoder_emb_time_linear = self.encoder_time_linear(en_emb_time_encoding)   # linear for BTE
        encoder_emb_linear = encoder_emb_linear + encoder_emb_time_linear
        if style_embedding is not None and fusion=='sum':
            style_embedding = self.style_linear(style_embedding)
            style_embedding = style_embedding.unsqueeze(1)
            encoder_emb_linear = encoder_emb_linear+style_embedding
        encoder_pos_emb = self.encoder_pos_emb(encoder_emb_linear, en_x[:, :, 9])   # Beat based Positional Encoding (BPE)
        if style_embedding is not None and fusion=='concat':
            style_embedding = self.style_linear(style_embedding)
            style_embedding = style_embedding.unsqueeze(1)
            encoder_pos_emb = torch.cat([encoder_pos_emb, style_embedding], dim = 1)
        # transformer encoder
        encoder_hidden = self.transformer_encoder(encoder_pos_emb)

        # decoder embeddings
        de_emb_barbeat = self.encoder_emb_barbeat(de_x[..., 0])
        de_emb_type = self.encoder_emb_type(de_x[..., 1])
        de_emb_beat_density = self.encoder_emb_beat_density(de_x[..., 2])
        de_emb_pitch = self.encoder_emb_pitch(de_x[..., 3])
        de_emb_duration = self.encoder_emb_duration(de_x[..., 4])
        de_emb_instr = self.encoder_emb_instr(de_x[..., 5])
        de_emb_velocity = self.encoder_emb_velocity(de_x[..., 6])
        de_emb_onset_density = self.encoder_emb_onset_density(de_x[..., 7])
        de_emb_time_encoding = self.encoder_emb_time_encoding(de_x[..., 8])
        de_embs = torch.cat(
            [
                de_emb_barbeat,
                de_emb_type,
                de_emb_beat_density,
                de_emb_pitch,
                de_emb_duration,
                de_emb_instr,
                de_emb_velocity,
                de_emb_onset_density
            ], dim=-1)

        decoder_emb_linear = self.decoder_in_linear(de_embs)
        decoder_emb_time_linear = self.encoder_time_linear(de_emb_time_encoding)
        decoder_emb_linear = decoder_emb_linear + decoder_emb_time_linear
        decoder_pos_emb = self.encoder_pos_emb(decoder_emb_linear, de_x[:, :, 9])

        # transformer decoder
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
        h: decoder_hidden
        y: target (Ground Truth)
        '''
        # for training
        tf_skip_type = self.encoder_emb_type(y[..., 1])
        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        y_barbeat = self.proj_barbeat(y_)
        y_beat_density = self.proj_beat_density(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_instr = self.proj_instr(y_)
        y_velocity = self.proj_velocity(y_)
        y_onset_density = self.proj_onset_density(y_)

        return y_barbeat, y_pitch, y_duration, y_instr, y_velocity, y_onset_density, y_beat_density

    def forward_output_sampling(self, h, y_type, recurrent=True):
        '''
        for inference
        '''
        y_type_logit = y_type[0, :]
        cur_word_type = sampling(y_type_logit, p=0.90)

        type_word_t = torch.from_numpy(
            np.array([cur_word_type])).long().unsqueeze(0)

        if torch.cuda.is_available():
            type_word_t = type_word_t.cuda()

        tf_skip_type = self.encoder_emb_type(type_word_t).squeeze(0)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        # project other
        y_barbeat = self.proj_barbeat(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_instr = self.proj_instr(y_)
        y_velocity = self.proj_velocity(y_)
        y_onset_density = self.proj_onset_density(y_)
        y_beat_density = self.proj_beat_density(y_)

        # sampling gen_cond
        cur_word_barbeat = sampling(y_barbeat, t=1.2)
        cur_word_pitch = sampling(y_pitch, p=0.9)
        cur_word_duration = sampling(y_duration, t=2, p=0.9)
        cur_word_instr = sampling(y_instr, p=0.90)
        cur_word_velocity = sampling(y_velocity, p=0.9)
        cur_word_onset_density = sampling(y_onset_density, p=0.90)
        cur_word_beat_density = sampling(y_beat_density, p=0.90)

        # collect
        next_arr = np.array([
            cur_word_barbeat,
            cur_word_type,
            cur_word_beat_density,
            cur_word_pitch,
            cur_word_duration,
            cur_word_instr,
            cur_word_velocity,
            cur_word_onset_density,
        ])
        return next_arr
    
    def forward_encoder(self, en_x, fusion, style_embedding):
        # encoder embeddings
        en_emb_barbeat = self.encoder_emb_barbeat(en_x[..., 0])
        en_emb_type = self.encoder_emb_type(en_x[..., 1])
        en_emb_beat_density = self.encoder_emb_beat_density(en_x[..., 2])
        en_emb_pitch = self.encoder_emb_pitch(en_x[..., 3])
        en_emb_duration = self.encoder_emb_duration(en_x[..., 4])
        en_emb_instr = self.encoder_emb_instr(en_x[..., 5])
        en_emb_velocity = self.encoder_emb_velocity(en_x[..., 6])
        en_emb_onset_density = self.encoder_emb_onset_density(en_x[..., 7])
        en_emb_time_encoding = self.encoder_emb_time_encoding(en_x[..., 8])
        en_embs = torch.cat(
            [
                en_emb_barbeat,
                en_emb_type,
                en_emb_beat_density,
                en_emb_pitch,
                en_emb_duration,
                en_emb_instr,
                en_emb_velocity,
                en_emb_onset_density,
            ], dim=-1)
        if style_embedding is not None and fusion =='tile':
            style_embedding = style_embedding.unsqueeze(1)
            style_embedding = style_embedding.repeat(1, en_emb_barbeat.size(1), 1)
            en_embs = torch.cat(
                [
                    en_embs,
                    style_embedding,
                ], dim=-1)
            encoder_emb_linear = self.encoder_in_linear(en_embs)
        else:
            encoder_emb_linear = self.encoder_in_linear_wostyle(en_embs)

        encoder_emb_time_linear = self.encoder_time_linear(en_emb_time_encoding)   # beat-timing encoding
        encoder_emb_linear = encoder_emb_linear + encoder_emb_time_linear
        if style_embedding is not None and fusion=='sum':
            style_embedding = self.style_linear(style_embedding)
            style_embedding = style_embedding.unsqueeze(1)
            encoder_emb_linear = encoder_emb_linear+style_embedding
        
        encoder_pos_emb = self.encoder_pos_emb(encoder_emb_linear, en_x[:, :, 9])
        if style_embedding is not None and fusion=='concat':
            style_embedding = self.style_linear(style_embedding)
            style_embedding = style_embedding.unsqueeze(1)
            encoder_pos_emb = torch.cat([encoder_pos_emb, style_embedding], dim = 1)
        # transformer encoder
        encoder_hidden = self.transformer_encoder(encoder_pos_emb)
        return encoder_hidden

    def forward_decoder(self, en_hidden, de_x, is_training=True):
        '''
        en_hidden: encoder hidden state
        de_x: input multi-track tokens
        '''
        # decoder embeddings
        de_emb_barbeat = self.encoder_emb_barbeat(de_x[..., 0])
        de_emb_type = self.encoder_emb_type(de_x[..., 1])
        de_emb_beat_density = self.encoder_emb_beat_density(de_x[..., 2])
        de_emb_pitch = self.encoder_emb_pitch(de_x[..., 3])
        de_emb_duration = self.encoder_emb_duration(de_x[..., 4])
        de_emb_instr = self.encoder_emb_instr(de_x[..., 5])
        de_emb_velocity = self.encoder_emb_velocity(de_x[..., 6])
        de_emb_onset_density = self.encoder_emb_onset_density(de_x[..., 7])
        de_emb_time_encoding = self.encoder_emb_time_encoding(de_x[..., 8])

        de_embs = torch.cat(
            [
                de_emb_barbeat,
                de_emb_type,
                de_emb_beat_density,
                de_emb_pitch,
                de_emb_duration,
                de_emb_instr,
                de_emb_velocity,
                de_emb_onset_density
            ], dim=-1)

        decoder_emb_linear = self.decoder_in_linear(de_embs)
        decoder_emb_time_linear = self.encoder_time_linear(de_emb_time_encoding)
        decoder_emb_linear = decoder_emb_linear + decoder_emb_time_linear
        decoder_pos_emb = self.encoder_pos_emb(decoder_emb_linear, de_x[:, :, 9])

        # transformer decoder
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
        en_x: drum track tokens
        n_beat: time step threshold, length of generated music
        fusion: fusion embedding operation, must be one of the follow candidates
            'sum':    style embedding is added to the embedding of en_x
            'concat': style embedding is connected with the embedding of en_x
            'tile':   style embedding is tiled across every dimension of time in the embedding of en_x
        style_embedding:  music style feature of multi-track

        return: 
        final_res : music token list
        '''
        en_x =  kwargs['en_x'] 
        n_beat =  kwargs['n_beat']
        fusion = kwargs['fusion']
        style_embedding = kwargs['style_embedding']

        def get_p_beat(cur_bar, cur_beat, n_beat):
            all_beat = cur_bar * 16 + cur_beat - 1  
            p_beat = round(all_beat / n_beat * 100)+1 
            return p_beat

        dictionary = {'bar': 17}
        init_density = np.random.randint(1,18)
        # initial token for encoder
        init = np.array([[17, 1, init_density, 0, 0, 0, 0, 0, 1, 0], ])
        count = 1
        with torch.no_grad():
            final_res = []
            h = None
            y_type = None
            init_t = torch.from_numpy(init).long()
            if torch.cuda.is_available():
                init_t = init_t.cuda()
                en_x = en_x.cuda()
                if style_embedding is not None:
                    style_embedding = style_embedding.cuda()
            en_hidden = self.forward_encoder(en_x, fusion, style_embedding) 
            
            for step in range(init.shape[0]):
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                h, y_type = self.forward_decoder(en_hidden, input_, is_training=False)

            p_beat = 1
            cur_bar = 0
            cur_beat = 1
            while (True):
                next_arr = self.forward_output_sampling(h, y_type)
                # next_arr = np.array([
                #     cur_word_barbeat,
                #     cur_word_type,
                #     cur_word_beat_density,
                #     cur_word_pitch,
                #     cur_word_duration,
                #     cur_word_instr,
                #     cur_word_velocity,
                #     cur_word_onset_density,])
                if next_arr[1] == 2 and next_arr[5] == 0:   # type==Note and instr==None
                    next_arr[5] = 1     # replaced by drum                   
                if next_arr[0] == dictionary['bar']:        # barbeat==bar, new bar
                    cur_bar += 1
                if next_arr[1] == 1:                        # type==Rhythm 
                    if next_arr[0] == 17:                   # barbeat==bar, new bar
                        cur_beat = 1
                    else:
                        cur_beat = next_arr[0]
                    p_beat = get_p_beat(cur_bar, cur_beat, n_beat)
                if p_beat >= 102:       # exceed max p_beat
                    break
                next_arr = np.concatenate([next_arr, [p_beat], [cur_bar * 16 + cur_beat - 1]])
                final_res.append(next_arr[None, ...])
                count+=1
                # forward
                input_cur = torch.from_numpy(next_arr).long().unsqueeze(0).unsqueeze(0)
                if torch.cuda.is_available():
                    input_cur = input_cur.cuda()
                input_ = torch.cat((input_, input_cur), dim=1)
                h, y_type = self.forward_decoder(en_hidden, input_, is_training=False)
                if next_arr[1] == 0:        # EOS   
                    break
                # if len(final_res) >= 999:
                #     break
  
        final_res = np.concatenate(final_res)
        return final_res


    def train_forward(self, **kwargs):
        en_x = kwargs['en_x']           # drum track tokens
        de_x = kwargs['de_x']           # input multi-track tokens
        target = kwargs['target']       # target multi-track tokens
        loss_mask = kwargs['loss_mask'] # loss mask
        fusion = kwargs['fusion']       # fusion embedding operation
        style_embedding = kwargs['style_embedding'] # music style features
        h, y_type = self.forward_hidden(en_x, de_x, fusion, style_embedding, is_training=True) # init_token=init_token)   # h [B, 999, 512] y_type [B, 999, 3]
        y_barbeat, y_pitch, y_duration, y_instr, y_velocity, y_onset_density, y_beat_density = self.forward_output(h, target)

        # reshape (b, s, f) -> (b, f, s)
        y_barbeat = y_barbeat[:, ...].permute(0, 2, 1)
        y_type = y_type[:, ...].permute(0, 2, 1)
        y_pitch = y_pitch[:, ...].permute(0, 2, 1)
        y_duration = y_duration[:, ...].permute(0, 2, 1)
        y_instr = y_instr[:, ...].permute(0, 2, 1)
        y_velocity = y_velocity[:, ...].permute(0, 2, 1)
        y_onset_density = y_onset_density[:, ...].permute(0, 2, 1)
        y_beat_density = y_beat_density[:, ...].permute(0, 2, 1)

        # loss
        loss_barbeat = self.compute_loss(y_barbeat, target[..., 0], loss_mask)
        loss_type = self.compute_loss(y_type, target[..., 1], loss_mask)
        loss_beat_density = self.compute_loss(y_beat_density, target[..., 2], loss_mask)
        loss_pitch = self.compute_loss(y_pitch, target[..., 3], loss_mask)
        loss_duration = self.compute_loss(y_duration, target[..., 4], loss_mask)
        loss_instr = self.compute_loss(y_instr, target[..., 5], loss_mask)
        loss_velocity = self.compute_loss(y_velocity, target[..., 6], loss_mask)
        loss_onset_density = self.compute_loss(y_onset_density, target[..., 7], loss_mask)

        return loss_barbeat, loss_type, loss_pitch, loss_duration, loss_instr, loss_velocity, loss_onset_density, loss_beat_density

    def forward(self, **kwargs):
        if kwargs['is_train']:
            return self.train_forward(**kwargs)
        return self.inference_from_scratch(**kwargs)
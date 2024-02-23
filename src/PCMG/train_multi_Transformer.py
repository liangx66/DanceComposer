'''
train multi-track Transformer
dataset: LPD
'''
import sys
import datetime
import argparse
import os
import time

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")

import utils
from utils import log, Saver, network_paras
from model_multi_Transformer import MultitrackTransformer
from dataset_multi import CPDataset

def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_dp():
    parser = argparse.ArgumentParser(description="Args for training Multi-track Transformer")
    parser.add_argument('-n', '--name', default="debug",
                        help="Name of the experiment, also the log file and checkpoint directory. If 'debug', checkpoints won't be saved")
    parser.add_argument('-g', '--gpus', type=int, nargs='+', help="Ids of gpu")

    #load, input, save configurations:
    parser.add_argument('-t', '--train_data', default='../../data/lpd_5_cleansed/train_data.npz',
                        help="Path of the training data (.npz file)")
    parser.add_argument('-v', '--val_data', default='../../data/lpd_5_cleansed/val_data.npz',
                        help="Path of the val data (.npz file)")
    parser.add_argument('-log_dir', '--log_dir', type=str, default='../../logs/multi_Transformer', help="log_dir")
    parser.add_argument('-p', '--path', default= None, help="If set, load model from the given path")
    parser.add_argument('-ckpt_path', '--ckpt_path', type=str, default= "../../ckpt/multi_Transformer", 
                        help="path to save the ckpt")
    parser.add_argument('-embedding_path', '--embedding_path', type=str, default= "../../data/lpd_cleansed/lpd_music_style_embedding", 
                        help="path of the music style embedding")

    #optimization hyper parameters:
    parser.add_argument('-l', '--lr', default=0.00001, help="Initial learning rate")
    parser.add_argument('-b', '--batch_size', default=16, help="Batch size")
    parser.add_argument('-e', '--epochs', default=200, help="Num of epochs")
    parser.add_argument('-wd', '--wd', default=0.0001, help="weight decay")
    parser.add_argument('-weight_v', '--weight_v', type=float, default=0.2, help="d_model")
    parser.add_argument('-weight_p', '--weight_p', type=float, default=5, help="d_model")
    parser.add_argument('-weight_t', '--weight_t', type=float, default=1, help="d_model")
    parser.add_argument('-patience_epoch', '--patience_epoch', default=30, help="early stop for How long to wait after last time validation loss improved")

    # model parameters:
    parser.add_argument('-f', '--fusion', type=str, default='tile', 
                        help="fusion embedding operation, if args.embedding_path is not None, USE this arg to choose the way of combining style embedding and drum track")
    parser.add_argument('-emb_sizes', '--emb_sizes', default=[32, 8, 32, 512, 32, 16, 32, 32], help="d_model")
    parser.add_argument('-d_model', '--d_model', default=512, help="d_model")
    parser.add_argument('-num_encoder_layers', '--num_encoder_layers', default=8, help="num_encoder_layers")
    parser.add_argument('-num_decoder_layers', '--num_decoder_layers', default=8, help="num_decoder_layers")
    parser.add_argument('-num_heads', '--num_heads', default=8, help="num heads")
    parser.add_argument('-dim_feedforward', '--dim_feedforward', default=2048, help="dim_feedforward")
    args = parser.parse_args()
    if args.gpus is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in list(range(torch.cuda.device_count()))])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in args.gpus])
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    init_lr = float(args.lr)
    batch_size = int(args.batch_size)
    n_epoch = int(args.epochs)
    wd = float(args.wd)
    weight_velocity = args.weight_v
    weight_pitch = args.weight_p
    weight_type = args.weight_t
    patience_epoch = int(args.patience_epoch)
    DEBUG = args.name == "debug"
    params = {
        "DECAY_EPOCH": [],
        "DECAY_RATIO": 0.1,
    }
    writer = SummaryWriter(log_dir=args.log_dir)

    # hyper params
    max_grad_norm = 3

    log("name:", args.name)
    log("args", args)
    if DEBUG:
        log("DEBUG MODE checkpoints will not be saved")
    else:
        utils.flog = open("../logs/" + args.name + ".log", "w")

    #### create data loader ####
    train_set = CPDataset(npz_file_path = args.train_data, embedding_path = args.embedding_path, split = 'train')
    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers=16, shuffle=True)
    decoder_n_class = train_set.get_decoder_n_class()   
    val_set = CPDataset(npz_file_path = args.val_data, embedding_path = args.embedding_path, split = 'val')
    val_loader = DataLoader(val_set, batch_size = 1, shuffle=False)
    log("decoder_n_class", decoder_n_class)
    log('emb_sizes:', args.emb_sizes)

    # create saver
    saver_agent = Saver(exp_dir="../exp/" + args.name, debug=DEBUG)

    #### create the model ######
    emb_sizes = args.emb_sizes
    d_model = int(args.d_model)
    num_encoder_layers = int(args.num_encoder_layers)
    num_decoder_layers = int(args.num_decoder_layers)
    num_heads = int(args.num_heads)
    dim_feedforward = int(args.dim_feedforward)

    net = MultitrackTransformer(decoder_n_class,
              emb_sizes = emb_sizes,
              d_model=d_model,
              num_encoder_layers=num_encoder_layers,
              num_decoder_layers=num_decoder_layers,
              num_heads=num_heads,
              dim_feedforward=dim_feedforward,
              dropout=0.1,)
    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
        net.cuda()
    DEVICE_COUNT = torch.cuda.device_count()
    log("DEVICE COUNT:", DEVICE_COUNT)
    log("VISIBLE: " + os.environ["CUDA_VISIBLE_DEVICES"])
    n_parameters = network_paras(net)
    log('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(' > params amount: {:,d}'.format(n_parameters))
    if args.path is not None:
        print('[*] load model from:', args.path)
        net.load_state_dict(torch.load(args.path))

    #### create optimizer #####
    optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=wd)

    #### start training ###
    for epoch in range(1, n_epoch+1):
        start_time = time.time()
        epoch_loss = 0
        epoch_losses = np.zeros(8)
        print(epoch)
        net.train()

        if epoch in params['DECAY_EPOCH']:
            log('LR decay by ratio', params['DECAY_RATIO'])
            for p in optimizer.param_groups:
                p['lr'] *= params['DECAY_RATIO']

        for i, (en_x, de_x, de_y, de_mask, style_embedding) in enumerate(train_loader):
            saver_agent.global_step_increment()

            en_x = en_x.cuda(non_blocking=True)         # drum track tokens
            de_x = de_x.cuda(non_blocking=True)         # input multi-track tokens
            de_y = de_y.cuda(non_blocking=True)         # target multi-track tokens
            de_mask = de_mask.cuda(non_blocking=True)   # loss mask
            style_embedding = style_embedding.cuda(non_blocking=True)       # music style features

            losses = net(is_train=True, en_x=en_x, de_x=de_x, target=de_y, loss_mask=de_mask, fusion=args.fusion, style_embedding=style_embedding)
            losses = [l.sum() for l in losses]
            # loss_barbeat, loss_type, loss_pitch, loss_duration, loss_instr, loss_velocity, loss_onset_density, loss_beat_density
            loss = (losses[0] + weight_type*losses[1] + weight_pitch*losses[2] + losses[3] + losses[4] + weight_velocity*losses[5] + losses[6] + losses[7]) / 8

            # Update
            net.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                epoch_losses += np.array([l.item() for l in losses])
                epoch_loss += loss.item()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        epoch_loss = epoch_loss / len(train_loader)
        epoch_losses = epoch_losses / len(train_loader)
        print('Epoch: {} '.format(epoch))
        print(time.ctime() + ' Train epoch: [{}]/[{}] | Loss: {:.9f} | Time: {}m {}s'.format(epoch, n_epoch, epoch_loss, epoch_mins, epoch_secs))
        losses_str = 'barbeat {:.6f}, type {:.6f}, pitch {:.6f}, duration {:.6f}, instr {:.6f}, velocity {:.6f}, strength {:.6f}, density {:.6f}\r'.format(
            epoch_losses[0], epoch_losses[1], epoch_losses[2], epoch_losses[3], epoch_losses[4], epoch_losses[5], epoch_losses[6], epoch_losses[7])
        print('losses | ' + losses_str)
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/barbeat_loss", epoch_losses[0], epoch)
        writer.add_scalar("train/type_loss", epoch_losses[1], epoch)
        writer.add_scalar("train/pitch_loss", epoch_losses[2], epoch)
        writer.add_scalar("train/duration_loss", epoch_losses[3], epoch)
        writer.add_scalar("train/instr_loss", epoch_losses[4], epoch)
        writer.add_scalar("train/velocity_loss", epoch_losses[5], epoch)
        writer.add_scalar("train/strength_loss", epoch_losses[6], epoch)
        writer.add_scalar("train/density_loss", epoch_losses[7], epoch)

        # validation        
        net.eval()
        val_epoch_loss = 0
        val_epoch_losses = np.zeros(8)
        with torch.no_grad():
            for i, (val_en_x, val_de_x, val_de_y, val_de_mask, style_embedding) in enumerate(val_loader):
                val_en_x = val_en_x.cuda(non_blocking=True)
                val_de_x = val_de_x.cuda(non_blocking=True)
                val_de_y = val_de_y.cuda(non_blocking=True)
                val_de_mask = val_de_mask.cuda(non_blocking=True)
                style_embedding = style_embedding.cuda(non_blocking=True)
                losses = net(is_train=True, en_x=val_en_x, de_x=val_de_x, target=val_de_y, loss_mask=val_de_mask, fusion=args.fusion, style_embedding=style_embedding)
                losses = [l.sum() for l in losses]
                loss = (losses[0] + weight_type*losses[1] + weight_pitch*losses[2] + losses[3] + losses[4] + weight_velocity*losses[5] + losses[6] + losses[7]) / 8
                val_epoch_losses += np.array([l.item() for l in losses])
                val_epoch_loss += loss.item()
        val_epoch_loss = val_epoch_loss / len(val_loader)
        val_epoch_losses = val_epoch_losses / len(val_loader) 
        print('Val epoch: [{}]/[{}] | Loss: {:.9f} '.format(epoch, n_epoch, val_epoch_loss))
        val_losses_str = 'barbeat {:.6f}, type {:.6f}, pitch {:.6f}, duration {:.6f}, instr {:.6f}, velocity {:.6f}, strength {:.6f}, density {:.6f}\r'.format(
            val_epoch_losses[0], val_epoch_losses[1], val_epoch_losses[2], val_epoch_losses[3], val_epoch_losses[4], val_epoch_losses[5], val_epoch_losses[6], val_epoch_losses[7])
        print('Val losses | ' + val_losses_str)
        writer.add_scalar("val/loss", val_epoch_loss, epoch)
        writer.add_scalar("val/barbeat_loss", val_epoch_losses[0], epoch)
        writer.add_scalar("val/type_loss", val_epoch_losses[1], epoch)
        writer.add_scalar("val/pitch_loss", val_epoch_losses[2], epoch)
        writer.add_scalar("val/duration_loss", val_epoch_losses[3], epoch)
        writer.add_scalar("val/instr_loss", val_epoch_losses[4], epoch)
        writer.add_scalar("val/velocity_loss", val_epoch_losses[5], epoch)
        writer.add_scalar("val/strength_loss", val_epoch_losses[6], epoch)
        writer.add_scalar("val/density_loss", val_epoch_losses[7], epoch)
        log('-' * 100)

        if not DEBUG:   # DEBUG MODE checkpoints will not be saved
            ckpt_path = args.ckpt_path
            if not os.path.exists(ckpt_path):
                os.mkdir(ckpt_path)
            ckpt_name = 'model_'+str(epoch)+'.pt'
            ckpt = os.path.join(ckpt_path, ckpt_name)
            torch.save(net.module.state_dict(), ckpt)

    writer.close()    

    
if __name__ == '__main__':
    train_dp()

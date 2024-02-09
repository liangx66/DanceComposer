'''
train MBPN
'''

import sys
import datetime
import argparse
import os
import time

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pprint import pprint
import numpy as np
from dataset import AISTDataset
from val import compute_accuracy, compute_f1_score
from music_transformer import music_transformer_dev_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="Args for training")
    parser.add_argument('-n', '--name', default="debug",
                        help="Name of the experiment, also the log file and checkpoint directory. If 'debug', checkpoints won't be saved")
    parser.add_argument('-g', '--gpus', type=int, nargs='+', help="Ids of gpu")

    #load, input, save configurations:
    parser.add_argument('-t', '--train_data', default= '../data/AIST/train_MBPN.txt',
                        help="Path of the training data")
    parser.add_argument('-v', '--val_data', default='../data/AIST/val_MBPN.txt',
                        help="Path of the val data")
    parser.add_argument('-music_beat_path', '--music_beat_path', default='../data/AIST/music_beat',
                        help="Path of the music beat data")
    parser.add_argument('-p', '--path', help="If set, load model from the given path")
    parser.add_argument('-ckpt_path', '--ckpt_path', type=str, default="../ckpt/MBPN", help="ckpt")
    parser.add_argument('-log_dir', '--log_dir', type=str, default='../logs/MBPN', help="log_dir")
    
    #optimization hyper parameters:
    parser.add_argument('-lr', '--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Num Epochs")
    parser.add_argument('-wd', '--wd', default=0.0001, help="weight decay")
    parser.add_argument('-patience_epoch', '--patience_epoch', default=1, help="early stop for How long to wait after last time validation loss improved")

    # model parameters:
    parser.add_argument('-d', '--emb_dim', type=int, default=256, help="Emb Dim")
    parser.add_argument('-nel', '--num_encoder_layers', type=int, default=4, help="num encoder layers")
    parser.add_argument('-ndl', '--num_decoder_layers', type=int, default=0, help="num decoder layers")
    parser.add_argument('--rpr', '--rpr', action='store_true', default=True, help="rpr")
    parser.add_argument('-dms', '--decoder_max_seq', type=int, default=512, help="decoder max seq")
    parser.add_argument('-pnl', '--pose_net_layers', type=int, default=10, help="pose net layers")
    parser.add_argument('-num_heads', '--num_heads', type=int, default=2, help="num_heads")
    
    # dataset parameters:
    parser.add_argument('-duration', '--duration', type=float, default=12.0, help="duration")
    parser.add_argument('-fps', '--fps', type=float, default=60.0, help="fps")
    parser.add_argument('-pose_layout', '--pose_layout', type=str, default="body25", help="pose_layout")
    args = parser.parse_args()
    if args.gpus is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in list(range(torch.cuda.device_count()))])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in args.gpus])
    
    return args


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train():
    args = parse_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    print('=' * 100)
    pprint(args)
    print('=' * 100)

    init_lr = float(args.lr)
    batch_size = int(args.batch_size)
    n_epoch = int(args.epochs)
    wd = float(args.wd)
    patience_epoch = int(args.patience_epoch)
    DEBUG = args.name == "debug"

    writer = SummaryWriter(log_dir=args.log_dir)
    weight = torch.Tensor([0.1, 1]).cuda(non_blocking=True)     # non beat: 0.1, beat: 1 
    train_criterion = nn.CrossEntropyLoss(weight = weight)
    val_criterion = nn.CrossEntropyLoss(weight = weight)

    #### create data loader ####
    train_set = AISTDataset(motion_files = args.train_data, music_beat_path = args.music_beat_path, duration = args.duration, fps = args.fps, split = 'train')
    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers=16, shuffle=True)
    val_set = AISTDataset(motion_files = args.val_data, music_beat_path = args.music_beat_path, duration = args.duration, fps = args.fps, split = 'val')
    val_loader = DataLoader(val_set, batch_size = 1, shuffle=False)   
    print("train set:", len(train_set))
    print("val set:", len(val_set))

    #### create the model ######
    model: nn.Module = music_transformer_dev_baseline(
        2,
        d_model = args.emb_dim,
        dim_feedforward = args.emb_dim * 4,
        encoder_max_seq = int(args.duration * args.fps),
        decoder_max_seq = args.decoder_max_seq,
        layout = args.pose_layout,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        rpr = args.rpr,
        use_control = False,
        rnn = None,
        layers = args.pose_net_layers,
        num_heads = args.num_heads,
        dropout = 0.1
    ).cuda()
    model = nn.DataParallel(model)

    #### create optimizer #####
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=wd, betas=(0.9, 0.98), eps=1e-9)

    best_val_f1 = None
    best_val_epoch = 0
    for epoch in range(1, n_epoch + 1):
        print(epoch)
        start_time = time.time()
        ###### train ######
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        for i, (music_beats, motion, _) in enumerate(train_loader):
            music_beats = music_beats.cuda(non_blocking=True)
            motion = motion.cuda(non_blocking=True) 
            output = model(motion)      # [B, T, D]
            """
                For CrossEntropy
                output: [B, T, D] -> [BT, D]
                target: [B, T] -> [BT]
            """
            loss = train_criterion(output.view(-1, output.shape[-1]), music_beats.flatten())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss += loss.item()
                output = F.softmax(output, dim=-1)
                output = torch.argmax(output, dim=-1) 
                output = output.flatten()
                music_beats = music_beats.flatten()
                acc = compute_accuracy(output, music_beats)
                epoch_acc += acc.item()
                f1_score = compute_f1_score(output, music_beats)
                epoch_f1 += f1_score
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = epoch_acc / len(train_loader)
        epoch_f1 = epoch_f1 / len(train_loader)
        print('Epoch: {} '.format(epoch))
        print(time.ctime() + ' Train epoch: [{}]/[{}] | Loss: {:.9f} | Time: {}m {}s'.format(epoch, n_epoch, epoch_loss, epoch_mins, epoch_secs))
        print('acc:{:.9f}, f1:{:.9f}'.format(epoch_acc, epoch_f1))
        # update tensorboard #
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/acc", epoch_acc, epoch)
        writer.add_scalar("train/f1", epoch_f1, epoch)

        ####### val #######
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        val_epoch_f1 = 0
        with torch.no_grad():
            for i, (music_beats, motion, _) in enumerate(val_loader):
                music_beats = music_beats.cuda(non_blocking=True)
                motion = motion.cuda(non_blocking=True)
                output = model(motion)
                loss = val_criterion(output.view(-1, output.shape[-1]), music_beats.flatten())
                val_epoch_loss += loss.item()
                output = F.softmax(output, dim=-1)
                output = torch.argmax(output, dim=-1)
                output = output.flatten()
                music_beats = music_beats.flatten()
                acc = compute_accuracy(output, music_beats)
                val_epoch_acc += acc.item()
                f1_score = compute_f1_score(output, music_beats)
                val_epoch_f1 += f1_score.item()
        val_epoch_loss = val_epoch_loss/len(val_loader)
        val_epoch_acc = val_epoch_acc/len(val_loader)
        val_epoch_f1 = val_epoch_f1/len(val_loader)
        print('Val epoch:[{}]/[{}] | Loss: {:.9f}, acc:{:.9f}, f1:{:.9f}'.format(epoch, n_epoch, val_epoch_loss, val_epoch_acc, val_epoch_f1))
        # update tensorboard #
        writer.add_scalar("val/loss", val_epoch_loss, epoch)
        writer.add_scalar("val/acc", val_epoch_acc, epoch)
        writer.add_scalar("val/f1", val_epoch_f1, epoch)

        if not DEBUG: 
            ckpt_path = args.ckpt_path
            if not os.path.exists(ckpt_path):
                os.mkdir(ckpt_path)
            ckpt_name = 'model_'+str(epoch)+'.pt'
            ckpt = os.path.join(ckpt_path, ckpt_name)
            torch.save(model.module.state_dict(), ckpt)
    writer.close()


if __name__ == '__main__':
    train()

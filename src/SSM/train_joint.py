'''
fine-tune SSM: train dance and music embedding networks jointly.
dataset: AIST
'''
import sys
import datetime
import argparse
import os
import time

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
# from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")

import utils
from utils import Saver, network_paras, CustomSchedule
from dataset import JointDataset
from model_joint import JointModel
from val import compute_accuracy


def log(*args, **kwargs):
    print(*args, **kwargs)


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_dp():
    parser = argparse.ArgumentParser(description="Args for training music classifier")
    parser.add_argument('-n', '--name', default="train_exp",     # default="train_exp",
                        help="Name of the experiment, also the log file and checkpoint directory. If 'debug', checkpoints won't be saved")
    parser.add_argument('-g', '--gpus', type=int, nargs='+', help="Ids of gpu")

    # model parameters:
    parser.add_argument('-duration', '--duration', type=float, default=12.0, help="duration")
    parser.add_argument('-fps', '--fps', type=float, default=30.0, help="fps")
    parser.add_argument('-pose_layout', '--pose_layout', type=str, default="body25", help="pose_layout")

    #load, input, save configurations:
    parser.add_argument('-t', '--train_data', default='../../data/AIST/train_SSM_joint.txt',
                        help="Path of the training data (.npy file)")
    parser.add_argument('-v', '--val_data', default='../../data/AIST/val_SSM_joint.txt',
                        help="Path of the val data (.npy file)")
    parser.add_argument('-music_ckpt', '--music_ckpt', type=str, default= "../../ckpt/music_network/model_100.pt",
                        help="If set, load the music encoder model from the given path")
    parser.add_argument('-dance_ckpt', '--dance_ckpt', type=str, default= "../../ckpt/dance_network/model_100.pt", 
                        help="If set, load the dance encoder model from the given path")                    
    parser.add_argument('-log_dir', '--log_dir', type=str, default='../../logs/joint_train_finetune', help="log_dir")
    parser.add_argument('-ckpt_path', '--ckpt_path', type=str, default= "../../ckpt/joint_model_finetune", help="path to save the ckpt")
    
    #optimization hyper parameters:
    parser.add_argument('-l', '--lr', default=0.00001, help="Initial learning rate")
    parser.add_argument('-b', '--batch_size', default=4, help="Batch size")
    parser.add_argument('-e', '--epochs', default=200, help="Num of epochs")
    parser.add_argument('-wd', '--wd', default=0.0001, help="weight decay")
    parser.add_argument('-patience_epoch', '--patience_epoch', default=30, help="early stop for How long to wait after last time validation loss improved")
    parser.add_argument('-w_f', '--weight_f', type=float, default=1.5, help='mse loss weight')
    parser.add_argument('-w_m', '--weight_m', type=float, default=1.0, help='cross entropy loss weight for music classify')
    parser.add_argument('-w_d', '--weight_d', type=float, default=1.0, help='cross entropy loss weight for dance classify')

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
    DEBUG = args.name == "debug"
    writer = SummaryWriter(log_dir=args.log_dir)
    m_ckpt = str(args.music_ckpt)
    d_ckpt = str(args.dance_ckpt)
    wd = float(args.wd)
    patience_epoch = int(args.patience_epoch)

    print("name:", args.name)
    print("args", args)

    #### create data loader ####
    train_set = JointDataset(motion_files=args.train_data, music_path = '../../data/AIST/mp3', duration=args.duration, fps=args.fps, split='train')
    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers=4, shuffle=True)
    val_set = JointDataset(motion_files =args.val_data, music_path = '../../data/AIST/mp3', duration=args.duration, fps=args.fps,split='val')
    val_loader = DataLoader(val_set, batch_size = 1, shuffle=False)
    print("len train data:", len(train_set))
    print("len val data:", len(val_set))

    #### create the model ######
    net = JointModel(m_ckpt, d_ckpt, args.pose_layout)
    DEVICE_COUNT = torch.cuda.device_count()
    print("DEVICE COUNT:", DEVICE_COUNT)
    print("VISIBLE: " + os.environ["CUDA_VISIBLE_DEVICES"])
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))

    # loss
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    #### create optimizer #####
    optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=wd)

    #### start training ###
    for epoch in range(1, n_epoch+1):
        running_loss_f = 0
        running_loss_m = 0
        running_loss_d = 0
        total_loss = 0.0
        epoch_acc_m = 0
        epoch_acc_d = 0
        print(epoch)
        start_time = time.time()
        net.train()
        for i, (dance, melgram, label) in enumerate(train_loader):
            melgram = melgram.cuda(non_blocking=True)      # [N, C, F, L] 
            dance = dance.cuda(non_blocking=True)          # [N, C, T, V, 1]
            label = label.cuda(non_blocking=True)          # [N, 1]
            m_pred, m_feature, d_pred, d_feature = net(melgram, dance)          
            loss_m = cross_entropy_loss(m_pred, label.flatten())
            loss_d = cross_entropy_loss(d_pred, label.flatten())
            loss_f = mse_loss(d_feature, m_feature)
            # Total loss
            loss = args.weight_f * loss_f + args.weight_m * loss_m + args.weight_d * loss_d
            # Update
            net.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss_f += loss_f.item()
                running_loss_m += loss_m.item()
                running_loss_d += loss_d.item()
                total_loss += loss.item()
                # acc
                m_pred = F.softmax(m_pred, dim=-1)
                m_pred = torch.argmax(m_pred, dim=1)
                m_pred = m_pred.flatten()
                label = label.flatten()
                acc_m = compute_accuracy(m_pred, label)
                epoch_acc_m += acc_m.item()

                d_pred = F.softmax(d_pred, dim=-1)
                d_pred = torch.argmax(d_pred, dim=1)
                d_pred = d_pred.flatten()
                acc_d = compute_accuracy(d_pred, label)
                epoch_acc_d += acc_d.item()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        running_loss_f /= len(train_loader)
        running_loss_m /= len(train_loader)
        running_loss_d /= len(train_loader)
        total_loss /= len(train_loader)
        epoch_acc_m /= len(train_loader)
        epoch_acc_d /= len(train_loader)
        print('Epoch: {} '.format(epoch))
        print(time.ctime() + ' Train epoch: [{}]/[{}] | Total Loss: {:.9f} | Time: {}m {}s'.format(epoch, n_epoch, total_loss, epoch_mins, epoch_secs))
        print('MSE Loss: {:.9f} | Music network Loss: {:.9f} | Dance network Loss: {:.9f}'.format(running_loss_f, running_loss_m, running_loss_d))
        print('Music classify acc: {:.9f} | Dance classify acc: {:.9f}'.format(epoch_acc_m, epoch_acc_d))
        writer.add_scalar("train/loss", total_loss, epoch)
        writer.add_scalar("train/loss_f", running_loss_f, epoch)
        writer.add_scalar("train/loss_m", running_loss_m, epoch)
        writer.add_scalar("train/loss_d", running_loss_d, epoch)
        writer.add_scalar("train/acc_m", epoch_acc_m, epoch)
        writer.add_scalar("train/acc_d", epoch_acc_d, epoch)

        ####### val #######
        net.eval()
        val_running_loss_f = 0
        val_running_loss_m = 0
        val_running_loss_d = 0
        val_total_loss = 0.0
        val_epoch_acc_m = 0
        val_epoch_acc_d = 0
        with torch.no_grad():
            for i, (dance, melgram, label) in enumerate(val_loader):
                melgram = melgram.cuda(non_blocking=True)      # [N, C, F, L] 
                dance = dance.cuda(non_blocking=True)          # [N, C, T, V, 1]
                label = label.cuda(non_blocking=True)          # [N, 1]
                m_pred, m_feature, d_pred, d_feature = net(melgram, dance) 
                loss_m = cross_entropy_loss(m_pred, label.flatten())
                loss_d = cross_entropy_loss(d_pred, label.flatten())
                loss_f = mse_loss(d_feature, m_feature)
                # Total loss
                loss = args.weight_f * loss_f + args.weight_m * loss_m + args.weight_d * loss_d
                val_running_loss_f += loss_f.item()
                val_running_loss_m += loss_m.item()
                val_running_loss_d += loss_d.item()
                val_total_loss += loss.item()
                # acc
                m_pred = F.softmax(m_pred, dim=-1)
                m_pred = torch.argmax(m_pred, dim=1)
                m_pred = m_pred.flatten()
                label = label.flatten()
                acc_m = compute_accuracy(m_pred, label)
                val_epoch_acc_m += acc_m.item()

                d_pred = F.softmax(d_pred, dim=-1)
                d_pred = torch.argmax(d_pred, dim=1)
                d_pred = d_pred.flatten()
                acc_d = compute_accuracy(d_pred, label)
                val_epoch_acc_d += acc_d.item()

        val_total_loss = val_total_loss / len(val_loader)  # !!!!小心变量名错误   # running_loss / num_batch 
        val_running_loss_f /= len(val_loader)
        val_running_loss_m /= len(val_loader)
        val_running_loss_d /= len(val_loader)
        val_epoch_acc_m /= len(val_loader)
        val_epoch_acc_d /= len(val_loader)
        print('Val epoch: [{}]/[{}] | Loss: {:.9f}  '.format(epoch, n_epoch, val_total_loss))
        print('MSE Loss: {:.9f} | Music network Loss: {:.9f} | Dance network Loss: {:.9f}'.format(val_running_loss_f, val_running_loss_m, val_running_loss_d))
        print('Music classify acc: {:.9f} | Dance classify acc: {:.9f}'.format(val_epoch_acc_m, val_epoch_acc_d))
        writer.add_scalar("val/loss", val_total_loss, epoch)
        writer.add_scalar("val/loss_f", val_running_loss_f, epoch)
        writer.add_scalar("val/loss_m", val_running_loss_m, epoch)
        writer.add_scalar("val/loss_d", val_running_loss_d, epoch)
        writer.add_scalar("val/acc_m", val_epoch_acc_m, epoch)
        writer.add_scalar("val/acc_d", val_epoch_acc_d, epoch)
        print('-' * 100)

        if not DEBUG:   # DEBUG MODE checkpoints will not be saved
            ckpt_path = args.ckpt_path
            if not os.path.exists(ckpt_path):
                os.mkdir(ckpt_path)
            ckpt_name = 'model_'+str(epoch)+'.pt'
            ckpt = os.path.join(ckpt_path, ckpt_name)
            torch.save(net.state_dict(), ckpt)
    writer.close()    


if __name__ == '__main__':
    train_dp()

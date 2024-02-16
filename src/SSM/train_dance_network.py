'''
train dance style embedding network independently
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
from dataset import MotionDataset
from model_dance_network import ST_GCN_18
from val import compute_accuracy


def log(*args, **kwargs):
    print(*args, **kwargs)


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_dp():
    parser = argparse.ArgumentParser(description="Args for training")
    parser.add_argument('-n', '--name', default="debug",
                        help="Name of the experiment, also the log file and checkpoint directory. If 'debug', checkpoints won't be saved")
    parser.add_argument('-g', '--gpus', type=int, nargs='+', help="Ids of gpu")

    # load, input, save configurations:
    parser.add_argument('-t', '--train_data', default='../../data/AIST/train_dance_network.txt',
                        help="Path of the training data (.npy file)")
    parser.add_argument('-v', '--val_data', default='../../data/AIST/val_dance_network.txt',
                        help="Path of the val data (.npy file)")
    parser.add_argument('-log_dir', '--log_dir', type=str, default='../../logs/dance_network', help="log_dir")
    parser.add_argument('-p', '--path', default= None, help="If set, load model from the given path")
    parser.add_argument('-ckpt_path', '--ckpt_path', type=str, default= "../../ckpt/dance_network", help="path to save the ckpt")

    # optimization hyper parameters:
    parser.add_argument('-l', '--lr', default=0.001, help="Initial learning rate")
    parser.add_argument('-b', '--batch_size', default=16, help="Batch size")
    parser.add_argument('-e', '--epochs', default=200, help="Num of epochs")
    parser.add_argument('-wd', '--wd', default=0.001, help="weight decay")
    parser.add_argument('-patience_epoch', '--patience_epoch', default=30, help="early stop for How long to wait after last time validation loss improved")

    # model parameters:
    parser.add_argument('-duration', '--duration', type=float, default=12.0, help="duration")
    parser.add_argument('-fps', '--fps', type=float, default=30.0, help="fps")
    parser.add_argument('-pose_layout', '--pose_layout', type=str, default="body25", help="pose_layout")
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
    patience_epoch = int(args.patience_epoch)
    DEBUG = args.name == "debug"
    writer = SummaryWriter(log_dir=args.log_dir)

    # hyper params
    # max_grad_norm = 3

    print("name:", args.name)
    print("args", args)

    #### create data loader ####
    train_set = MotionDataset(motion_files=args.train_data, duration=args.duration, fps=args.fps, split='train')
    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers=16, shuffle=True)
    val_set = MotionDataset(motion_files=args.val_data, duration=args.duration, fps=args.fps,split='val')
    val_loader = DataLoader(val_set, batch_size = 1, shuffle=False)
    print("len train data:", len(train_set))
    print("len val data:", len(val_set))

    #### create the model ######
    graph_cfg = {
        "layout": args.pose_layout,
        "strategy": "spatial"
    }
    net = ST_GCN_18(
        in_channels=2,
        num_class = 10,
        num_layer = 4,
        graph_cfg = graph_cfg,
        dropout=0.1,
    )
    if torch.cuda.is_available():
        net.cuda()
    DEVICE_COUNT = torch.cuda.device_count()
    print("DEVICE COUNT:", DEVICE_COUNT)
    print("VISIBLE: " + os.environ["CUDA_VISIBLE_DEVICES"])
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))
    # saver_agent.add_summary_msg(' > params amount: {:,d}'.format(n_parameters))
    if args.path is not None:
        print('[*] load model from:', args.path)
        net.load_state_dict(torch.load(args.path))
    # loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizers
    #### create optimizer #####
    optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=wd)

    #### start training ###
    for epoch in range(1, n_epoch+1):
        running_loss = 0
        epoch_acc = 0
        print(epoch)
        start_time = time.time()
        net.train()
        for i, (x, y, _) in enumerate(train_loader):
            x = x.cuda(non_blocking=True)          # [N, C, T, V, 1] 
            y = y.cuda(non_blocking=True)
            y_pred = net(x)
            loss = loss_fn(y_pred, y.flatten())
            # Update
            net.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                y_pred = F.softmax(y_pred, dim=-1)
                y_pred = torch.argmax(y_pred, dim=1)
                y_pred = y_pred.flatten()
                y = y.flatten()
                acc = compute_accuracy(y_pred, y)
                epoch_acc += acc.item()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = epoch_acc / len(train_loader)

        print('Epoch: {} '.format(epoch))
        print(time.ctime() + ' Train epoch: [{}]/[{}] | Loss: {:.3f} | acc: {:.3f} | Time: {}m {}s'.format(epoch, n_epoch, epoch_loss, epoch_acc, epoch_mins, epoch_secs))
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/acc", epoch_acc, epoch)

        ####### val #######
        net.eval()
        val_running_loss = 0
        val_epoch_acc = 0
        with torch.no_grad():
            for i, (x, y, _) in enumerate(val_loader):
                x = x.cuda(non_blocking=True)          # [4, C, H, W]
                y = y.cuda(non_blocking=True)          # [4, 1]
                y_pred = net(x) # [4, 10]
                loss = loss_fn(y_pred, y.flatten())
                val_running_loss += loss.item()
                y_pred = F.softmax(y_pred, dim=-1)
                y_pred = torch.argmax(y_pred, dim=1)
                y_pred = y_pred.flatten()
                y = y.flatten()
                acc = compute_accuracy(y_pred, y)
                val_epoch_acc += acc.item()
        val_epoch_loss = val_running_loss / len(val_loader)  # !!!!小心变量名错误   # running_loss / num_batch 
        val_epoch_acc = val_epoch_acc / len(val_loader)
        print('Val epoch: [{}]/[{}] | Loss: {:.3f} | acc: {:.3f} '.format(epoch, n_epoch, val_epoch_loss, val_epoch_acc))
        writer.add_scalar("val/loss", val_epoch_loss, epoch)
        writer.add_scalar("val/acc", val_epoch_acc, epoch)
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

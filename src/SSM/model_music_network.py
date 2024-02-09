'''
MusicTaggerCRNN model 
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
import numpy as np


class music_encoder(nn.Module):
    def __init__(self, 
                num_class,
                 dropout=0.4,
                ):
        super().__init__()
        self.lin1 = nn.Linear(128, 32)
        self.lin2 = nn.Linear(32, num_class)
        
        self.conv_block1 = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout, inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            # Conv block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout(dropout, inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            # Conv block 3
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(dropout, inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            # Conv block 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(dropout, inplace=True),            
        )
        self.gru1 = nn.GRU(input_size=128, hidden_size=256, num_layers=1,dropout=dropout)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128, num_layers=1,dropout=dropout)
        # self.dropout = nn.Dropout(0.3, inplace=True), 


    def forward(self, x):
        ##### conv
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        N,C_input_size,H,W_seq_len = x.size()
        if H == 1:
            x = x.permute(3,0,1,2)
            x = x.reshape((W_seq_len, N, C_input_size))
            
        ##### GRU
        h0 =  torch.randn(1, N, 256).cuda()
        x, _ = self.gru1(x, h0)
        h1 =  torch.randn(1, N, 128).cuda()
        x, _ = self.gru2(x, h1) # (seq_len, batch, num_directions * hidden_size) 
       
        ##### FC
        out = x[-1,:,:]
        out = self.lin1(out)
        # out = self.dropout(out)
        out = torch.sigmoid(out)    
        out = self.lin2(out)
        return out

    def extract_feature(self, x):
        ##### conv
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        N,C_input_size,H,W_seq_len = x.size()
        if H == 1:
            x = x.permute(3,0,1,2)
            x = x.reshape((W_seq_len, N, C_input_size))
            
        ##### GRU
        h0 =  torch.randn(1, N, 256).cuda()
        x, _ = self.gru1(x, h0)
        h1 =  torch.randn(1, N, 128).cuda() 
        x, _ = self.gru2(x, h1)
        # x = self.dropout(x)
       
        ##### FC
        out = x[-1,:,:]
        feature = self.lin1(out)
        out = torch.sigmoid(feature)    
        out = self.lin2(out)
        return out, feature


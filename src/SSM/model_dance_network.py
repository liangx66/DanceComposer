'''
Motion classification model 
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
import numpy as np

from graph import Graph 
from gconv_origin import ConvTemporalGraphical

class ST_GCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 num_layer,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 dropout=0.1,    # 0.1
                #  encoder_max_seq = 360,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(
            in_channels * A.size(1)) if data_bn else lambda x: x

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(num_layer)
            ])
        else:
            self.edge_importance = [1] * num_layer

        self.dropout = dropout
        self.st_gcn_block1 = st_gcn_block(in_channels, 64, kernel_size, stride=2, residual=False, dropout = self.dropout)
        self.st_gcn_block2 = st_gcn_block(64, 128, kernel_size, stride=2, dropout = self.dropout)
        self.st_gcn_block3 = st_gcn_block(128, 128, kernel_size, stride=2, dropout = self.dropout)
        self.st_gcn_block4 = st_gcn_block(128, 128, kernel_size, stride=2, dropout = self.dropout)
        self.gru1 = nn.GRU(input_size=128, hidden_size=256, num_layers=1)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128, num_layers=1)
        self.lin1 = nn.Linear(128, 32)
        self.lin2 = nn.Linear(32, self.num_class)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()   # [N, C, T, V, M]->[N, M, V, C, T]
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # [N, M, V, C, T]->[N, M, C, T, V]
        x = x.view(N * M, C, T, V)                  # [N, C, T, V]

        # forwad
        ##### conv
        # ST-GCN block 1
        x,_ = self.st_gcn_block1(x, self.A*self.edge_importance[0])
        # ST-GCN block 2
        x,_ = self.st_gcn_block2(x, self.A*self.edge_importance[1])
        # ST-GCN block 3
        x,_ = self.st_gcn_block3(x, self.A*self.edge_importance[2])
        # ST-GCN block 4
        x,_ = self.st_gcn_block4(x, self.A*self.edge_importance[3])

        x = F.avg_pool2d(x, (1, V))                 # [NM, C, T, V] -> [NM, C, T, 1]
        
        ##### GRU
        N,C_input_size,W_seq_len,H, = x.size()
        if H == 1:
            x = x.permute(2,0,1,3)
            x = x.reshape((W_seq_len, N, C_input_size))
        h0 =  torch.randn(1, N, 256).cuda()
        x, _ = self.gru1(x, h0)
        h1 =  torch.randn(1, N, 128).cuda()
        x, _ = self.gru2(x, h1)
        
        ##### FC
        out = x[-1,:,:]
        out = self.lin1(out) 
        out = torch.sigmoid(out)
        out = self.lin2(out)
        return out

    def extract_feature(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        # ST-GCN block 1
        x,_ = self.st_gcn_block1(x, self.A*self.edge_importance[0])
        # ST-GCN block 2
        x,_ = self.st_gcn_block2(x, self.A*self.edge_importance[1])
        # ST-GCN block 3
        x,_ = self.st_gcn_block3(x, self.A*self.edge_importance[2])
        # ST-GCN block 4
        x,_ = self.st_gcn_block4(x, self.A*self.edge_importance[3])

        x = F.avg_pool2d(x, (1, V))

        ##### GRU
        N,C_input_size,W_seq_len,H, = x.size()
        if H == 1:
            x = x.permute(2,0,1,3)
            x = x.reshape((W_seq_len, N, C_input_size))
        h0 =  torch.randn(1, N, 256).cuda()
        x, _ = self.gru1(x, h0)
        h1 =  torch.randn(1, N, 128).cuda()
        x, _ = self.gru2(x, h1) 

        ##### FC
        out = x[-1,:,:]
        feature = self.lin1(out)
        out = torch.sigmoid(feature)
        out = self.lin2(out)

        return out, feature



class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels, 
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.drop_path_prob = 0.0


    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
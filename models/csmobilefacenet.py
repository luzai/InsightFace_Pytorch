import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import ConvBlock, ConvDepthWiseBlock, InvertedResidual_as12_3


class CSMobileFaceNet2(nn.Module):
    def __init__(self, embedding_size):
        super(CSMobileFaceNet2, self).__init__()
        self.activation = 'PReLU'
        self.conv1 = ConvBlock(3, 66, kernel=3, stride=2, padding=1, activation=self.activation)
        self.dw_conv1 = ConvDepthWiseBlock(66, 3, 1, 1, activation=self.activation)
        
        inverted_residual_parameter_list = [
            # t:expansion factor, c:kernel number/output channel, n:repeat times, s:stride
            # t, c, n, s
            (2, 66, 5, 2),
            (4, 132, 1, 2),
            (2, 132, 6, 1),
            (4, 132, 1, 2),
            (2, 132, 2, 1)
        ]
        self.in_c = 66
        block = InvertedResidual_as12_3
        self.bottleneck_block = self._make_layer(block, inverted_residual_parameter_list)
        
        self.conv2 = ConvBlock(132, 512, 1, 1, 0, activation=self.activation)
        self.linear_gdw_conv = ConvDepthWiseBlock(512, kernel=7, stride=1, padding=0, activation='Linear')
        # self.linear_conv = ConvBlock(512, 128, 1, 1, 0, activation='Linear')
        
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.bottleneck_block(x)
        x = self.conv2(x)
        x = self.linear_gdw_conv(x)
        # x = self.linear_conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)
        return x
    
    def _make_layer(self, block, parameter_list):
        layers = []
        for t, c, n, s in parameter_list:
            out_c = c
            for i in range(n):
                if i == 0:
                    layers.append(block(self.in_c, out_c, s, t, activation=self.activation))
                else:
                    layers.append(block(out_c, out_c, 1, t, activation=self.activation))
            self.in_c = out_c
        return nn.Sequential(*layers)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    # conv -> BN -> active function
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=0, groups=1,
                 bias=False, activation='ReLU'):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, stride=stride,
                              padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        if not activation == 'Linear':
            if activation == 'PReLU':
                self.af = nn.PReLU(out_c)
            elif activation == 'ReLU6':
                self.af = nn.ReLU6(inplace=True)
            else:
                self.af = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if not self.activation == 'Linear':
            x = self.af(x)
        return x


class ConvLinearBlock(nn.Module):
    # conv -> BN
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=0, groups=1, bias=False):
        super(ConvLinearBlock, self).__init__()
        self.linearconv = ConvBlock(in_c, out_c, kernel=kernel, stride=stride,
                                    padding=padding, groups=groups, bias=bias, activation='Linear')
    
    def forward(self, x):
        out = self.linearconv(x)
        return out


class ConvDepthWiseBlock(nn.Module):
    def __init__(self, in_c, kernel=3, stride=1, padding=1, bias=False, activation='ReLU'):
        super(ConvDepthWiseBlock, self).__init__()
        self.convdw = ConvBlock(in_c=in_c, out_c=in_c, kernel=kernel, stride=stride,
                                padding=padding, groups=in_c, bias=bias, activation=activation)
    
    def forward(self, x):
        out = self.convdw(x)
        return out


class ConvPointWiseBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, padding=0, groups=1, bias=False, activation='RuLU'):
        super(ConvPointWiseBlock, self).__init__()
        self.convpw = ConvBlock(in_c=in_c, out_c=out_c, kernel=1, stride=stride,
                                padding=padding, groups=groups, bias=bias, activation=activation)
    
    def forward(self, x):
        out = self.convpw(x)
        return out


class ConvDSBlock(nn.Module):
    # depthwise separable convolution
    def __init__(self, in_c, out_c, stride=1, activation='ReLU'):
        super(ConvDSBlock, self).__init__()
        self.convdw = ConvDepthWiseBlock(in_c=in_c, kernel=3, stride=stride, activation=activation)
        self.convpw = ConvPointWiseBlock(in_c=in_c, out_c=out_c, stride=1,
                                         padding=0, groups=1, activation=activation)
    
    def forward(self, x):
        out = self.convdw(x)
        out = self.convpw(out)
        return out


class InvertedResidual(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
    
    def forward(self, x):
        if self.use_connect:
            x = x + self.conv_block(x)
        else:
            x = self.conv_block(x)
        return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_c, out_c, stride, groups, activation='ReLU', use_group=True):
        super(ShuffleUnit, self).__init__()
        self.bottleneck_channels = out_c // 4
        self.stride = stride
        self.groups = groups
        self.first_layer_groups = groups if use_group else 1
        
        self.group_conv1 = ConvPointWiseBlock(in_c, self.bottleneck_channels,
                                              groups=self.first_layer_groups, activation=activation)
        self.dw_conv = ConvDepthWiseBlock(self.bottleneck_channels, kernel=3,
                                          stride=self.stride, activation='Linear')
        self.group_conv2 = ConvPointWiseBlock(self.bottleneck_channels, out_c, groups=groups, activation='Linear')
        self.average_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x1 = x
        
        out = self.group_conv1(x)
        out = channel_shuffle(out, self.groups)
        out = self.dw_conv(out)
        out = self.group_conv2(out)
        
        if self.stride == 2:
            x1 = self.average_pool(x1)
            out = torch.cat((x1, out), 1)
        
        elif self.stride == 1:
            out = out + x
        
        return F.relu(out)


def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.data.size()
    channels_per_group = channels // groups
    
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    
    # transpose
    torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batch_size, -1, height, width)
    
    return x


def channel_concatenate(x, out):
    return torch.cat((x, out), 1)


class ShuffleUnitv2(nn.Module):
    def __init__(self, in_c, out_c, stride, activation='ReLU'):
        super(ShuffleUnitv2, self).__init__()
        self.stride = stride
        
        out_half_c = out_c // 2
        if self.stride == 1:
            self.unit1 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
        elif self.stride == 2:
            self.unit2_branch1 = nn.Sequential(
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation),
                ConvDepthWiseBlock(out_half_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        else:
            print("error stride!")
    
    def forward(self, x):
        if self.stride == 1:
            x_first_half = x[:, :(x.shape[1] // 2), :, :]
            x_last_half = x[:, (x.shape[1] // 2):, :, :]
            out = channel_concatenate(x_first_half, self.unit1(x_last_half))
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch1(x), self.unit2_branch2(x))
        else:
            print("error stride!")
        return channel_shuffle(out, 2)


class InvertedResidual_as1(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as1, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        self.conv_part_c = out_c - in_c
        
        if self.stride == 2:
            out_c = self.conv_part_c
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        if self.use_connect:
            x = x + self.conv_block(x)
        elif self.stride == 2:
            banch1 = self.maxpool(x)
            # banch1 = self.avgpool(x)
            banch2 = self.conv_block(x)
            x = channel_concatenate(banch1, banch2)
        else:
            x = self.conv_block(x)
        return x


class InvertedResidual_as2(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as2, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 2
            in_c = in_c // 2
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_half = x[:, :(x.shape[1] // 2), :, :]
            x_last_half = x[:, (x.shape[1] // 2):, :, :]
            out = channel_concatenate(x_first_half, self.conv_block(x_last_half))
            out = channel_shuffle(out, 2)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as3(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as3, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        self.conv_part_c = out_c - in_c
        
        if self.stride == 2:
            out_c = self.conv_part_c
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        if self.use_connect:
            x = x + self.conv_block(x)
        elif self.stride == 2:
            banch1 = self.maxpool(x)
            # banch1 = self.avgpool(x)
            banch2 = self.conv_block(x)
            x = channel_concatenate(banch1, banch2)
            x = channel_shuffle(x, 2)
        else:
            x = self.conv_block(x)
        return x


class InvertedResidual_as4(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as4, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
        self.se = SEBlock(out_c)
    
    def forward(self, x):
        if self.use_connect:
            x = x + self.se(self.conv_block(x))
        else:
            x = self.se(self.conv_block(x))
        return x


class SEBlock(nn.Module):
    def __init__(self, in_c, reduction=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_c, in_c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // reduction, in_c, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)


class InvertedResidual_as5(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as5, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        self.out_c = out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 2
            in_c = in_c // 2
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.out_c // 2 >= 16:
            self.se = SEBlock(out_half_c)
    
    def forward(self, x):
        if self.use_connect:
            x_first_half = x[:, :(x.shape[1] // 2), :, :]
            x_last_half = x[:, (x.shape[1] // 2):, :, :]
            out = self.conv_block(x_last_half)
            if (self.out_c // 2) >= 16:
                out = self.se(out)
            out = channel_concatenate(x_first_half, out)
            out = channel_shuffle(out, 2)
        elif self.stride == 2:
            out = self.conv_block(x)
            if (self.out_c // 2) >= 16:
                out = self.se(out)
            out = channel_concatenate(self.se(self.unit2_branch2(x)), out)
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
            if (self.out_c // 2) >= 16:
                out = self.se(out)
        
        return out


class InvertedResidual_as5_3(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as5_3, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        self.out_c = out_c
        
        # as5_4:
        # if self.out_c // 2 >= 16:
        #     self.reduction = 16
        # else:   
        #     self.reduction = 8
        
        self.reduction = 8
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 2
            in_c = in_c // 2
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.out_c // 2 >= self.reduction:
            self.se = SEBlock(out_half_c, self.reduction)
    
    def forward(self, x):
        if self.use_connect:
            x_first_half = x[:, :(x.shape[1] // 2), :, :]
            x_last_half = x[:, (x.shape[1] // 2):, :, :]
            out = self.conv_block(x_last_half)
            if (self.out_c // 2) >= self.reduction:
                out = self.se(out)
            out = channel_concatenate(x_first_half, out)
            out = channel_shuffle(out, 2)
        elif self.stride == 2:
            out = self.conv_block(x)
            if (self.out_c // 2) >= self.reduction:
                out = self.se(out)
            out = channel_concatenate(self.se(self.unit2_branch2(x)), out)
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
            if (self.out_c // 2) >= self.reduction:
                out = self.se(out)
        
        return out


class InvertedResidual_as6(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as6, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        self.padding = 0 if self.stride == 1 else 1
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                ConvDepthWiseBlock_as6(hidden_c, stride=(stride, 1), kernel=(3, 1), activation=activation),
                # ConvPointWiseBlock(hidden_c, hidden_c, activation=activation),
                ConvDepthWiseBlock_as6(hidden_c, stride=(1, stride), kernel=(1, 3), activation=activation, padding=0),
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock_as6(hidden_c, stride=(stride, 1), kernel=(3, 1), activation=activation),
                # ConvPointWiseBlock(hidden_c, hidden_c, activation=activation),
                ConvDepthWiseBlock_as6(hidden_c, stride=(1, stride), kernel=(1, 3), activation=activation, padding=0),
                ConvPointWiseBlock(hidden_c, out_c, activation='Linear')
            )
    
    def forward(self, x):
        if self.use_connect:
            x = x + self.conv_block(x)
        else:
            x = self.conv_block(x)
        return x


class ConvBlock_as6(nn.Module):
    # conv -> BN -> active function
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(1, 1), padding=0, groups=1,
                 bias=False, activation='ReLU'):
        super(ConvBlock_as6, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, stride=stride,
                              padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        if not activation == 'Linear':
            if activation == 'PReLU':
                self.af = nn.PReLU(out_c)
            elif activation == 'ReLU6':
                self.af = nn.ReLU6(inplace=True)
            else:
                self.af = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if not self.activation == 'Linear':
            x = self.af(x)
        return x


class ConvDepthWiseBlock_as6(nn.Module):
    def __init__(self, in_c, kernel=(3, 3), stride=(1, 1), padding=1, bias=False, activation='ReLU'):
        super(ConvDepthWiseBlock_as6, self).__init__()
        self.convdw = ConvBlock(in_c=in_c, out_c=in_c, kernel=kernel, stride=stride,
                                padding=padding, groups=in_c, bias=bias, activation=activation)
    
    def forward(self, x):
        out = self.convdw(x)
        return out


class InvertedResidual_as8(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as8, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            out = channel_concatenate(self.branch3(x_first_part), self.conv_block(x_second_part))
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 4)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as2_2(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as2_2, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        self.out_c = out_c
        self.in_c = in_c
        if self.use_connect:
            out_half_c = out_c // 2
            in_half_c = in_c // 2
        elif self.stride == 2:
            out_half_c = out_c // 2
            in_half_c = in_c // 2
        else:
            if in_c < out_c:
                out_half_c = in_c
                in_half_c = in_c
                self.remain_c = out_c - in_c
            else:
                out_half_c = out_c
                in_half_c = in_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_half_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_half_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_half_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_half_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_half = x[:, :(x.shape[1] // 2), :, :]
            x_last_half = x[:, (x.shape[1] // 2):, :, :]
            out = channel_concatenate(x_first_half, self.conv_block(x_last_half))
            out = channel_shuffle(out, 2)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            if self.out_c > self.in_c:
                out = channel_concatenate(self.unit3_branch2(x), self.conv_block(x))
                out = channel_shuffle(out, 2)
            else:
                out = self.conv_block(x)
        return out


class InvertedResidual_as8_2(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as8_2, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            out = channel_concatenate(self.conv_block(x_first_part), self.conv_block(x_second_part))
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 4)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as8_3(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as8_3, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            out = channel_concatenate(self.branch3(x_first_part), self.branch3(x_second_part))
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 4)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as8_4(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as8_4, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            out = channel_concatenate(self.branch3(x_first_part), self.conv_block(x_second_part))
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 3)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as9(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as9, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            out = channel_concatenate(self.branch3(x_first_part), self.conv_block(x_second_part))
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 4)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as9_2(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as9_2, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation=activation),
                ConvDepthWiseBlock(out_half_c, stride=1, activation=activation),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            out = channel_concatenate(self.branch3(x_first_part), self.conv_block(x_second_part))
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 4)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as10(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as10, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 4
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
            self.branch4 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 4), :, :]
            x_second_part = x[:, (x.shape[1] // 4):(x.shape[1] // 4) * 2, :, :]
            x_third_part = x[:, (x.shape[1] // 4) * 2:(x.shape[1] // 4) * 3, :, :]
            x_last_part = x[:, (x.shape[1] // 4) * 3:, :, :]
            out = channel_concatenate(self.branch4(x_first_part), self.conv_block(x_second_part))
            out = channel_concatenate(out, self.branch3(x_third_part))
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 4)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as11(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as11, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        self.out_c = out_c
        
        self.reduction = 16
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 2
            in_c = in_c // 2
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.out_c >= self.reduction:
            self.se = SEBlock(self.out_c, self.reduction)
    
    def forward(self, x):
        if self.use_connect:
            x_first_half = x[:, :(x.shape[1] // 2), :, :]
            x_last_half = x[:, (x.shape[1] // 2):, :, :]
            out = self.conv_block(x_last_half)
            out = channel_concatenate(x_first_half, out)
            if self.out_c >= self.reduction:
                out = self.se(out)
            out = channel_shuffle(out, 2)
        elif self.stride == 2:
            out = self.conv_block(x)
            out = channel_concatenate(self.unit2_branch2(x), out)
            if self.out_c >= self.reduction:
                out = self.se(out)
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
            if (self.out_c // 2) >= self.reduction:
                out = self.se(out)
        
        return out


class InvertedResidual_as11_5(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as11_5, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        self.out_c = out_c
        
        self.reduction = 8
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 2
            in_c = in_c // 2
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.out_c >= self.reduction:
            self.se = SEBlock(self.out_c, self.reduction)
    
    def forward(self, x):
        if self.use_connect:
            x_first_half = x[:, :(x.shape[1] // 2), :, :]
            x_last_half = x[:, (x.shape[1] // 2):, :, :]
            out = self.conv_block(x_last_half)
            out = channel_concatenate(x_first_half, out)
            if self.out_c >= self.reduction:
                out = self.se(out)
            out = channel_shuffle(out, 2)
        elif self.stride == 2:
            out = self.conv_block(x)
            out = channel_concatenate(self.unit2_branch2(x), out)
            if self.out_c >= self.reduction:
                out = self.se(out)
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
            if (self.out_c // 2) >= self.reduction:
                out = self.se(out)
        
        return out


class InvertedResidual_as13(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as13, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
            self.branch2 = nn.Sequential(
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            out = channel_concatenate(self.branch3(x_first_part), self.conv_block(x_second_part))
            out = channel_concatenate(out, self.branch2(x_last_part))
            out = channel_shuffle(out, 3)
        elif self.stride == 2:
            out = channel_concatenate(self.unit2_branch2(x), self.conv_block(x))
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
        return out


class InvertedResidual_as12(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as12, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        self.out_half_c = out_half_c
        self.reduction = 8
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
        if out_half_c >= self.reduction:
            self.se = SEBlock(out_half_c, self.reduction)
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            x_first_part = self.branch3(x_first_part)
            x_second_part = self.conv_block(x_second_part)
            if self.out_half_c >= self.reduction:
                x_first_part = self.se(x_first_part)
                x_second_part = self.se(x_second_part)
            out = channel_concatenate(x_first_part, x_second_part)
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 3)
        elif self.stride == 2:
            x1 = self.unit2_branch2(x)
            x2 = self.conv_block(x)
            if self.out_half_c >= self.reduction:
                x1 = self.se(x1)
                x2 = self.se(x2)
            out = channel_concatenate(x1, x2)
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
            if self.out_half_c >= self.reduction:
                out = self.se(out)
        return out


class InvertedResidual_as12_3(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as12_3, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        self.out_half_c = out_half_c
        self.reduction = 6
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
        if out_half_c >= self.reduction:
            self.se = SEBlock(out_half_c, self.reduction)
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            x_first_part = self.branch3(x_first_part)
            x_second_part = self.conv_block(x_second_part)
            if self.out_half_c >= self.reduction:
                x_first_part = self.se(x_first_part)
                x_second_part = self.se(x_second_part)
            out = channel_concatenate(x_first_part, x_second_part)
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 3)
        elif self.stride == 2:
            x1 = self.unit2_branch2(x)
            x2 = self.conv_block(x)
            if self.out_half_c >= self.reduction:
                x1 = self.se(x1)
                x2 = self.se(x2)
            out = channel_concatenate(x1, x2)
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
            if self.out_half_c >= self.reduction:
                out = self.se(out)
        return out


class InvertedResidual_as12_5(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as12_5, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        self.out_half_c = out_half_c
        self.reduction = 4
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
        if out_half_c >= self.reduction:
            self.se = SEBlock(out_half_c, self.reduction)
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            x_first_part = self.branch3(x_first_part)
            x_second_part = self.conv_block(x_second_part)
            if self.out_half_c >= self.reduction:
                x_first_part = self.se(x_first_part)
                x_second_part = self.se(x_second_part)
            out = channel_concatenate(x_first_part, x_second_part)
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 3)
        elif self.stride == 2:
            x1 = self.unit2_branch2(x)
            x2 = self.conv_block(x)
            if self.out_half_c >= self.reduction:
                x1 = self.se(x1)
                x2 = self.se(x2)
            out = channel_concatenate(x1, x2)
            out = channel_shuffle(out, 2)
        else:
            out = self.conv_block(x)
            if self.out_half_c >= self.reduction:
                out = self.se(out)
        return out


class InvertedResidual_as16(nn.Module):
    # inverted residual with linear bottleneck
    def __init__(self, in_c, out_c, stride=1, expansion_factor=1, activation='ReLU6'):
        super(InvertedResidual_as16, self).__init__()
        hidden_c = round(in_c) * expansion_factor
        self.stride = stride
        self.use_connect = self.stride == 1 and in_c == out_c
        # self.conv_part_c = out_c - in_c
        if self.use_connect:
            out_half_c = out_c // 3
            in_c = out_half_c
        elif self.stride == 2:
            out_half_c = out_c // 2
        else:
            out_half_c = out_c
        self.out_half_c = out_half_c
        self.reduction = 6
        
        if self.stride == 2:
            # out_c = self.conv_part_c
            self.unit2_branch2 = nn.Sequential(
                ConvDepthWiseBlock(in_c, stride=2, activation='Linear'),
                ConvPointWiseBlock(in_c, out_half_c, stride=1, activation=activation)
            )
        
        if expansion_factor == 1:
            self.conv_block = nn.Sequential(
                # dw
                ConvDepthWiseBlock(in_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        else:
            self.conv_block = nn.Sequential(
                # pw
                ConvPointWiseBlock(in_c, hidden_c, activation=activation),
                # dw
                ConvDepthWiseBlock(hidden_c, stride=stride, activation=activation),
                # pw linear
                ConvPointWiseBlock(hidden_c, out_half_c, activation='Linear')
            )
        if self.use_connect:
            self.branch3 = nn.Sequential(
                # pw -> dw -> pw-linear
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation),
                # input and output have same channels
                ConvDepthWiseBlock(out_half_c, stride=1, activation='Linear'),
                ConvPointWiseBlock(out_half_c, out_half_c, stride=1, activation=activation)
            )
        if out_half_c >= self.reduction:
            self.se = SEBlock(out_half_c, self.reduction)
        if self.use_connect or self.stride == 2:
            self.pwconv = ConvPointWiseBlock(out_c, out_c, stride=1, activation=activation)
    
    def forward(self, x):
        if self.use_connect:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            x_first_part = self.branch3(x_first_part)
            x_second_part = self.conv_block(x_second_part)
            if self.out_half_c >= self.reduction:
                x_first_part = self.se(x_first_part)
                x_second_part = self.se(x_second_part)
            out = channel_concatenate(x_first_part, x_second_part)
            out = channel_concatenate(out, x_last_part)
            out = self.pwconv(out)
        elif self.stride == 2:
            x1 = self.unit2_branch2(x)
            x2 = self.conv_block(x)
            if self.out_half_c >= self.reduction:
                x1 = self.se(x1)
                x2 = self.se(x2)
            out = channel_concatenate(x1, x2)
            # out = channel_shuffle(out, 2)
            out = self.pwconv(out)
        else:
            out = self.conv_block(x)
            if self.out_half_c >= self.reduction:
                out = self.se(out)
        return out

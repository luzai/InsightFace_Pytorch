from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
from config import conf as gl_conf
import functools, logging
from torch import nn
import numpy as np

upgrade = True  # todo
if gl_conf.use_chkpnt:
    BatchNorm2d = functools.partial(BatchNorm2d, momentum=1 - np.sqrt(0.9))


##################################  Original Arcface Model


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1, need_norm=False, ):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    if need_norm:
        return output, norm
    else:
        return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = PReLU(channels // reduction) if upgrade else ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


from modules.bn import InPlaceABN


def bn_act(depth, with_act):
    if gl_conf.ipabn:
        if with_act:
            return [InPlaceABN(depth, activation='none'),
                    PReLU(depth, ), ]
        else:
            return [
                InPlaceABN(depth, activation='none')]
    else:
        if with_act:
            return [BatchNorm2d(depth), PReLU(depth, ), ]
        else:
            return [BatchNorm2d(depth)]


def bn2d(depth):
    if gl_conf.ipabn:
        return InPlaceABN(depth, activation='none')
    else:
        return BatchNorm2d(depth)


# todo @deprecated
class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            *bn_act(in_channel, False),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            *bn_act(depth, True),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            *bn_act(depth, False))
    
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if upgrade and in_channel == depth and stride == 1:
            self.shortcut_layer = None
        elif not upgrade and in_channel == depth:
            self.shortcut_layer = MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                *bn_act(depth, False)
            )
        if upgrade:
            self.res_layer = Sequential(
                *bn_act(in_channel, False),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                *bn_act(depth, True),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                *bn_act(depth, False),
                SEModule(depth, 16)
            )
        else:
            # todo deprecated
            self.res_layer = Sequential(
                BatchNorm2d(in_channel),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                PReLU(depth),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                BatchNorm2d(depth),
                SEModule(depth, 16)
            )
    
    def forward(self, x):
        if self.shortcut_layer is not None:
            if gl_conf.ipabn:
                shortcut = self.shortcut_layer(x.clone())
            else:
                shortcut = self.shortcut_layer(x)
        else:
            if gl_conf.ipabn:
                shortcut = x.clone()
            else:
                shortcut = x
        res = self.res_layer(x)
        res.add_(shortcut)
        return res


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


from torch.utils.checkpoint import checkpoint_sequential


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      bn2d(64),
                                      PReLU(64))
        
        self.output_layer = Sequential(bn2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        
        # in_channels = 512
        # self.output_layer = nn.Sequential(
        #     Linear_block(in_channels, in_channels, groups=in_channels, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
        #     Flatten(),
        #     nn.Linear(in_channels, 512),
        #     nn.BatchNorm1d(512),
        # )
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
    
    def forward(self, x, normalize=True, return_norm=False, mode='train'):
        if mode == 'finetune':
            with torch.no_grad():
                x = self.input_layer(x)
                x = self.body(x)
        elif mode == 'train':
            x = self.input_layer(x)
            if not gl_conf.use_chkpnt:
                x = self.body(x)
            else:
                x = checkpoint_sequential(self.body, 2, x)
        else:
            raise ValueError(mode)
        x = self.output_layer(x)
        x_norm, norm = l2_norm(x, axis=1, need_norm=True)
        if normalize:
            if return_norm:
                return x_norm, norm
            else:
                return x_norm  # the default one
        else:
            if return_norm:
                return x, norm
            else:
                return x


##################################  MobileFaceNet


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
    
    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    
    def forward(self, x, ):
        return self.model(x)


class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
    
    def forward(self, x, *args, **kwargs):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)


##########################################################
class CSMobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(CSMobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 66, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(66, 66, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=66)
        self.conv_23 = Depth_Wise_2(66, 66, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=132)
        self.conv_3 = Residual_2(66, num_block=4, groups=132, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise_2(66, 132, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=264)
        self.conv_4 = Residual_2(132, num_block=6, groups=264, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise_2(132, 132, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=528)
        self.conv_5 = Residual_2(132, num_block=2, groups=264, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(132, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)


class Depth_Wise_2(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise_2, self).__init__()
        self.residual = residual
        self.reduction = 6
        
        if self.residual:
            out_c = out_c // 3
            in_c = in_c // 3
        else:
            out_c = out_c // 2
        self.out_c = out_c
        
        self.branch1 = nn.Sequential(
            Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        )
        
        if self.residual:
            self.branch2 = nn.Sequential(
                Conv_block(in_c, out_c=in_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
                Linear_block(in_c, in_c, groups=in_c, kernel=kernel, padding=padding, stride=stride),
                Conv_block(in_c, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
            )
        else:
            self.branch3 = nn.Sequential(
                Linear_block(in_c, in_c, kernel=kernel, groups=in_c, padding=padding, stride=stride),
                Conv_block(in_c, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
            )
        if out_c >= self.reduction:
            self.se = SEBlock(out_c, self.reduction)
    
    def forward(self, x):
        if self.residual:
            x_first_part = x[:, :(x.shape[1] // 3), :, :]
            x_second_part = x[:, (x.shape[1] // 3):(x.shape[1] // 3) * 2, :, :]
            x_last_part = x[:, (x.shape[1] // 3) * 2:, :, :]
            x_first_part = self.branch2(x_first_part)
            x_second_part = self.branch1(x_second_part)
            if self.out_c >= self.reduction:
                x_first_part = self.se(x_first_part)
                x_second_part = self.se(x_second_part)
            out = channel_concatenate(x_first_part, x_second_part)
            out = channel_concatenate(out, x_last_part)
            out = channel_shuffle(out, 3)
        else:
            x1 = self.branch1(x)
            x2 = self.branch3(x)
            if self.out_c >= self.reduction:
                x1 = self.se(x1)
                x2 = self.se(x2)
            out = channel_concatenate(x1, x2)
            out = channel_shuffle(out, 2)
        return out


def channel_concatenate(x, out):
    return torch.cat((x, out), 1)


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


class Residual_2(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual_2, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise_2(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    
    def forward(self, x):
        return self.model(x)


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


##################################  Arcface head #################
from torch.nn.utils import weight_norm


# use_kernel2 = False  # kernel2 not work!
# nB = gl_conf.batch_size
# idx_ = torch.arange(0, nB, dtype=torch.long)


class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332, s=gl_conf.scale, m=gl_conf.margin):
        super(Arcface, self).__init__()
        self.classnum = classnum
        # if not use_kernel2:
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # else:
        #     self.kernel = weight_norm(nn.Linear(embedding_size, classnum, bias=False))
        #     self.kernel.weight_g.data = torch.ones((classnum, 1))
        #     self.kernel.weight_v.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # initial kernel
        if gl_conf.fp16:
            m = np.float16(m)
            pi = np.float16(np.pi)
        else:
            m = np.float32(m)
            pi = np.float32(np.pi)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(pi - m)
    
    def forward_eff(self, embbedings, label=None):
        nB = embbedings.shape[0]
        idx_ = torch.arange(0, nB, dtype=torch.long)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        if label is None:
            cos_theta *= self.s
            return cos_theta
        output = cos_theta.clone()  # todo avoid copy ttl
        cos_theta_need = cos_theta[idx_, label]
        cos_theta_2 = torch.pow(cos_theta_need, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta_need * self.cos_m - sin_theta * self.sin_m)
        cond_mask = (cos_theta_need - self.threshold) <= 0
        
        if torch.any(cond_mask).item():
            logging.info('this concatins a difficult sample')
        keep_val = (cos_theta_need - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask].type_as(cos_theta_m)
        output[idx_, label] = cos_theta_m.type_as(output)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        # with torch.no_grad():
        #     o2 = self.forward_neff(embbedings, label)
        return output
    
    def forward_neff(self, embbedings, label):
        nB = embbedings.shape[0]
        idx_ = torch.arange(0, nB, dtype=torch.long)
        ## weights norm
        # if not use_kernel2:
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        # else:
        #     cos_theta = self.kernel(embbedings)
        #         output = torch.mm(embbedings,kernel_norm)
        
        # cos_theta.clamp_(-1, 1)  # for numerical stability
        # cos_theta_m = (cos_theta * self.cos_m - torch.sqrt((1 - torch.pow(cos_theta, 2))) * self.sin_m)
        
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        
        ## this condition controls the theta+m should in range [0, pi]
        ##      0<=theta+m<=pi
        ##     -m<=theta<=pi-m
        
        cond_mask = (cos_theta - self.threshold) <= 0
        
        if torch.any(cond_mask).item():
            logging.info('this concatins a difficult sample')
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output
    
    forward = forward_eff
    # forward = forward_neff


##################################  Cosface head #################

class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = 0.35  # additive margin recommended by the paper
        self.s = 30.  # see normface https://arxiv.org/abs/1704.06369
    
    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


class MySoftmax(Module):
    def __init__(self, embedding_size=512, classnum=51332):
        super(MySoftmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = gl_conf.scale
    
    def forward(self, embeddings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1) * self.s
        return cos_theta


class TripletLoss(Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, inputs, targets, return_info=False):
        n = inputs.size(0)  # todo is this version  correct?
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist.addmm(1, -2, inputs, inputs.t()).clamp(min=1e-6).sqrt() * gl_conf.scale
        # todo how to use triplet only, can use temprature decay/progessive learinig curriculum learning
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # a = to_numpy(targets)
        # print(a.shape,  np.unique(a).shape)
        daps = dist[mask].view(n, -1)  # here can use -1, assume the number of ap is the same, e.g., all is 4!
        # todo how to copy with varied length?
        dans = dist[mask == 0].view(n, -1)
        ap_wei = F.softmax(daps.detach(), dim=1)
        an_wei = F.softmax(-dans.detach(), dim=1)
        dist_ap = (daps * ap_wei).sum(dim=1)
        dist_an = (dans * an_wei).sum(dim=1)
        loss = F.softplus(dist_ap - dist_an).mean()
        if not return_info:
            return loss
        else:
            info = {'dap': dist_ap.mean().item(), 'dan': dist_an.mean().item()}
            return loss, info
    
    def forward_slow(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t()).clamp_(min=1e-12).sqrt_()
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):  # todo turn to matrix operation
            # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            # dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            tmp = mask[i]
            # assert tmp[i].item()==1
            tmp[i] = 0
            daps = dist[i][mask[i]]  # todo fx  bug  : should  remove self, ? will it affects?
            ap_wei = F.softmax(daps.detach(), dim=0)
            # ap_wei = F.softmax(daps.detach() / (128 ** (1/2)), dim=0)
            # ap_wei = F.softmax(daps, dim=0) # allow atention on weright
            dist_ap.append((daps * ap_wei).sum().unsqueeze(0))
            
            dans = dist[i][mask[i] == 0]
            an_wei = F.softmax(-dans.detach(), dim=0)
            # an_wei = F.softmax(-dans.detach() / (128 ** (1/2)) , dim=0)
            # an_wei = F.softmax(-dans, dim=0)
            dist_an.append((dans * an_wei).sum().unsqueeze(0))
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        # y = torch.ones_like(dist_an)
        # loss = self.ranking_loss(dist_an, dist_ap, y)
        ## soft margin
        loss = F.softplus(dist_ap - dist_an).mean()
        return loss

# -*- coding: future_fstrings -*-
from lz import *
from torch.nn import BatchNorm1d, BatchNorm2d, Dropout, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
from collections import namedtuple
from config import conf
import functools, logging
from torch import nn
import numpy as np

if conf.use_chkpnt:
    BatchNorm2d = functools.partial(BatchNorm2d, momentum=1 - np.sqrt(0.9))


def l2_norm(input, axis=1, need_norm=False, ):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    if need_norm:
        return output, norm
    else:
        return output


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(Module):
    def forward(self, x):
        return x


class Swish(nn.Module):
    def __init__(self, *args):
        super(Swish, self).__init__()

    def forward(self, x):
        res = x * torch.sigmoid(x)
        # assert not torch.isnan(res).any().item()
        return res


class Mish(nn.Module):
    def __init__(self, *args):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


if conf.use_act == 'elu':
    NonLin = lambda *args: nn.ELU(inplace=True)
elif conf.use_act == 'prelu':
    NonLin = nn.PReLU
elif conf.use_act == 'mish':
    NonLin = Mish
elif conf.use_act == 'swish':
    NonLin = Swish
else:
    raise NotImplementedError()


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4, mult=conf.sigmoid_mult):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = NonLin(channels // reduction) if conf.upgrade_irse else nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        # todo tanh+1 or sigmoid or sigmoid*2
        self.sigmoid = nn.Sigmoid()  # nn.Tanh()
        self.mult = mult
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # +1
        # def norm(x):
        #     return torch.norm(x[1])
        # norm(module_input), norm(module_input*x), norm(module_input * x * self.mult)
        return module_input * x * self.mult


from modules.bn import InPlaceABN, InPlaceABNSync


def bn_act(depth, with_act, ipabn=None):
    if ipabn:
        if with_act:
            if ipabn == 'sync':
                return [InPlaceABNSync(depth, activation='none'),
                        NonLin(depth, ), ]
            else:
                return [InPlaceABN(depth, activation='none'),
                        NonLin(depth, ), ]
        else:
            return [InPlaceABN(depth, activation='none')]
    else:
        if with_act:
            return [BatchNorm2d(depth), NonLin(depth, ), ]
        else:
            return [BatchNorm2d(depth)]


def bn2d(depth, ipabn=None):
    if ipabn:
        if ipabn == 'sync':
            return InPlaceABNSync(depth, activation='none')
        else:
            return InPlaceABN(depth, activation='none')
    else:
        return BatchNorm2d(depth)


# @deprecated
# todo gl_conf.upgrade_ir
class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride, ):
        super(bottleneck_IR, self).__init__()
        ipabn = conf.ipabn
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        if conf.upgrade_irse:
            self.res_layer = Sequential(
                *bn_act(in_channel, False, ipabn),
                nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                *bn_act(depth, True, ipabn),
                nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                *bn_act(depth, False, ipabn))
        else:
            self.res_layer = Sequential(
                BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), NonLin(depth),
                nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class true_bottleneck_IR_SE(Module):  # this is botleneck block
    def __init__(self, in_channel, depth, stride):
        super(true_bottleneck_IR_SE, self).__init__()
        if in_channel == depth and stride == 1:
            self.shortcut_layer = Identity()
        else:
            # todo arch ft
            self.shortcut_layer = Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                *bn_act(depth, False, conf.ipabn)
            )
        self.res_layer = Sequential(
            *bn_act(in_channel, False, conf.ipabn),
            nn.Conv2d(in_channel, in_channel // 4, (1, 1), (1, 1), 0, bias=False),
            *bn_act(in_channel // 4, True, conf.ipabn),
            nn.Conv2d(in_channel // 4, in_channel // 4, (3, 3), stride, 1, bias=False),
            *bn_act(in_channel // 4, True, conf.ipabn),
            nn.Conv2d(in_channel // 4, depth, (1, 1), 1, 0, bias=False),
            *bn_act(depth, False, conf.ipabn),
            SEModule(depth, 16),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        res.add_(shortcut)
        return res


class bottleneck_IR_SE(Module):  # this is basic block
    def __init__(self, in_channel, depth, stride, use_in=False):
        super(bottleneck_IR_SE, self).__init__()
        if not conf.spec_norm:
            conv_op = nn.Conv2d
        else:
            conv_op = lambda *args, **kwargs: nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
        if in_channel == depth and stride == 1:
            self.shortcut_layer = Identity()
        else:
            if not conf.arch_ft:
                self.shortcut_layer = Sequential(
                    nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                    *bn_act(depth, False, conf.ipabn)
                )
            else:
                self.shortcut_layer = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channel, depth, 1, 1, bias=False),
                    *bn_act(depth, False, conf.ipabn)
                )
        assert conf.upgrade_irse
        self.res_layer = Sequential(
            *bn_act(in_channel, False, conf.ipabn),
            conv_op(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            *bn_act(depth, True, conf.ipabn),
            conv_op(depth, depth, (3, 3), stride, 1, bias=False),
            *bn_act(depth, False, conf.ipabn),
            SEModule(depth, 16)
        )
        if not use_in:
            self.IN = None
        else:
            self.IN = nn.InstanceNorm2d(depth, affine=True)

    def forward_ipabn(self, x):
        shortcut = self.shortcut_layer(x.clone())
        res = self.res_layer(x)
        res.add_(shortcut)
        if self.IN:
            res = self.IN(res)
        return res

    def forward_ori(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        res.add_(shortcut)
        if self.IN:
            res = self.IN(res)
        return res

    if conf.ipabn:
        forward = forward_ipabn
    else:
        forward = forward_ori


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if conf.bottle_neck:
        filter_list = [64, 256, 512, 1024, 2048]
    else:
        filter_list = [64, 64, 128, 256, 512]
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:  # basic
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 98:
        units = [3, 4, 38, 3]
    elif num_layers == 99:
        units = [3, 8, 35, 3]
    elif num_layers == 100:  # basic
        units = [3, 13, 30, 3]
    elif num_layers == 134:
        units = [3, 10, 50, 3]
    elif num_layers == 136:
        units = [3, 13, 48, 3]
    elif num_layers == 140:  # basic
        units = [3, 15, 48, 3]
    elif num_layers == 124:  # basic
        units = [3, 13, 40, 5]
    elif num_layers == 160:  # basic
        units = [3, 24, 49, 3]
    elif num_layers == 101:  # bottleneck
        units = [3, 4, 23, 3]
    elif num_layers == 152:  # bottleneck
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]

    blocks = [get_block(filter_list[0], filter_list[1], units[0]),
              get_block(filter_list[1], filter_list[2], units[1]),
              get_block(filter_list[2], filter_list[3], units[2]),
              get_block(filter_list[3], filter_list[4], units[3])
              ]

    return blocks


from torch.utils.checkpoint import checkpoint_sequential


class Backbone(Module):
    def __init__(self, num_layers=conf.net_depth, drop_ratio=conf.drop_ratio, mode=conf.net_mode,
                 ebsize=conf.embedding_size):
        super(Backbone, self).__init__()
        # assert num_layers in [18, 34, 20, 50, 100, 152, ], 'num_layers should be not defined'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            if not conf.bottle_neck:
                unit_module = bottleneck_IR_SE
            else:
                unit_module = true_bottleneck_IR_SE
        if not conf.arch_ft and not conf.use_in:
            self.input_layer = Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                          bn2d(64, conf.ipabn),
                                          NonLin(64))
        elif conf.arch_ft and not conf.use_in:
            self.input_layer = Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                          bn2d(64, conf.ipabn),
                                          NonLin(64),
                                          nn.Conv2d(64, 64, (3, 3), 1, 1, bias=False),
                                          bn2d(64, conf.ipabn),
                                          NonLin(64),
                                          )
        elif conf.arch_ft and conf.use_in:
            self.input_layer = Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                          nn.InstanceNorm2d(64, affine=True),
                                          NonLin(64),
                                          nn.Conv2d(64, 64, (3, 3), 1, 1, bias=False),
                                          nn.InstanceNorm2d(64, affine=True),
                                          NonLin(64),
                                          )
        elif not conf.arch_ft and conf.use_in:
            self.input_layer = Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                          nn.InstanceNorm2d(64, affine=True),
                                          NonLin(64), )
        out_resolution = conf.input_size // 16

        if conf.bottle_neck:
            expansions = 4
        else:
            expansions = 1
        out_planes = 512 * expansions
        if conf.out_type == 'fc':
            self.output_layer = Sequential(
                bn2d(out_planes, conf.ipabn),
                Dropout(drop_ratio),
                Flatten(),
                nn.Linear(out_planes * out_resolution ** 2, ebsize,
                          bias=True if not conf.upgrade_bnneck else False),
                nn.BatchNorm1d(ebsize))
            if conf.pfe:
                self.output_layer_sigma = Sequential(
                    bn2d(out_planes, conf.ipabn),
                    Dropout(drop_ratio),
                    Flatten(),
                    nn.Linear(out_planes * out_resolution ** 2, ebsize,
                              bias=True if not conf.upgrade_bnneck else False),
                    nn.BatchNorm1d(ebsize))

        elif conf.out_type == 'gpool':
            self.output_layer = nn.Sequential(
                bn2d(out_planes, conf.ipabn),
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(out_planes, ebsize, bias=False),
                nn.BatchNorm1d(ebsize),
            )
        modules = []
        for ind, block in enumerate(blocks):
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride,
                                use_in=ind==0 if conf.use_in else False,
                                ))
        self.body = Sequential(*modules)
        if conf.mid_type == 'gpool':
            self.fcs = nn.Sequential(
                nn.Sequential(bn2d(64 * expansions, conf.ipabn),
                              nn.AdaptiveAvgPool2d(1),
                              Flatten(),
                              nn.Linear(64 * expansions, ebsize, bias=False),
                              nn.BatchNorm1d(ebsize), ),
                nn.Sequential(bn2d(128 * expansions, conf.ipabn),
                              nn.AdaptiveAvgPool2d(1),
                              Flatten(),
                              nn.Linear(128 * expansions, ebsize, bias=False),
                              nn.BatchNorm1d(ebsize), ),
                nn.Sequential(bn2d(256 * expansions, conf.ipabn),
                              nn.AdaptiveAvgPool2d(1),
                              Flatten(),
                              nn.Linear(256 * expansions, ebsize, bias=False),
                              nn.BatchNorm1d(ebsize), ),
            )
        elif conf.mid_type == 'fc':
            self.fcs = nn.Sequential(
                nn.Sequential(bn2d(64 * expansions, conf.ipabn),
                              Dropout(drop_ratio),
                              Flatten(),
                              nn.Linear(64 * expansions * (out_resolution * 8) ** 2, ebsize, bias=False),
                              nn.BatchNorm1d(ebsize), ),
                nn.Sequential(bn2d(128 * expansions, conf.ipabn),
                              Dropout(drop_ratio),
                              Flatten(),
                              nn.Linear(128 * expansions * (out_resolution * 4) ** 2, ebsize, bias=False),
                              nn.BatchNorm1d(ebsize), ),
                nn.Sequential(bn2d(256 * expansions, conf.ipabn),
                              Dropout(drop_ratio),
                              Flatten(),
                              nn.Linear(256 * expansions * (out_resolution * 2) ** 2, ebsize, bias=False),
                              nn.BatchNorm1d(ebsize), ),
            )
        else:
            self.fcs = None
        if conf.use_bl:
            self.bls = nn.Sequential(
                nn.Sequential(*[unit_module(64 * expansions, 64 * expansions, 1) for _ in range(3)]),
                nn.Sequential(*[unit_module(128 * expansions, 128 * expansions, 1) for _ in range(2)]),
                nn.Sequential(*[unit_module(256 * expansions, 256 * expansions, 1) for _ in range(1)]),
            )
            self.fuse_wei = nn.Linear(4, 1, bias=False)
        else:
            self.bls = nn.Sequential(
                Identity(),
                Identity(),
                Identity(),
            )
        self._initialize_weights()
        if conf.pfe:
            nn.init.constant_(self.output_layer_sigma[-1].weight, 0)
            nn.init.constant_(self.output_layer_sigma[-1].bias, 1)

    def forward(self, inp, *args, **kwargs, ):
        bs = inp.shape[0]
        if conf.input_size != inp.shape[-1]:
            inp = F.interpolate(inp, size=conf.input_size, mode='bilinear', align_corners=True)  # bicubic
        x = self.input_layer(inp)
        # x = self.body(x)
        if conf.net_depth == 18:
            break_inds = [1, 3, 5, ]  # r18
        else:
            break_inds = [3 - 1, 3 + 4 - 1, 3 + 4 + 13 - 1, ]  # r50
        xs = []
        shps = []
        for ind, layer in enumerate(self.body):
            x = layer(x)
            shps.append(x.shape)
            if ind in break_inds:
                xs.append(x)
        xs.append(x)
        assert not judgenan(x)
        if conf.mid_type:
            v4 = self.output_layer(x)
            v1 = self.fcs[0](self.bls[0](xs[0]))
            v2 = self.fcs[1](self.bls[1](xs[1]))
            v3 = self.fcs[2](self.bls[2](xs[2]))
            v5 = self.fuse_wei(torch.stack([v1, v2, v3, v4], dim=-1)).view(bs, -1)
            if conf.ds and self.training:
                return [v5, v4, v3, v2, v1]
            elif conf.ds and not self.training:
                return v5 # todo for test mid performance
                # return v3
        else:
            v5 = self.output_layer(x)
            assert not judgenan(v5)
            if conf.pfe:
                sigma = self.output_layer_sigma(x)
                return v5, sigma
            return v5

    def forward_ori(self, inp, *args, **kwargs):
        if conf.input_size != inp.shape[-1]:
            inp = F.interpolate(inp, size=conf.input_size, mode='bilinear', align_corners=True)  # bicubic
        x = self.input_layer(inp)
        x = self.body(x)
        x = self.output_layer(x)
        return x

    def forward_old(self, x, normalize=True, return_norm=False, mode='train'):
        if mode == 'finetune':
            with torch.no_grad():
                x = self.input_layer(x)
                x = self.body(x)
        elif mode == 'train':
            x = self.input_layer(x)
            if not conf.use_chkpnt:
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        # todo
        for m in self.modules():
            if isinstance(m, bottleneck_IR_SE):
                nn.init.constant_(m.res_layer[5].weight, 0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 ):
        super(DoubleConv, self).__init__()
        if isinstance(kernel_size, tuple) and kernel_size[0] == kernel_size[1]:
            kernel_size = kernel_size[0]
        zp = kernel_size + 1
        self.cl, self.cl2, self.zp, self.z, = in_channels, out_channels, zp, kernel_size,
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride, self.padding = stride, padding
        self.bias = None
        self.groups = groups
        self.dilation = dilation
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, zp, zp))
        self.reset_parameters()

    def reset_parameters(self):
        from torch.nn import init
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        cl, cl2, zp, z, = self.cl, self.cl2, self.zp, self.z,
        cl //= self.groups
        with torch.no_grad():
            Il = torch.eye(cl * z * z).type_as(self.weight)
            Il = Il.view(cl * z * z, cl, z, z)
        Wtl = F.conv2d(self.weight, Il)
        zpz = zp - z + 1
        Wtl = Wtl.view(cl2 * zpz * zpz, cl, z, z)
        Ol2 = F.conv2d(input, Wtl, bias=None, stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation, groups=self.groups, )
        bs, _, wl2, hl2 = Ol2.size()
        Ol2 = Ol2.view(bs, -1, zpz, zpz)
        Il2 = F.adaptive_avg_pool2d(Ol2, (1, 1))
        res = Il2.view(bs, -1, wl2, hl2)
        return res


# DoubleConv(16,32)(torch.randn(4,16,112,112))
def count_double_conv(m, x, y):
    x = x[0]

    cin = m.in_channels
    cout = m.out_channels
    kh = kw = m.kernel_size
    batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)
    multiply_adds = 1
    kernel_ops = multiply_adds * kh * kw
    output_elements = batch_size * out_w * out_h * cout
    total_ops = output_elements * kernel_ops * cin // m.groups
    zp, z = m.zp, m.z
    zpz = zp - z + 1
    total_ops *= zpz ** 2
    total_ops += y.numel() * zpz ** 2
    m.total_ops = torch.Tensor([int(total_ops)])


class STNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 controller=None,
                 reduction=1,
                 ):
        super(STNConv, self).__init__()

        if isinstance(kernel_size, tuple) and kernel_size[0] == kernel_size[1]:
            kernel_size = kernel_size[0]
        zmeta = kernel_size + 1
        if controller is None:
            controller = get_controller(scale=(1,))  # todo kernel_size / (kernel_size + .5)
        self.in_plates, self.out_plates, self.zmeta, self.z, = in_channels, out_channels, zmeta, kernel_size,
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = None
        self.groups = groups
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.FloatTensor(out_channels // 4, in_channels // groups, zmeta, zmeta))  # todo
        self.reset_parameters()
        self.register_buffer('theta', torch.FloatTensor(controller).view(-1, 2, 3))
        # self.stride2 = self.theta.shape[0] # todo
        self.stride2 = 1
        self.n_inst, self.n_inst_sqrt = (self.zmeta - self.z + 1) * (self.zmeta - self.z + 1), self.zmeta - self.z + 1

    def reset_parameters(self):
        from torch.nn import init
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        bs = input.size(0)
        weight_l = []
        for theta_ in (self.theta):
            grid = F.affine_grid(theta_.expand(self.weight.size(0), 2, 3), self.weight.size())
            weight_l.append(F.grid_sample(self.weight, grid))
        # todo
        # weight_l.append(self.weight.transpose(2,3).flip(3)) # 270
        # weight_l.append(self.weight.flip(2).flip(3) # 180
        # weight_l.append(self.weight.transpose(2,3).flip(2)) # 90
        weight_inst = torch.cat(weight_l)
        weight_inst = weight_inst[:, :, :self.kernel_size, :self.kernel_size]
        out = F.conv2d(input, weight_inst, bias=None, stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation, groups=self.groups, )
        # self.out_inst = out
        h, w = out.shape[2], out.shape[3]
        out = out.view(bs, -1, self.out_plates, h, w)
        out = out.permute(0, 3, 4, 1, 2).contiguous().view(bs, self.out_plates * h * w, -1)
        out = F.avg_pool1d(out, self.stride2)
        # out = F.max_pool1d(out, self.stride2) # todo
        out = out.permute(0, 2, 1).contiguous().view(bs, -1, h, w)
        # self.out=out
        # out = F.avg_pool2d(out, out.size()[2:])
        # out = out.view(out.size(0), -1)
        return out


def get_controller(
        scale=(1,
                # 3 / 3.5,
               ),
        translation=(0,
                # 2 / (meta_kernel_size - 1),
                     ),
        theta=(0,
               np.pi,
               # np.pi / 16, -np.pi / 16,
               np.pi / 2, -np.pi / 2,
                # np.pi / 4, -np.pi / 4,
                # np.pi * 3 / 4, -np.pi * 3 / 4,
               )
):
    controller = []
    for sx in scale:
        # for sy in scale:
        sy = sx
        for tx in translation:
            for ty in translation:
                for th in theta:
                    controller.append([sx * np.cos(th), -sx * np.sin(th), tx,
                                       sy * np.sin(th), sy * np.cos(th), ty])
    logging.info(f'controller stride is {len(controller)} ', )
    controller = np.stack(controller)
    controller = controller.reshape(-1, 2, 3)
    controller = np.ascontiguousarray(controller, np.float32)
    return controller


# m = STNConv(4, 16, controller=get_controller()).cuda()
# m(torch.randn(1, 4, 112, 112).cuda())


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, ):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c,
                              kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                              bias=False)
        if conf.spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.bn = bn2d(out_c, conf.ipabn)
        self.PReLU = NonLin(out_c)

    # @jit.script_method
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.PReLU(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, spec_norm=conf.spec_norm):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c,
                              kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                              bias=False)
        if spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.bn = bn2d(out_c, conf.ipabn)

    # @jit.script_method
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


from models.lpf import Downsample


class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, ):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1), )
        if conf.lpf and stride[0] == 2:
            self.conv_dw = Linear_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=(1, 1),

                                        )
        else:
            self.conv_dw = Linear_block(
                groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride,
                spec_norm=False  # todo
            )
        if conf.mbfc_se:
            self.se = SEModule(groups)
        else:
            self.se = Identity()
        self.conv_dw_nlin = NonLin(groups)
        if conf.lpf and stride[0] == 2:
            self.dnsmpl = Downsample(channels=groups, filt_size=5, stride=2)
        else:
            self.dnsmpl = Identity()
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1), )
        self.residual = residual

    def forward(self, x):
        xx = self.conv(x)
        xx = self.conv_dw(xx)
        xx = self.se(xx)
        xx = self.conv_dw_nlin(xx)
        xx = self.dnsmpl(xx)
        xx = self.project(xx)
        if self.residual:
            output = x + xx
        else:
            output = xx
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), width_mult=1.):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups,
                           ))
        self.model = Sequential(*modules)

    def forward(self, x, ):
        return self.model(x)


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileFaceNet(Module):
    def __init__(self, embedding_size=conf.embedding_size,
                 width_mult=conf.mbfc_wm, depth_mult=conf.mbfc_dm):
        super(MobileFaceNet, self).__init__()
        # if mode == 'small':
        #     blocks = [1, 4, 6, 2]
        # else:
        #     blocks = [2, 8, 16, 4]
        blocks = [1, 4, 8, 2]
        blocks = [make_divisible(b * depth_mult, 1) for b in blocks]
        self.conv1 = Conv_block(3, make_divisible(64 * width_mult),
                                kernel=(3, 3), stride=(2, 2), padding=(1, 1), )
        # if blocks[0] == 1:
        #     self.conv2_dw = Conv_block(make_divisible(64 * width_mult), make_divisible(64 * width_mult), kernel=(3, 3),
        #                                stride=(1, 1), padding=(1, 1), groups=make_divisible(64 * width_mult),
        #                                )
        # else:
        self.conv2_dw = Residual(make_divisible(64 * width_mult), num_block=blocks[0],
                                 groups=make_divisible(64 * width_mult), kernel=(3, 3), stride=(1, 1),
                                 padding=(1, 1),
                                 )
        self.conv_23 = Depth_Wise(make_divisible(64 * width_mult), make_divisible(64 * width_mult),
                                  kernel=(3, 3),
                                  stride=(2, 2), padding=(1, 1),
                                  groups=make_divisible(128 * width_mult),
                                  )
        self.conv_3 = Residual(make_divisible(64 * width_mult), num_block=blocks[1],
                               groups=make_divisible(128 * width_mult), kernel=(3, 3), stride=(1, 1), padding=(1, 1),
                               )
        self.conv_34 = Depth_Wise(make_divisible(64 * width_mult), make_divisible(128 * width_mult), kernel=(3, 3),
                                  stride=(2, 2), padding=(1, 1), groups=make_divisible(256 * width_mult),
                                  )
        self.conv_4 = Residual(make_divisible(128 * width_mult), num_block=blocks[2],
                               groups=make_divisible(256 * width_mult), kernel=(3, 3), stride=(1, 1), padding=(1, 1),
                               )
        self.conv_45 = Depth_Wise(make_divisible(128 * width_mult), make_divisible(128 * width_mult), kernel=(3, 3),
                                  stride=(2, 2), padding=(1, 1), groups=make_divisible(512 * width_mult),
                                  )
        self.conv_5 = Residual(make_divisible(128 * width_mult), num_block=blocks[3],
                               groups=make_divisible(256 * width_mult), kernel=(3, 3), stride=(1, 1), padding=(1, 1),
                               )
        # Conv2d_bk = nn.Conv2d
        # nn.Conv2d = STNConv
        self.conv_6_sep = Conv_block(make_divisible(128 * width_mult), make_divisible(512 * width_mult), kernel=(1, 1),
                                     stride=(1, 1), padding=(0, 0), )
        out_resolution = conf.input_size // 16
        self.conv_6_dw = Linear_block(make_divisible(512 * width_mult), make_divisible(512 * width_mult),
                                      groups=make_divisible(512 * width_mult), kernel=(out_resolution, out_resolution),
                                      stride=(1, 1),
                                      padding=(0, 0),
                                      )
        # nn.Conv2d = Conv2d_bk
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(make_divisible(512 * width_mult), embedding_size, bias=False, )
        if conf.spec_norm:
            self.linear = nn.utils.spectral_norm(self.linear)
        self.bn = nn.BatchNorm1d(embedding_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, DoubleConv, STNConv)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    # @jit.script_method
    def forward(self, x, *args, **kwargs):
        if conf.input_size != x.shape[-1]:
            x = F.interpolate(x, size=conf.input_size, mode='bicubic', align_corners=True)
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        fea7x7 = self.conv_6_sep(out)
        out = self.conv_6_dw(fea7x7)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        if kwargs.get('use_of', False):
            return out, fea7x7
        else:
            return out


class CSMobileFaceNet(nn.Module):
    def __init__(self):
        raise ValueError('deprecated')


# nB = gl_conf.batch_size
# idx_ = torch.arange(0, nB, dtype=torch.long)

class AdaCos(nn.Module):
    def __init__(self, num_classes=None, m=conf.margin, num_features=conf.embedding_size):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)  # todo maybe scale
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(num_features, num_classes))
        nn.init.xavier_uniform_(self.kernel)
        self.device_id = list(range(conf.num_devs))
        self.step = 0
        self.writer = conf.writer
        self.interval = conf.log_interval
        self.k = 0.5
        assert self.writer is not None

    def update_mrg(self, m=conf.margin, s=conf.scale):
        self.m = m
        self.s = s

    def forward(self, input, label=None):
        bs = input.shape[0]
        x = F.normalize(input, dim=1)
        W = F.normalize(self.kernel, dim=0)
        # logits = F.linear(x, W)
        logits = torch.mm(x, W).clamp(-1, 1)
        logits = logits.float()
        if label is None:
            return logits
        # add margin
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        if self.m != 0:
            target_logits = torch.cos(theta + self.m)
            output = logits * (1 - one_hot) + target_logits * one_hot
        else:
            output = logits
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.logsumexp((self.s * logits)[one_hot < 1], dim=0) - np.log(input.shape[0])
            B_avg = B_avg + np.log(self.k / (1 - self.k))
            # print(B_avg, self.s)
            theta_neg = theta[one_hot < 1].view(bs, self.n_classes - 1)
            theta_pos = theta[one_hot == 1]
            theta_med = torch.median(theta_pos + self.m)
            s_now = B_avg / torch.cos(torch.min(
                (math.pi / 4 + self.m) * torch.ones_like(theta_med),
                theta_med))
            # self.s = self.s * 0.9 + s_now * 0.1
            self.s = s_now
            if self.step % self.interval == 0:
                self.writer.add_scalar('theta/pos_med', theta_med.item(), self.step)
                self.writer.add_scalar('theta/pos_mean', theta_pos.mean().item(), self.step)
                self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)
                self.writer.add_scalar('theta/neg_mean', theta_neg.mean().item(), self.step)
                self.writer.add_scalar('theta/bavg', B_avg.item(), self.step)
                self.writer.add_scalar('theta/scale', self.s, self.step)
            if self.step % 9999 == 0:
                self.writer.add_histogram('theta/pos_th', theta_pos, self.step)
                self.writer.add_histogram('theta/pos_neg', theta_neg, self.step)
            self.step += 1

        output *= self.s
        return output


class AdaMrg(nn.Module):
    def __init__(self, num_classes=None, m=conf.margin, num_features=conf.embedding_size):
        super(AdaMrg, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = conf.scale
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(num_features, num_classes))
        nn.init.xavier_uniform_(self.kernel)
        self.device_id = list(range(conf.num_devs))
        self.step = 0
        self.writer = conf.writer
        self.interval = conf.log_interval
        self.k = 0.5
        assert self.writer is not None

    def update_mrg(self, m=conf.margin, s=conf.scale):
        self.m = m
        self.s = s

    def forward(self, input, label=None):
        bs = input.shape[0]  # (bs, fdim)
        x = F.normalize(input, dim=1)  # (bs, fdim)
        W = F.normalize(self.kernel, dim=0)  # (fdim, ncls)
        logits = torch.mm(x, W).clamp(-1, 1)  # (bs, ncls)
        logits = logits.float()
        if label is None:
            return logits
        # add margin
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))

        # calc margin
        with torch.no_grad():
            B_avg = torch.logsumexp(self.s * logits[one_hot < 1], dim=0) - np.log(bs)
            # B_avg = torch.FloatTensor([np.log(logits.shape[1] - 1)])
            B_avg = B_avg + np.log(self.k / (1 - self.k))
            theta_neg = theta[one_hot < 1].view(bs, self.n_classes - 1)
            theta_pos = theta[one_hot == 1]
            theta_med = torch.median(theta_pos)
            m_now = torch.acos(B_avg / self.s) - min(theta_med.item(),
                                                     self.k * np.pi / 2
                                                     )
            # m_now = m_now.clamp(0.1, 0.5)
            m_now = m_now.item()
            self.m = m_now
            if self.step % self.interval == 0:
                print('margin ', m_now, theta_med.item(), torch.acos(B_avg / self.s).item(), )
                self.writer.add_scalar('theta/mrg', m_now, self.step)
                self.writer.add_scalar('theta/pos_med', theta_med.item(), self.step)
                self.writer.add_scalar('theta/pos_mean', theta_pos.mean().item(), self.step)
                self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)
                self.writer.add_scalar('theta/neg_mean', theta_neg.mean().item(), self.step)
                self.writer.add_scalar('theta/bavg', B_avg.item(), self.step)
                self.writer.add_scalar('theta/scale', self.s, self.step)
            if self.step % 999 == 0:
                self.writer.add_histogram('theta/pos_th', theta_pos, self.step)
                self.writer.add_histogram('theta/pos_neg', theta_neg, self.step)
            self.step += 1
        if self.m != 0:
            target_logits = torch.cos(theta + self.m)
            output = logits * (1 - one_hot) + target_logits * one_hot
        else:
            output = logits
        output *= self.s
        return output


class AdaMArcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=conf.embedding_size, classnum=None, s=conf.scale, m=conf.margin):
        super(AdaMArcface, self).__init__()
        self.classnum = classnum
        kernel = Parameter(torch.FloatTensor(embedding_size, classnum))
        kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.kernel = kernel
        self.update_mrg()
        self.easy_margin = False
        self.step = 0
        self.writer = conf.writer
        self.interval = conf.log_interval
        m = Parameter(torch.Tensor(self.classnum))
        m.data.fill_(0.5)
        self.m = m
        self.update_mrg()

    def update_mrg(self, m=conf.margin, s=conf.scale):
        self.s = s

    def clamp_m(self):
        self.m.data.clamp_(0, 0.8)

    def forward_eff_v1(self, embeddings, label=None):
        assert not torch.isnan(embeddings).any().item()
        dev = self.m.get_device()
        if dev == -1:
            dev = 0
        cos_m = (torch.cos(self.m))
        sin_m = (torch.sin(self.m))
        mm = (torch.sin(self.m) * (self.m))
        threshold = (torch.cos(np.pi - self.m))

        bs = embeddings.shape[0]
        idx_ = torch.arange(0, bs, dtype=torch.long)
        # self.m = self.m.clamp(min=0)
        m_mean = torch.mean(self.m)
        if self.interval >= 1 and self.step % self.interval == 0:
            with torch.no_grad():
                norm_mean = torch.norm(embeddings, dim=1).mean()
                m_mean = torch.mean(self.m).cuda()
                if self.writer:
                    self.writer.add_scalar('theta/norm_mean', norm_mean.item(), self.step)
                    self.writer.add_scalar('theta/m_mean', m_mean.item(), self.step)
                    self.writer.add_histogram('ms', to_numpy(self.m), self.step)
                logging.info(f'norm {norm_mean.item():.2e}')
                logging.info(f'm_mean {m_mean.item():.2e}')
        embeddings = F.normalize(embeddings, dim=1)
        kernel_norm = l2_norm(self.kernel, axis=0)  # 0 dim is emd dim
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        if label is None:
            return cos_theta
        with torch.no_grad():
            if self.interval >= 1 and self.step % self.interval == 0:
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                theta = torch.acos(cos_theta)
                theta_neg = theta[one_hot < 1].view(bs, self.classnum - 1)
                theta_pos = theta[one_hot == 1].view(bs)
                if self.writer:
                    self.writer.add_scalar('theta/pos_med', torch.median(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)
                logging.info(f'pos_med: {torch.median(theta_pos).item():.2e} ' +
                             f'neg_med: {torch.median(theta_neg).item():.2e} '
                             )
        output = cos_theta.clone()  # todo avoid copy ttl
        cos_theta_need = cos_theta[idx_, label]
        cos_theta_2 = torch.pow(cos_theta_need, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)

        cos_theta_m = (cos_theta_need * cos_m[label] - sin_theta * sin_m[label])
        cond_mask = (cos_theta_need - threshold[label]) <= 0

        if torch.any(cond_mask).item():
            logging.info(f'this concatins a difficult sample, {cond_mask.sum().item()}')
        if self.easy_margin:
            keep_val = cos_theta_need
        else:
            keep_val = (cos_theta_need - mm[label])  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask].type_as(cos_theta_m)
        if self.writer and self.step % self.interval == 0:
            # self.writer.add_scalar('theta/cos_th_m_mean', torch.median(cos_theta_m).item(), self.step)
            self.writer.add_scalar('theta/cos_th_m_median', cos_theta_m.mean().item(), self.step)
        output[idx_, label] = cos_theta_m.type_as(output)
        output *= self.s
        self.step += 1

        return output, m_mean

    forward = forward_eff_v1


class MySoftmax(Module):
    def __init__(self, embedding_size=conf.embedding_size, classnum=None):
        super(MySoftmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = conf.scale
        self.step = 0
        self.writer = conf.writer

    def forward(self, embeddings, label):
        embeddings = F.normalize(embeddings, dim=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        logits = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        with torch.no_grad():
            if self.step % conf.log_interval == 0:
                bs = embeddings.shape[0]
                one_hot = torch.zeros_like(logits)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                theta = torch.acos(logits)
                theta_neg = theta[one_hot < 1].view(bs, self.classnum - 1)
                theta_pos = theta[one_hot == 1].view(bs)
                self.writer.add_scalar('theta/pos_med', torch.median(theta_pos).item(), self.step)
                self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)
            self.step += 1
        logits *= self.s
        return logits


class ArcSinMrg(nn.Module):
    def __init__(self, embedding_size=conf.embedding_size, classnum=None, s=conf.scale, m=conf.margin):
        super(ArcSinMrg, self).__init__()
        self.classnum = classnum
        kernel = nn.Linear(embedding_size, classnum, bias=False)
        # kernel = Parameter(torch.Tensor(embedding_size, classnum))
        kernel.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # todo
        if conf.spec_norm:
            kernel = nn.utils.spectral_norm(kernel)
        self.kernel = kernel
        self.update_mrg()
        self.easy_margin = False
        self.step = 0
        self.writer = conf.writer
        self.interval = conf.log_interval

    def update_mrg(self, m=conf.margin, s=conf.scale):
        m = np.float32(m)
        pi = np.float32(np.pi)
        dev = conf.model1_dev[0]
        if dev == -1:
            dev = 0
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.threshold = torch.FloatTensor([math.cos(pi - m)]).to(dev)

    def forward(self, embeddings, label):
        assert not torch.isnan(embeddings).any().item()
        bs = embeddings.shape[0]
        idx_ = torch.arange(0, bs, dtype=torch.long)
        if self.interval >= 1 and self.step % self.interval == 0:
            with torch.no_grad():
                norm_mean = torch.norm(embeddings, dim=1).mean()
                if self.writer:
                    self.writer.add_scalar('theta/norm_mean', norm_mean.item(), self.step)
                logging.info(f'norm {norm_mean.item():.2e}')
        embeddings = F.normalize(embeddings, dim=1)
        cos_theta = self.kernel(embeddings)
        cos_theta = cos_theta / torch.norm(self.kernel.weight, dim=1)
        if label is None:
            cos_theta *= self.s
            return cos_theta
        with torch.no_grad():
            if self.interval >= 1 and self.step % self.interval == 0:
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                theta = torch.acos(cos_theta)
                theta_neg = theta[one_hot < 1].view(bs, self.classnum - 1)
                theta_pos = theta[one_hot == 1].view(bs)
                if self.writer:
                    self.writer.add_scalar('theta/pos_med', torch.median(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/pos_min', torch.min(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/pos_max', torch.max(theta_pos).item(), self.step)
                    self.writer.add_histogram('theta/pos', theta_pos, self.step)
                    self.writer.add_histogram('theta/neg', theta_neg, self.step)
                    self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)

                logging.info(f'pos_med: {torch.median(theta_pos).item():.2e} ' +
                             f'neg_med: {torch.median(theta_neg).item():.2e} '
                             )
        output = cos_theta
        cos_theta_need = cos_theta[idx_, label].clone()
        theta = torch.acos(cos_theta_need)
        sin_theta = torch.sin(theta.clamp(0, np.pi / 2))
        cos_theta_m = torch.cos(theta + self.m * (1 - sin_theta) + 0.3)

        # cos_theta_m = torch.cos(theta + self.m * cos_theta + 0.1)

        cond_mask = (cos_theta_need - self.threshold) <= 0
        if torch.any(cond_mask).item():
            logging.info(f'this concatins a difficult sample, {cond_mask.sum().item()}')
        output[idx_, label] = cos_theta_m.type_as(output)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        self.step += 1
        return output


class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=conf.embedding_size, classnum=None, s=conf.scale, m=conf.margin):
        super(Arcface, self).__init__()
        self.classnum = classnum
        kernel = nn.Linear(embedding_size, classnum, bias=False)
        # kernel = Parameter(torch.Tensor(embedding_size, classnum))
        kernel.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # todo
        if conf.spec_norm:
            kernel = nn.utils.spectral_norm(kernel)
        self.kernel = kernel
        self.update_mrg()
        self.easy_margin = False
        self.step = 0
        self.writer = conf.writer
        self.interval = conf.log_interval

    def update_mrg(self, m=conf.margin, s=conf.scale):
        m = np.float32(m)
        pi = np.float32(np.pi)
        dev = conf.model1_dev[0]
        if dev == -1:
            dev = 0
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = torch.FloatTensor([np.cos(m)]).to(dev)
        self.sin_m = torch.FloatTensor([np.sin(m)]).to(dev)
        self.mm = torch.FloatTensor([np.sin(m) * m]).to(dev)
        self.threshold = torch.FloatTensor([math.cos(pi - m)]).to(dev)

    def forward_eff_v1(self, embeddings, label=None):
        assert not torch.isnan(embeddings).any().item()
        bs = embeddings.shape[0]
        idx_ = torch.arange(0, bs, dtype=torch.long)
        if self.interval >= 1 and self.step % self.interval == 0:
            with torch.no_grad():
                norm_mean = torch.norm(embeddings, dim=1).mean()
                if self.writer:
                    self.writer.add_scalar('theta/norm_mean', norm_mean.item(), self.step)
                logging.info(f'norm {norm_mean.item():.2e}')
        embeddings = F.normalize(embeddings, dim=1)
        kernel_norm = l2_norm(self.kernel, axis=0)  # 0 dim is emd dim

        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        if label is None:
            # cos_theta *= self.s # todo whether?
            return cos_theta
        with torch.no_grad():
            if self.interval >= 1 and self.step % self.interval == 0:
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                theta = torch.acos(cos_theta)
                theta_neg = theta[one_hot < 1].view(bs, self.classnum - 1)
                theta_pos = theta[one_hot == 1].view(bs)
                if self.writer:
                    self.writer.add_scalar('theta/pos_med', torch.median(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)
                logging.info(f'pos_med: {torch.median(theta_pos).item():.2e} ' +
                             f'neg_med: {torch.median(theta_neg).item():.2e} '
                             )
        output = cos_theta.clone()  # todo avoid copy ttl
        cos_theta_need = cos_theta[idx_, label]
        cos_theta_2 = torch.pow(cos_theta_need, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta_need * self.cos_m - sin_theta * self.sin_m)
        cond_mask = (cos_theta_need - self.threshold) <= 0

        if torch.any(cond_mask).item():
            logging.info(f'this concatins a difficult sample, {cond_mask.sum().item()}')
        if self.easy_margin:
            keep_val = cos_theta_need
        else:
            keep_val = (cos_theta_need - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask].type_as(cos_theta_m)
        if self.writer and self.step % self.interval == 0:
            # self.writer.add_scalar('theta/cos_th_m_mean', torch.median(cos_theta_m).item(), self.step)
            self.writer.add_scalar('theta/cos_th_m_median', cos_theta_m.mean().item(), self.step)
        output[idx_, label] = cos_theta_m.type_as(output)
        output *= self.s
        self.step += 1

        return output

    def forward_eff_v2(self, embeddings, label=None):
        assert not torch.isnan(embeddings).any().item()
        bs = embeddings.shape[0]
        idx_ = torch.arange(0, bs, dtype=torch.long)
        if self.interval >= 1 and self.step % self.interval == 0:
            with torch.no_grad():
                norm_mean = torch.norm(embeddings, dim=1).mean()
                if self.writer:
                    self.writer.add_scalar('theta/norm_mean', norm_mean.item(), self.step)
                logging.info(f'norm {norm_mean.item():.2e}')
        embeddings = F.normalize(embeddings, dim=1)
        # kernel_norm = l2_norm(self.kernel, axis=0)  # 0 dim is emd dim
        # cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        cos_theta = self.kernel(embeddings)
        cos_theta /= torch.norm(self.kernel.weight, dim=1)
        # torch.norm(cos_theta, dim=1)
        # stat(cos_theta)
        if label is None:
            cos_theta *= self.s
            return cos_theta
        with torch.no_grad():
            if self.interval >= 1 and self.step % self.interval == 0:
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                theta = torch.acos(cos_theta)
                theta_neg = theta[one_hot < 1].view(bs, self.classnum - 1)
                theta_pos = theta[one_hot == 1].view(bs)
                if self.writer:
                    self.writer.add_scalar('theta/pos_med', torch.median(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/pos_min', torch.min(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/pos_max', torch.max(theta_pos).item(), self.step)
                    self.writer.add_histogram('theta/pos', theta_pos, self.step)
                    self.writer.add_histogram('theta/neg', theta_neg, self.step)
                    self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)

                logging.info(f'pos_med: {torch.median(theta_pos).item():.2e} ' +
                             f'neg_med: {torch.median(theta_neg).item():.2e} '
                             )
        output = cos_theta.clone()
        cos_theta_need = cos_theta[idx_, label]
        theta = torch.acos(cos_theta_need)
        cos_theta_m = torch.cos(theta + self.m)
        cond_mask = (cos_theta_need - self.threshold) <= 0
        if torch.any(cond_mask).item():
            logging.info(f'this concatins a difficult sample, {cond_mask.sum().item()}')
        output[idx_, label] = cos_theta_m.type_as(output)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        self.step += 1
        return output

    def forward_eff_v3(self, embeddings, label=None):
        assert not torch.isnan(embeddings).any().item()
        bs = embeddings.shape[0]
        idx_ = torch.arange(0, bs, dtype=torch.long)
        embeddings = F.normalize(embeddings, dim=1)
        kernel_norm = l2_norm(self.kernel, axis=0)  # 0 dim is emd dim
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        if label is None:
            # cos_theta *= self.s # todo whether?
            return cos_theta
        with torch.no_grad():
            if self.interval >= 1 and self.step % self.interval == 0:
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                theta = torch.acos(cos_theta)
                theta_neg = theta[one_hot < 1].view(bs, self.classnum - 1)
                theta_pos = theta[one_hot == 1].view(bs)
                if self.writer:
                    self.writer.add_scalar('theta/pos_med', torch.median(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)
                else:
                    logging.info(f'pos_med: {torch.median(theta_pos).item():.2e} ' +
                                 f'neg_med: {torch.median(theta_neg).item():.2e} '
                                 )
        output = cos_theta
        cos_theta_need = cos_theta[idx_, label].clone()
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta_need, 2))
        cos_theta_m = (cos_theta_need * self.cos_m - sin_theta * self.sin_m)
        cond_mask = (cos_theta_need - self.threshold) <= 0

        if torch.any(cond_mask).item():
            logging.info(f'this concatins a difficult sample, {cond_mask.sum().item()}')
        if self.easy_margin:
            keep_val = cos_theta_need
        else:
            keep_val = (cos_theta_need - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask].type_as(cos_theta_m)
        if self.writer and self.step % self.interval == 0:
            # self.writer.add_scalar('theta/cos_th_m_mean', torch.median(cos_theta_m).item(), self.step)
            self.writer.add_scalar('theta/cos_th_m_median', cos_theta_m.mean().item(), self.step)
        output[idx_, label] = cos_theta_m.type_as(output)
        output *= self.s
        self.step += 1

        return output

    def forward_neff(self, embeddings, label):
        nB = embeddings.shape[0]
        idx_ = torch.arange(0, nB, dtype=torch.long)
        embeddings = F.normalize(embeddings, dim=1)
        cos_theta = self.kernel(embeddings)
        cos_theta /= torch.norm(self.kernel.weight, dim=1)
        # kernel_norm = l2_norm(self.kernel, axis=0)
        # cos_theta = torch.mm(embeddings, kernel_norm)

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

    forward = forward_eff_v2
    # forward = forward_eff_v1
    # forward = forward_neff


class ArcfaceNeg(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=conf.embedding_size, classnum=None, s=conf.scale, m=conf.margin):
        super(ArcfaceNeg, self).__init__()
        self.classnum = classnum
        kernel = Parameter(torch.Tensor(embedding_size, classnum))
        kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.kernel = kernel
        if conf.fp16:
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
        self.threshold2 = math.cos(m)
        self.m2 = conf.margin2
        self.interval = conf.log_interval
        self.step = 0
        self.writer = conf.writer

    def forward_eff(self, embeddings, label=None):
        bs = embeddings.shape[0]
        embeddings = F.normalize(embeddings, dim=1)
        idx_ = torch.arange(0, bs, dtype=torch.long)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        if label is None:
            cos_theta *= self.s
            return cos_theta
        with torch.no_grad():
            if self.interval >= 1 and self.step % self.interval == 0:
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                theta = torch.acos(cos_theta)
                theta_neg = theta[one_hot < 1].view(bs, self.classnum - 1)
                theta_pos = theta[one_hot == 1].view(bs)
                if self.writer:
                    self.writer.add_scalar('theta/pos_med', torch.median(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)
                else:
                    logging.info(f'pos_med: {torch.median(theta_pos).item():.2e} ' +
                                 f'neg_med: {torch.median(theta_neg).item():.2e} '
                                 )
        output = cos_theta
        if self.m != 0:
            cos_theta_need = cos_theta[idx_, label].clone()
            cos_theta_2 = torch.pow(cos_theta_need, 2)
            sin_theta_2 = 1 - cos_theta_2
            sin_theta = torch.sqrt(sin_theta_2)
            cos_theta_m = (cos_theta_need * self.cos_m - sin_theta * self.sin_m)
            cond_mask = (cos_theta_need - self.threshold) <= 0  # those should be replaced
            if torch.any(cond_mask).item():
                logging.info(f'this concatins a difficult sample {cond_mask.sum().item()}')
            keep_val = (cos_theta_need - self.mm)  # when theta not in [0,pi], use cosface instead
            cos_theta_m[cond_mask] = keep_val[cond_mask].type_as(cos_theta_m)
            output[idx_, label] = cos_theta_m.type_as(output)
        if self.m2 != 0:
            with torch.no_grad():
                cos_theta_neg = cos_theta.clone()
                cos_theta_neg[idx_, label] = -self.s * 999
                topk = conf.topk
                topkind = torch.argsort(cos_theta_neg, dim=1)[:, -topk:]
                idx = torch.stack([idx_] * topk, dim=1)
            cos_theta_neg_need = cos_theta_neg[idx, topkind]
            sin_theta_neg = torch.sqrt(1 - torch.pow(cos_theta_neg_need, 2))
            cos_theta_neg_m = (cos_theta_neg_need * np.cos(self.m2) + sin_theta_neg * np.sin(self.m2))
            cond_mask = (cos_theta_neg_need < self.threshold2)  # what is masked is waht should not be replaced
            if torch.any(cos_theta_neg_need >= self.threshold2).item():
                logging.info(f'neg concatins difficult samples {(cos_theta_neg_need >= self.threshold2).sum().item()}')
            cos_theta_neg_need = cos_theta_neg_need.clone()
            cos_theta_neg_need[cond_mask] = cos_theta_neg_m[cond_mask]
            output[idx, topkind] = cos_theta_neg_need.type_as(output)
        output *= self.s
        self.step += 1
        return output

    def forward_eff_v2(self, embeddings, label=None):
        bs = embeddings.shape[0]
        embeddings = F.normalize(embeddings, dim=1)
        idx_ = torch.arange(0, bs, dtype=torch.long)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        if label is None:
            cos_theta *= self.s
            return cos_theta
        with torch.no_grad():
            if self.interval >= 1 and self.step % self.interval == 0:
                one_hot = torch.zeros_like(cos_theta)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                theta = torch.acos(cos_theta)
                theta_neg = theta[one_hot < 1]
                theta_pos = theta[idx_, label]
                if self.writer:
                    self.writer.add_scalar('theta/pos_med', torch.median(theta_pos).item(), self.step)
                    self.writer.add_scalar('theta/neg_med', torch.median(theta_neg).item(), self.step)
                logging.info(f'pos_med: {torch.median(theta_pos).item():.2e} ' +
                             f'neg_med: {torch.median(theta_neg).item():.2e} '
                             )
        output = cos_theta.clone()
        if self.m != 0:
            cos_theta_need = cos_theta[idx_, label]
            theta = torch.acos(cos_theta_need)
            cos_theta_m = torch.cos(theta + self.m)
            cond_mask = (cos_theta_need - self.threshold) <= 0  # those should be replaced
            if torch.any(cond_mask).item():
                logging.info(f'this concatins a difficult sample, {cond_mask.sum().item()}')
                # exit(1)
            output[idx_, label] = cos_theta_m.type_as(output)
        if self.m2 != 0:
            with torch.no_grad():
                cos_theta_neg = cos_theta.clone()
                cos_theta_neg[idx_, label] = -self.s * 999
                topk = conf.topk
                topkind = torch.argsort(cos_theta_neg, dim=1)[:, -topk:]
                idx = torch.stack([idx_] * topk, dim=1)
            cos_theta_neg_need = cos_theta[idx, topkind]
            theta = torch.acos(cos_theta_neg_need)
            cos_theta_neg_m = torch.cos(theta - self.m2)
            cond_mask = (cos_theta_neg_need >= self.threshold2)  # < is masked is what should not be replaced
            if torch.any(cond_mask).item():
                logging.info(f'neg concatins difficult samples '
                             f'{(cond_mask).sum().item()}')
                # exit(1)
            output[idx, topkind] = cos_theta_neg_m.type_as(output)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        self.step += 1
        return output

    forward = forward_eff_v2


class CosFace(Module):
    # class CosFace(jit.ScriptModule):
    __constants__ = ['m', 's']

    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, embedding_size, classnum, s=conf.scale, m=conf.margin):
        super(CosFace, self).__init__()
        self.in_features = embedding_size
        self.out_features = classnum
        # self.s = torch.jit.const(s)
        # self.m = torch.jit.Const(m)
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    # @jit.script_method
    def forward(self, embeddings, label=None):
        embeddings = F.normalize(embeddings, dim=1)
        nB = embeddings.shape[0]
        idx_ = torch.arange(0, nB, dtype=torch.long)
        cosine = F.linear(embeddings, F.normalize(self.weight))
        if label is None:
            return cosine
        phi = cosine[idx_, label] - self.m
        output = cosine.clone()
        output[idx_, label] = phi
        # # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size()).cuda()
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # output = (one_hot * phi) + (
        #         (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'


class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=conf.embedding_size, classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = 0.35  # additive margin recommended by the paper
        self.s = 30.  # see normface https://arxiv.org/abs/1704.06369

    def forward(self, embeddings, label):
        embeddings = F.normalize(embeddings, dim=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
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

    def forward(self, embeddings, targets, return_info=False):
        embeddings = F.normalize(embeddings, dim=1)
        n = embeddings.size(0)  # todo is this version  correct?
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist.addmm(1, -2, embeddings, embeddings.t()).clamp(min=1e-6).sqrt() * conf.scale
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
        loss_indiv = F.softplus(dist_ap - dist_an)
        loss = loss_indiv.mean()
        if not return_info:
            return loss
        else:
            info = {'dap': dist_ap.mean().item(), 'dan': dist_an.mean().item(), 'indiv': loss_indiv}
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


if __name__ == '__main__':
    from lz import *
    from pathlib import Path

    init_dev(3)
    # conf.input_size = 112
    # conf.embedding_size = 512
    # conf.bottle_neck = False
    # conf.mbfc_se = False
    # conf.net_depth = 18
    # conf.out_type = 'fc'
    # conf.mid_type = ''  # 'gpool'
    model = Backbone(conf.net_depth, 0.3, 'ir_se', ebsize=512)
    # model = MobileFaceNet(2048,
    #                       width_mult=1,
    #                       depth_mult=1,)
    model.eval()
    model.cuda()
    print(model)
    # params = []
    # # wmdm = "1.0,2.25 1.1,1.86 1.2,1.56 1.3,1.33 1.4,1.15 1.5,1.0".split(' ') # 1,2 1.56,2  1.0,1.0
    # wmdm = "1.2,1.56".split(' ')
    # wmdm = [(float(wd.split(',')[0]), float(wd.split(',')[1])) for wd in wmdm]
    # for wd in wmdm:
    #     wm, dm = wd
    #     model = MobileFaceNet(512,
    #                           width_mult=wm,
    #                           depth_mult=dm,
    #                           ).cuda()
    #     model.eval()
    #     print('mbfc:\n', model)
    #     ttl_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    #     print('Total params: %.2fM' % ttl_params)
    #     params.append(ttl_params)
    # print(params)
    # plt.plot(wms, params)
    # plt.show()
    # dms = np.arange(1, 2, .01)
    # params2 = []
    # for dm in dms:
    #     model = MobileFaceNet(512,
    #                           width_mult=1.,
    #                           depth_mult=dm,
    #                           ).cuda()
    #     model.eval()
    #     print('mobilenetv3:\n', model)
    #     ttl_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    #     params2.append(ttl_params)
    #     print('Total params: %.2fM' % ttl_params)
    # plt.plot(dms, params2)
    # plt.show()

    from thop import profile
    from lz import timer

    input = torch.randn(1, 3, conf.input_size, conf.input_size).cuda()
    flops, params = profile(model, inputs=(input,),
                            # only_ops=(nn.Conv2d, nn.Linear),
                            verbose=False,
                            )
    flops /= 10 ** 9
    params /= 10 ** 6
    print(params, flops, )
    exit(1)
    classifier = AdaMArcface(classnum=10).cuda()
    classifier.train()
    model.train()
    bs = 32
    input_size = (bs, 3, 112, 112)
    target = to_torch(np.random.randint(low=0, high=10, size=(bs,)), ).cuda()
    x = torch.rand(input_size).cuda()

    out = model(x, )
    logits, mrg_mn = classifier(out, target)
    loss = nn.CrossEntropyLoss()(logits, target)
    (loss - 10 * mrg_mn).backward()
    print(loss, classifier.m.grad)

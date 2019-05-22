# -*- coding: future_fstrings -*-

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
from config import conf as gl_conf
import functools, logging
from torch import nn, jit
import numpy as np

if gl_conf.use_chkpnt:
    BatchNorm2d = functools.partial(BatchNorm2d, momentum=1 - np.sqrt(0.9))


class Flatten(Module):
    # class Flatten(jit.ScriptModule):
    #     @jit.script_method
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1, need_norm=False, ):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    if need_norm:
        return output, norm
    else:
        return output


# class SEModule(jit.ScriptModule):
class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.relu = PReLU(channels // reduction) if gl_conf.upgrade_irse else ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    # @jit.script_method
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


from modules.bn import InPlaceABN, InPlaceABNSync


def bn_act(depth, with_act, ipabn=None):
    if ipabn:
        if with_act:
            if ipabn == 'sync':
                return [InPlaceABNSync(depth, activation='none'),
                        PReLU(depth, ), ]
            else:
                return [InPlaceABN(depth, activation='none'),
                        PReLU(depth, ), ]
        else:
            return [InPlaceABN(depth, activation='none')]
    else:
        if with_act:
            return [BatchNorm2d(depth), PReLU(depth, ), ]
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


# todo gl_conf.upgrade_irse
class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        ipabn = gl_conf.ipabn
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        if gl_conf.upgrade_irse:
            self.res_layer = Sequential(
                *bn_act(in_channel, False, ipabn),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                *bn_act(depth, True, ipabn),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                *bn_act(depth, False, ipabn))
        else:
            self.res_layer = Sequential(
                BatchNorm2d(in_channel),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    # @jit.script_method
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Identity(Module):
    # class Identity(jit.ScriptModule):
    #     @jit.script_method
    def forward(self, x):
        return x


class bottleneck_IR_SE(Module):
    # class bottleneck_IR_SE(jit.ScriptModule):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        self.ipabn = int(bool(gl_conf.ipabn))
        if gl_conf.upgrade_irse and in_channel == depth and stride == 1:
            self.shortcut_layer = Identity()
        elif not gl_conf.upgrade_irse and in_channel == depth:
            self.shortcut_layer = MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                *bn_act(depth, False, gl_conf.ipabn)
            )
        if gl_conf.upgrade_irse:
            self.res_layer = Sequential(
                *bn_act(in_channel, False, gl_conf.ipabn),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                *bn_act(depth, True, gl_conf.ipabn),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                *bn_act(depth, False, gl_conf.ipabn),
                SEModule(depth, 16)
            )
        else:
            self.res_layer = Sequential(
                BatchNorm2d(in_channel),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                PReLU(depth),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                BatchNorm2d(depth),
                SEModule(depth, 16)
            )

    # @jit.script_method
    def forward_ipabn(self, x):
        shortcut = self.shortcut_layer(x.clone())
        res = self.res_layer(x)
        res.add_(shortcut)
        return res

    # @jit.script_method
    def forward_ori(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        res.add_(shortcut)
        return res

    if gl_conf.ipabn:
        forward = forward_ipabn
    else:
        forward = forward_ori


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
    elif num_layers == 20:  # this is 26 in fact!
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=3),
            get_block(in_channel=128, depth=256, num_units=5),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    return blocks


from torch.utils.checkpoint import checkpoint_sequential


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir', ebsize=gl_conf.embedding_size):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152, 20], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      bn2d(64, gl_conf.ipabn),
                                      PReLU(64))

        self.output_layer = Sequential(bn2d(512, gl_conf.ipabn),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, ebsize, bias=True if not gl_conf.upgrade_bnneck else False),
                                       BatchNorm1d(ebsize))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        self._initialize_weights()

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


##################################  MobileFaceNet

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
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = bn2d(out_c, gl_conf.ipabn)
        self.prelu = PReLU(out_c)

    # @jit.script_method
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
        self.bn = bn2d(out_c, gl_conf.ipabn)

    # @jit.script_method
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

    # @jit.script_method
    def forward(self, x, ):
        return self.model(x)


# class MobileFaceNet(jit.ScriptModule):
class MobileFaceNet(Module):
    def __init__(self, embedding_size, mode='large'):
        super(MobileFaceNet, self).__init__()
        global Conv2d
        if mode == 'small':
            blocks = [1, 4, 6, 2]
        else:
            blocks = [2, 8, 16, 4]
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        if blocks[0] == 1:
            self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        else:
            self.conv2_dw = Residual(64, num_block=blocks[0], groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        # Conv2d = STNConv
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        Conv2d = nn.Conv2d

        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

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
        # from IPython import embed; embed()
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
        return F.normalize(out, dim=1)


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


# nB = gl_conf.batch_size
# idx_ = torch.arange(0, nB, dtype=torch.long)

class Arcface2(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=gl_conf.embedding_size, classnum=None, s=gl_conf.scale, m=gl_conf.margin):
        super(Arcface2, self).__init__()
        self.classnum = classnum
        kernel = Parameter(torch.Tensor(embedding_size, classnum))
        kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # kernel = torch.chunk(kernel, gl_conf.num_devs, dim=1)
        self.device_id = list(range(gl_conf.num_devs))
        # kernel = tuple(kernel[ind].cuda(self.device_id[ind]) for ind in range(gl_conf.num_devs))
        self.kernel = kernel

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
        self.easy_margin = False

    def forward_eff(self, embbedings, label=None):
        assert not torch.isnan(embbedings).any().item()
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
        theta = torch.acos(cos_theta_need)
        cos_theta_m = torch.cos(theta + self.m)
        cond_mask = (cos_theta_need - self.threshold) <= 0

        if torch.any(cond_mask).item():
            logging.info(f'this concatins a difficult sample, {cond_mask.sum().item()}')
            # from IPython import embed; embed()
        output[idx_, label] = cos_theta_m.type_as(output)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output

    forward = forward_eff


class AdaCos(nn.Module):
    def __init__(self, num_classes=None, m=.0, num_features=gl_conf.embedding_size):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)
        self.device_id = list(range(gl_conf.num_devs))
        self.step = 0
        self.writer = gl_conf.writer
        assert self.writer is not None

    def forward(self, input, label=None):
        x = F.normalize(input, dim=1)
        W = F.normalize(self.W, dim=1)
        logits = F.linear(x, W)
        logits = logits.float()
        # sub_weights = torch.chunk(self.W, gl_conf.num_devs, dim=0)  # (ncls//4,nfeas)
        # temp_x = x.cuda(self.device_id[0])  # (bs,nfeas)
        # weight = sub_weights[0].cuda(self.device_id[0])
        # cos_theta_l = [F.linear(temp_x, F.normalize(weight, dim=1))]  # (bs, ncls//4)
        # for i in range(1, len(self.device_id)):
        #     temp_x = x.cuda(self.device_id[i])
        #     weight = sub_weights[i].cuda(self.device_id[i])
        #     cos_theta_l.append(
        #         F.linear(temp_x, F.normalize(weight, dim=1)).cuda(self.device_id[0])
        #     )
        # logits = torch.cat(cos_theta_l, dim=1)  # (bs,ncls)
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
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta + self.m)
            if self.step % 999 == 0:
                self.writer.add_scalar('info/th_med', theta_med.item(), self.step)
            if self.step % 9999 == 0:
                self.writer.add_histogram('info/th', theta, self.step)
            self.step += 1

            s_now = torch.log(B_avg) / torch.cos(torch.min(
                (math.pi / 4 + self.m) * torch.ones_like(theta_med),
                theta_med))
            # self.s = self.s * 0.9 + s_now * 0.1
            self.s = s_now
        output *= self.s
        return output


class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=gl_conf.embedding_size, classnum=None, s=gl_conf.scale, m=gl_conf.margin):
        super(Arcface, self).__init__()
        self.classnum = classnum
        kernel = Parameter(torch.Tensor(embedding_size, classnum))
        kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # kernel = torch.chunk(kernel, gl_conf.num_devs, dim=1)
        self.device_id = list(range(gl_conf.num_devs))
        # kernel = tuple(kernel[ind].cuda(self.device_id[ind]) for ind in range(gl_conf.num_devs))
        self.kernel = kernel
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
        self.easy_margin = False

    def forward_eff(self, embbedings, label=None):
        assert not torch.isnan(embbedings).any().item()
        # assert (torch.norm(embbedings, dim=1) == 1).all().item()
        nB = embbedings.shape[0]
        idx_ = torch.arange(0, nB, dtype=torch.long)
        # if gl_conf.num_devs == 0:
        kernel_norm = l2_norm(self.kernel, axis=0)  # 0 dim is emd dim
        cos_theta = torch.mm(embbedings, kernel_norm)
        # else:
        #     x = embbedings
        #     sub_weights = torch.chunk(self.kernel, gl_conf.num_devs, dim=1)
        #     temp_x = embbedings.cuda(self.device_id[0])
        #     weight = sub_weights[0].cuda(self.device_id[0])
        #     cos_theta = torch.mm(temp_x, F.normalize(weight, dim=0))
        #     for i in range(1, len(self.device_id)):
        #         temp_x = x.cuda(self.device_id[i])
        #         weight = sub_weights[i].cuda(self.device_id[i])
        #         cos_theta = torch.cat(
        #             (cos_theta,
        #              torch.mm(temp_x, F.normalize(weight, dim=0)).cuda(self.device_id[0])),
        #             dim=1)
        cos_theta = cos_theta.clamp(-1, 1)
        if label is None:
            # cos_theta *= self.s # todo
            return cos_theta
        output = cos_theta.clone()  # todo avoid copy ttl
        cos_theta_need = cos_theta[idx_, label]
        cos_theta_2 = torch.pow(cos_theta_need, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta_need * self.cos_m - sin_theta * self.sin_m)
        cond_mask = (cos_theta_need - self.threshold) <= 0

        if torch.any(cond_mask).item():
            logging.info(f'this concatins a difficult sample, {cond_mask.sum().item()}')
            # from IPython import embed; embed()
        if self.easy_margin:
            keep_val = cos_theta_need
        else:
            keep_val = (cos_theta_need - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask].type_as(cos_theta_m)
        output[idx_, label] = cos_theta_m.type_as(output)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output

    def forward_neff(self, embbedings, label):
        nB = embbedings.shape[0]
        idx_ = torch.arange(0, nB, dtype=torch.long)
        ## weights norm
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)

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


class ArcfaceNeg(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=gl_conf.embedding_size, classnum=None, s=gl_conf.scale, m=gl_conf.margin):
        super(ArcfaceNeg, self).__init__()
        self.classnum = classnum
        kernel = Parameter(torch.Tensor(embedding_size, classnum))
        kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # kernel = torch.chunk(kernel, gl_conf.num_devs, dim=1)
        self.device_id = list(range(gl_conf.num_devs))
        # kernel = tuple(kernel[ind].cuda(self.device_id[ind]) for ind in range(gl_conf.num_devs))
        self.kernel = kernel

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
        self.threshold2 = math.cos(m)
        self.m2 = gl_conf.margin2

    def forward_eff(self, embbedings, label=None):
        nB = embbedings.shape[0]
        idx_ = torch.arange(0, nB, dtype=torch.long)
        if gl_conf.num_devs == 0:
            kernel_norm = l2_norm(self.kernel, axis=0)
            cos_theta = torch.mm(embbedings, kernel_norm)
        else:
            x = embbedings
            sub_weights = torch.chunk(self.kernel, gl_conf.num_devs, dim=1)
            temp_x = embbedings.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cos_theta = torch.mm(temp_x, F.normalize(weight, dim=0))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cos_theta = torch.cat(
                    (cos_theta,
                     torch.mm(temp_x, F.normalize(weight, dim=0)).cuda(self.device_id[0])),
                    dim=1)
        cos_theta = cos_theta.clamp(-1, 1)
        if label is None:
            cos_theta *= self.s
            return cos_theta
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
            cos_theta_neg = cos_theta.clone()
            cos_theta_neg[idx_, label] = -self.s
            topk = gl_conf.topk
            topkind = torch.argsort(cos_theta_neg, dim=1)[:, -topk:]
            idx = torch.stack([idx_] * topk, dim=1)
            cos_theta_neg_need = cos_theta_neg[idx, topkind]
            cos_theta_neg_need_2 = torch.pow(cos_theta_neg_need, 2)
            sin_theta_neg_2 = 1 - cos_theta_neg_need_2
            sin_theta_neg = torch.sqrt(sin_theta_neg_2)
            cos_theta_neg_m = (cos_theta_neg_need * np.cos(self.m2) + sin_theta_neg * np.sin(self.m2))
            cond_mask = (cos_theta_neg_need < self.threshold2)  # what is masked is waht should not be replaced
            if torch.any(cos_theta_neg_need >= self.threshold2).item():
                logging.info(f'neg concatins difficult samples {(cos_theta_neg_need >= self.threshold2).sum().item()}')
            cos_theta_neg_need = cos_theta_neg_need.clone()
            cos_theta_neg_need[cond_mask] = cos_theta_neg_m[cond_mask]
            output[idx, topkind] = cos_theta_neg_need.type_as(output)
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output

    forward = forward_eff


##################################  Cosface head #################
import torch.jit
from torch import jit


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

    def __init__(self, embedding_size, classnum, s=gl_conf.scale, m=gl_conf.margin):
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
    def forward(self, input, label=None):
        # assert not torch.isnan(input).any().item()
        nB = input.shape[0]
        idx_ = torch.arange(0, nB, dtype=torch.long)
        cosine = F.linear((input), F.normalize(self.weight))
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

    # model = Backbone(50, 0, 'ir_se').cuda()
    model = MobileFaceNet(512).cuda()
    model.eval()
    # model2 = torch.jit.trace(model, torch.rand(2, 3, 112, 112).cuda())
    # model2.eval()
    # # model.train(), model2.train()
    #
    # inp = torch.rand(32, 3, 112, 112).cuda()
    # model(inp)
    # diff = model(inp) - model2(inp)
    # print(diff, diff.sum())
    #
    # timer.since_last_check('start')
    # for _ in range(99):
    #     f = model(torch.rand(32, 3, 112, 112).cuda())
    #     f.mean().backward()
    # torch.cuda.synchronize()
    # timer.since_last_check('100 times')
    #
    # timer.since_last_check('start')
    # for _ in range(99):
    #     f = model2(torch.rand(32, 3, 112, 112).cuda())
    #     f.mean().backward()
    # torch.cuda.synchronize()
    # timer.since_last_check('100 times')
    #
    # exit()

    from thop import profile
    from lz import timer

    flops, params = profile(model, input_size=(1, 3, 112, 112),
                            custom_ops={DoubleConv: count_double_conv},
                            device='cuda:0',
                            )
    flops /= 10 ** 9
    params /= 10 ** 6

    for i in range(5):
        img = torch.rand(1, 3, 112, 112).cuda()
        f = model(img)
        f.mean().backward()

    timer.since_last_check()
    for i in range(100):
        img = torch.rand(1, 3, 112, 112).cuda()
        f = model(img)
        f.mean().backward()
    interval = timer.since_last_check('finish')
    interval /= 100
    print(flops, params, interval)

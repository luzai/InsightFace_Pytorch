from lz import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import conf

__all__ = ['singlepath']

use_hard = False


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if use_hard:
            res = x * (F.relu6(x + 3., inplace=self.inplace) / 6.)
        else:
            res = x * torch.sigmoid(x)
        # assert not torch.isnan(res).any().item()
        if torch.isnan(res).any().item():
            raise ValueError()
        return res


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if use_hard:
            res = F.relu6(x + 3., inplace=self.inplace) / 6.
        else:
            res = torch.sigmoid(x)
        # assert not torch.isnan(res).any().item()
        if torch.isnan(res).any().item():
            raise ValueError()
        return res


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def Indicator(x):
    sigx = F.sigmoid(x)
    return ((x >= 0).float() - sigx).detach() + sigx


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6?
        elif nl == 'PRE':
            nlin_layer = lambda inplace: nn.PReLU()
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if torch.isnan(x).any().item():
            raise ValueError()
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def judgenan(x):
    return not not torch.isnan(x).any().item()


class Conv2dMask(nn.Module):
    def __init__(self, exp, kernel, stride, padding, groups=None, bias=False):
        super(Conv2dMask, self).__init__()
        self.padding = padding
        self.stride = stride
        self.exp = exp
        self.name = f'{exp}'
        self.t5x5 = nn.Parameter(torch.FloatTensor([0.]))
        self.t50c = nn.Parameter(torch.FloatTensor([0.]))
        self.t100c = nn.Parameter(torch.FloatTensor([0.]))
        self.depthwise_kernel = nn.Parameter(torch.FloatTensor(exp, 1, 5, 5))  # out, in, k, k
        nn.init.kaiming_normal_(self.depthwise_kernel, mode='fan_out')

        mask3x3 = np.zeros((exp, 1, 5, 5), dtype='float32')
        mask3x3[:, :, 1:4, 1:4] = 1.
        mask5x5 = np.ones((exp, 1, 5, 5), dtype='float32') - mask3x3
        self.mask3x3 = to_torch(mask3x3).cuda()
        self.mask5x5 = to_torch(mask5x5).cuda()

        num_channels = int(exp)
        c50 = int(round(1.0 * num_channels / 2.0))  # 50 %
        c100 = int(round(2.0 * num_channels / 2.0))  # 100 %
        mask_50c = np.zeros((exp, 1, 5, 5), dtype='float32')
        mask_50c[0:c50, ...] = 1.0  # from 0% to 50% channels
        mask_100c = np.zeros((exp, 1, 5, 5), dtype='float32')
        mask_100c[c50:c100, ...] = 1.0  # from 50% to 100% channels !!
        self.mask50c = to_torch(mask_50c).cuda()
        self.mask100c = to_torch(mask_100c).cuda()
        self.runtimes = None
        self.step = 0
        self.interval = 999
        self.writer = conf.writer

    def forward(self, x):
        kernel_3x3 = self.depthwise_kernel * self.mask3x3
        kernel_5x5 = self.depthwise_kernel * self.mask5x5
        norm5x5 = torch.norm(kernel_5x5)
        x5x5 = norm5x5 - self.t5x5
        dropout_rate = conf.conv2dmask_drop_ratio
        d5x5 = F.dropout(Indicator(x5x5), dropout_rate)  # note: dropout scale the value
        depthwise_kernel_masked_outside = kernel_3x3 + kernel_5x5 * d5x5

        kernel_50c = depthwise_kernel_masked_outside * self.mask50c
        kernel_100c = depthwise_kernel_masked_outside * self.mask100c
        norm50c = torch.norm(kernel_50c)
        norm100c = torch.norm(kernel_100c)

        x100c = norm100c - self.t100c
        d100c = F.dropout(Indicator(x100c), dropout_rate)
        if not self.runtimes:
            flops = x.shape[1] ** 2 * 5 ** 2 * self.exp / 10 ** 6
            self.runtimes = [0, x.shape[1] ** 2 * 3 ** 2 * self.exp / 10 ** 6, flops / 2, flops, 0.7]
            self.R50c = self.runtimes[2]
            self.R5x5 = self.R100c = self.runtimes[3]
            self.R3x3 = self.runtimes[1]

        if self.stride == 1 and len(self.runtimes) == 5:
            x50c = norm50c - self.t50c
            d50c = F.dropout(Indicator(x50c), dropout_rate)
        else:
            d50c = 1.
        depthwise_kernel_masked = d50c * (kernel_50c + d100c * kernel_100c)

        ratio = self.R3x3 / self.R5x5
        runtime_channels = d50c * (self.R50c + d100c * (self.R100c - self.R50c))
        runtime_reg = runtime_channels * ratio + runtime_channels * (1 - ratio) * d5x5
        conf.conv2dmask_runtime_reg.append(runtime_reg)
        output = F.conv2d(
            input=x, weight=depthwise_kernel_masked, stride=self.stride,
            padding=self.padding, groups=self.exp,
        )
        self.step += 1
        if self.step % self.interval == 0:
            self.writer.add_scalar(f'sglpth/{self.name}/d5x5', d5x5.item())
            self.writer.add_scalar(f'sglpth/{self.name}/d50c', d50c.item())
            self.writer.add_scalar(f'sglpth/{self.name}/d5x5', d100c.item())
            self.writer.add_scalar(f'sglpth/{self.name}/rtreg', runtime_reg.item())
        if torch.isnan(output).any().item():
            raise ValueError()
        return output


class MobileBottleneck5x5(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck5x5, self).__init__()

        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6?
        elif nl == 'PRE':
            nlin_layer = lambda inplace: nn.PReLU()
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            # conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            Conv2dMask(exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MobileNetV3(nn.Module):
    def __init__(self, n_class=None, mode='face.large', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 512
        mobile_setting = [
            # k, exp, c,  se,     nl,  s,
            [3, 16, 16, False, 'RE', 1],
            [3, 16, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1],
            [5, 72, 40, True, 'RE', 2],
            [5, 120, 40, True, 'RE', 1],
            [5, 120, 40, True, 'RE', 1],
            [3, 240, 80, False, 'PRE', 2],
            [3, 200, 80, False, 'PRE', 1],
            [3, 184, 80, False, 'PRE', 1],
            [3, 184, 80, False, 'PRE', 1],
            [3, 480, 112, True, 'PRE', 1],
            [3, 672, 112, True, 'PRE', 1],
            [5, 672, 112, True, 'PRE', 1],  # c = 112, paper set it to 160 by error
            [5, 672, 160, True, 'PRE', 2],
            [5, 960, 160, True, 'PRE', 1],  # HS
        ]
        # building first layer
        # assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 16!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 1, nlin_layer=lambda inplace: nn.PReLU()  # Hswish
                                 )]

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            k = 5
            if exp / c < 5.9:
                exp = exp * 2
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            if k == 5:
                self.features.append(MobileBottleneck5x5(input_channel, output_channel, k, s, exp_channel, se, nl))
            else:
                self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        last_conv = make_divisible(960 * width_mult)
        self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=lambda inplace: nn.PReLU()  # Hswish
                                         ))
        self.pool = Linear_block(last_conv, last_conv, groups=last_conv, kernel=(7, 7), stride=(1, 1),
                                 padding=(0, 0))
        # self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.linear = nn.Linear(last_conv, last_channel, bias=False)
        self.bn = nn.BatchNorm1d(last_channel)

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x, *args, **kwargs):
        # bs, nc, nh, nw = x.shape
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        assert not torch.isnan(x).any().item()
        x = self.linear(x)
        assert not torch.isnan(x).any().item()
        x = self.bn(x)
        x = F.normalize(x, dim=1)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def singlepath(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    init_dev(3)
    net = singlepath(
        # width_mult=1.285,
        width_mult=1.15,
    )
    # state_dict = torch.load('mobilenetv3_small_67.218.pth.tar')
    # net.load_state_dict(state_dict)
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

    input_size = (16, 3, 112, 112)
    net = net.cuda()
    x = torch.randn(input_size).cuda()
    out = net(x)

    # from thop import profile
    # from lz import timer
    #
    # model = net.to('cuda:0')
    # flops, params = profile(model, input_size=(2, 3, 112, 112),
    #                         device='cuda:0',
    #                         )
    # flops /= 10 ** 9
    # params /= 10 ** 6
    # print(flops, params)

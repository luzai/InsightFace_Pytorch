from lz import *
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileNetV3', 'mobilenetv3']

use_hard = True


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
        assert not torch.isnan(res).any().item()
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
        assert not torch.isnan(res).any().item()
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
            nlin_layer = nn.ReLU  # or ReLU6
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
    def __init__(self, n_class=None, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 512
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 112, True, 'HS', 1],  # c = 112, paper set it to 160 by error
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
            ]
        elif mode == 'face.large':
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 112, True, 'HS', 1],  # c = 112, paper set it to 160 by error
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],  # stride = 2, paper set it to 1 by error
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
        elif mode == 'face.small':
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'RE', 1],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],  # stride = 2, paper set it to 1 by error
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
        else:
            raise NotImplementedError
        # building first layer
        # assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 16!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        if 'face' in mode:
            self.features = [conv_bn(3, input_channel, 1, nlin_layer=Hswish)]
        else:
            self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(Hswish(inplace=True))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
            self.features.append(nn.Conv2d(last_channel, n_class, 1, 1, 0))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(SEModule(last_conv))  # refer to paper Table2
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(Hswish(inplace=True))
            self.features.append(conv_1x1_bn(last_conv, last_channel, nlin_layer=Hswish))
            self.features.append(conv_1x1_bn(last_channel, n_class, nlin_layer=Hswish))
        elif mode == 'face.large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.pool = Linear_block(last_conv, last_conv, groups=last_conv,  kernel=(7, 7), stride=(1, 1), padding=(0, 0))
            # self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = Flatten()
            self.linear = nn.Linear(last_conv, last_channel, bias=False)
            self.bn = nn.BatchNorm1d(last_channel)
        elif mode == 'face.small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(SEModule(last_conv))  # refer to paper Table2
            self.pool = Linear_block(last_conv, last_conv, groups=last_conv,
                                     kernel=(7, 7), stride=(1, 1), padding=(0, 0))
            self.flatten = Flatten()
            self.linear = nn.Linear(last_conv, last_channel, bias=False)
            self.bn = nn.BatchNorm1d(last_channel)
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x, *args, **kwargs):
        # bs, nc, nh, nw = x.shape
        # if nh == 112:
        #     x = F.upsample_bilinear(x, scale_factor=2)
        # todo
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


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    init_dev(3)
    net = mobilenetv3(mode='face.large',
                      width_mult=1.285, )
    # state_dict = torch.load('mobilenetv3_small_67.218.pth.tar')
    # net.load_state_dict(state_dict)
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    input_size = (16, 3, 112, 112)
    x = torch.randn(input_size)
    out = net(x)
    exit()
    from thop import profile
    from lz import timer

    model = net.cuda()
    flops, params = profile(model, input_size=(1, 3, 112, 112),
                            device='cuda:0',
                            )
    flops /= 10 ** 9
    params /= 10 ** 6
    print(flops, params)

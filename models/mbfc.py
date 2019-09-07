from lz import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import conf

__all__ = ['mbfc']

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
            nlin_layer = lambda inplace: nn.PReLU(exp)
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.pw_conv = conv_layer(inp, exp, 1, 1, 0, bias=False)
        self.pw_norm = norm_layer(exp)
        self.pw_nlin = nlin_layer(inplace=True)
        self.dw_conv = conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False)
        self.dw_norm = norm_layer(exp)
        self.dw_se = SELayer(exp)
        self.dw_nlin = nlin_layer(inplace=True)
        self.pwl_conv = conv_layer(exp, oup, 1, 1, 0, bias=False)
        self.pwl_norm = norm_layer(oup)

    def forward(self, x):
        assert not torch.isnan(x).any().item()
        xx = self.pw_conv(x)
        xx = self.pw_norm(xx)
        xx = self.pw_nlin(xx)
        xx = self.dw_conv(xx)
        xx = self.dw_norm(xx)
        xx = self.dw_se(xx)
        xx = self.dw_nlin(xx)
        xx = self.pwl_conv(xx)
        xx = self.pwl_norm(xx)
        if self.use_res_connect:
            res = x + xx
        else:
            res = xx
        if self.use_res_connect: # todo
            xx = self.pw_conv(res)
            xx = self.pw_norm(xx)
            xx = self.pw_nlin(xx)
            xx = self.dw_conv(xx)
            xx = self.dw_norm(xx)
            xx = self.dw_se(xx)
            xx = self.dw_nlin(xx)
            xx = self.pwl_conv(xx)
            xx = self.pwl_norm(xx)
            res = res + xx
        return res


def judgenan(x):
    return not not torch.isnan(x).any().item() or not not torch.isinf(x).any().item()


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


class MBFC(nn.Module):
    def __init__(self,
                 width_mult=conf.mbfc_wm,
                 depth_mult=conf.mbfc_dm, ):
        super(MBFC, self).__init__()
        input_channel = make_divisible(64 * width_mult)
        last_channel = 512
        blocks = [1, 4, 8, 2]
        blocks = [make_divisible(b * depth_mult, 1) for b in blocks]
        mobile_setting = [
            # k, exp, c,  se,     nl,  s,
            *([[3, 64, 64, False, 'PRE', 1]] * blocks[0]),
            [3, 128, 64, False, 'PRE', 2],
            *([[3, 128, 64, False, 'PRE', 1]] * blocks[1]),
            [3, 256, 128, False, 'PRE', 2],
            *([[3, 256, 128, False, 'PRE', 1]] * blocks[2]),
            [3, 512, 128, False, 'PRE', 2],
            *([[3, 256, 128, False, 'PRE', 1]] * blocks[3]),
        ]
        # building first layer
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.head = conv_bn(3, input_channel, 2, nlin_layer=lambda inplace: nn.PReLU(input_channel))  # Hswish
        self.features = []
        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            se = False  # todo
            nl = 'PRE'
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        last_conv = make_divisible(512 * width_mult)  # todo or just last_channel/512
        self.tail = conv_1x1_bn(input_channel, last_conv, nlin_layer=lambda inplace: nn.PReLU(last_conv))  # Hswish
        self.pool = Linear_block(last_conv, last_conv, groups=last_conv, kernel=(7, 7), stride=(1, 1),
                                 padding=(0, 0))
        self.flatten = Flatten()
        self.linear = nn.Linear(last_conv, last_channel, bias=False)
        self.bn = nn.BatchNorm1d(last_channel)

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()
        self.const_one = to_torch(np.ones(1, 'float32')).cuda()
        self.const_zero = to_torch(np.zeros(1, 'float32')).cuda()

    def forward(self, x, *args, **kwargs):
        # bs, nc, nh, nw = x.shape
        device = x.device
        ttl_runtime_reg = 0
        x = self.head(x)
        x = self.features(x)
        x = self.tail(x)
        x = self.pool(x)
        x = self.flatten(x)
        assert not torch.isnan(x).any().item()
        x = self.linear(x)
        assert not torch.isnan(x).any().item()
        x = self.bn(x)
        # x = F.normalize(x, dim=1)
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

    def get_decisions(self):
        decs = []
        for mb5x5 in self.features:
            dec = mb5x5.get_decision()
            decs.append(dec)
        return decs


def mbfc(**kwargs):
    model = MBFC(**kwargs)
    return model


if __name__ == '__main__':
    from config import conf

    conf.conv2dmask_drop_ratio = 0
    init_dev((2))
    net = mbfc(
        width_mult=1,
    )
    # state_dict = torch.load('mobilenetv3_small_67.218.pth.tar')
    # net.load_state_dict(state_dict)
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net = net.cuda()
    net = nn.DataParallel(net).cuda()
    net.train()
    classifier = nn.Linear(512, 10).cuda()
    classifier.train()
    opt = torch.optim.SGD(list(net.parameters()) + list(classifier.parameters()), lr=1e-1)

    bs = 4
    input_size = (bs, 3, 112, 112)
    target = to_torch(np.random.randint(low=0, high=10, size=(bs,)), ).cuda()
    x = torch.rand(input_size).cuda()

    # from thop import profile
    # flops, params = profile(net, input_size=(2, 3, 112, 112),
    #                         device='cuda:0',
    #                         )
    # flops /= 10 ** 9
    # params /= 10 ** 6
    # print(flops, params)

    # exit()
    for i in range(99):
        print(' forward ----- ', i)
        opt.zero_grad()
        ttl_runtime = 0.452 * 10 ** 6
        target_runtime = 2.5 * 10 ** 6
        out = net(x, )
        logits = classifier(out)
        loss = nn.CrossEntropyLoss()(logits, target)
        (loss).backward()
        opt.step()
        print(' now loss: ', loss.item(), )

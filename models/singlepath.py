from lz import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import conf

__all__ = ['singlepath']

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
    return not not torch.isnan(x).any().item() or not not torch.isinf(x).any().item()


def Indicator(x):
    sigx = torch.sigmoid(x)
    return ((x >= 0).float() - sigx).detach() + sigx


def my_dropout(x, drop_ratio):
    # x = F.dropout(Indicator(x), drop_ratio) * (1 - drop_ratio)

    x = F.dropout(Indicator(x), drop_ratio)  # note: dropout scale the value

    # if np.random.rand() < drop_ratio: # drop
    #     x = torch.FloatTensor([0]).cuda()
    # else:
    #     x = x
    return x


class SuperKernel(nn.Module):
    def __init__(self, exp, kernel, stride, padding,
                 use_res_connect=True):
        super(SuperKernel, self).__init__()
        self.padding = padding
        self.stride = stride
        self.exp = exp
        self.use_res_connect = use_res_connect
        self.name = f'{exp}-{randomword(3)}'
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
        self.const_one = to_torch(np.ones(1, 'float32')).cuda()
        self.runtimes = None

    def get_decision(self):
        with torch.no_grad():
            kernel_3x3 = self.depthwise_kernel * self.mask3x3
            kernel_5x5 = self.depthwise_kernel * self.mask5x5
            norm5x5 = torch.norm(kernel_5x5)
            x5x5 = norm5x5 - self.t5x5
            d5x5 = Indicator(x5x5)
            depthwise_kernel_masked_outside = kernel_3x3 + kernel_5x5 * d5x5
            kernel_50c = depthwise_kernel_masked_outside * self.mask50c
            kernel_100c = depthwise_kernel_masked_outside * self.mask100c
            norm50c = torch.norm(kernel_50c)
            norm100c = torch.norm(kernel_100c)
            x100c = norm100c - self.t100c
            d100c = Indicator(x100c)
            if self.stride == 1 and self.use_res_connect:
                x50c = norm50c - self.t50c
                d50c = Indicator(x50c)
            else:
                d50c = self.const_one
        return (d5x5, d100c, d50c)

    def build_kernel(self):
        dropout_rate = conf.conv2dmask_drop_ratio
        device = self.depthwise_kernel.device
        kernel_3x3 = self.depthwise_kernel * self.mask3x3.to(device)
        kernel_5x5 = self.depthwise_kernel * self.mask5x5.to(device)
        norm5x5 = torch.norm(kernel_5x5)
        x5x5 = norm5x5 - self.t5x5
        d5x5 = my_dropout(Indicator(x5x5), dropout_rate)
        depthwise_kernel_masked_outside = kernel_3x3 + kernel_5x5 * d5x5

        kernel_50c = depthwise_kernel_masked_outside * self.mask50c.to(device)
        kernel_100c = depthwise_kernel_masked_outside * self.mask100c.to(device)
        norm50c = torch.norm(kernel_50c)
        norm100c = torch.norm(kernel_100c)

        x100c = norm100c - self.t100c
        d100c = my_dropout(Indicator(x100c), dropout_rate)

        if self.stride == 1 and self.use_res_connect:
            x50c = norm50c - self.t50c
            d50c = my_dropout(Indicator(x50c), dropout_rate)
        else:
            d50c = self.const_one.to(device)
        depthwise_kernel_masked = d50c * (kernel_50c + d100c * kernel_100c)

        if not self.runtimes:
            # flops = x.shape[1] ** 2 * 5 ** 2 * self.exp / 10 ** 6 # x is the input feature matp size
            # flops_336 = x.shape[1] ** 2 * 3 ** 2 * self.exp / 10 ** 6
            flops = 5 ** 2 * self.exp / 200  # 556
            flops_336 = 3 ** 2 * self.exp / 200  # 336
            self.runtimes = [0, flops_336, flops / 2, flops, 0.7]  # [0, 336, 553, 556]
            self.R50c = self.runtimes[2]
            self.R5x5 = self.R100c = self.runtimes[3]
            self.R3x3 = self.runtimes[1]
        ratio = self.R3x3 / self.R5x5
        runtime_channels = d50c * (self.R50c + d100c * (self.R100c - self.R50c))
        runtime_reg = runtime_channels * ratio + runtime_channels * (1 - ratio) * d5x5
        conf.conv2dmask_runtime_reg.append( runtime_reg)

        # if self.writer and self.step_internal % self.interval == 0 and self.depthwise_kernel.device.index == 0:
        #     self.writer.add_scalar(f'd5x5/{self.name}', d5x5.item(), global_step=self.step_internal)
        #     self.writer.add_scalar(f'd50c/{self.name}', d50c.item(), global_step=self.step_internal)
        #     self.writer.add_scalar(f'd100c/{self.name}', d100c.item(), global_step=self.step_internal)
        #     self.writer.add_scalar(f't100c/{self.name}', self.t100c.item(), global_step=self.step_internal)
        #     self.writer.add_scalar(f't50c/{self.name}', self.t50c.item(), global_step=self.step_internal)
        #     self.writer.add_scalar(f'rtreg/{self.name}', runtime_reg.item(), global_step=self.step_internal)

        return depthwise_kernel_masked

    def forward(self, x):
        depthwise_kernel_masked = self.build_kernel()
        output = F.conv2d(
            input=x, weight=depthwise_kernel_masked, stride=self.stride,
            padding=self.padding, groups=self.exp,
        )
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
            SuperKernel(exp, kernel, stride, padding, use_res_connect=self.use_res_connect),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        xx = self.conv(x)
        if self.use_res_connect:
            return x + xx
        else:
            return xx


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


class SinglePath(nn.Module):
    def __init__(self, width_mult=1.0, use_superkernel=True):
        super(SinglePath, self).__init__()
        input_channel = 64
        last_channel = 512
        blocks = [1, 4, 8, 2]
        mobile_setting = [
            # k, exp, c,  se,     nl,  s,
            *([[5, 64, 64, False, 'PRE', 1]] * blocks[0]),
            [5, 128, 64, False, 'PRE', 2],
            *([[3, 128, 64, False, 'RE', 1]] * blocks[1]),
            [5, 256, 128, True, 'RE', 2],
            *([[5, 128, 128, True, 'RE', 1]] * blocks[2]),
            [5, 512, 128, True, 'RE', 2],
            *([[5, 256, 128, True, 'RE', 1]] * blocks[3]),
        ]
        # building first layer
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=lambda inplace: nn.PReLU()  # Hswish
                                 )]

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            k = 5
            se = True
            nl = 'PRE'
            if exp / c < 5.9:
                exp = exp * 2
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            if use_superkernel and k == 5:
                self.features.append(MobileBottleneck5x5(input_channel, output_channel, k, s, exp_channel, se, nl))
            else:
                self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        last_conv = make_divisible(512 * width_mult)  # todo or just last_channel/512
        self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=lambda inplace: nn.PReLU()  # Hswish
                                         ))
        self.pool = Linear_block(last_conv, last_conv, groups=last_conv, kernel=(7, 7), stride=(1, 1),
                                 padding=(0, 0))
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


def singlepath(pretrained=False, **kwargs):
    model = SinglePath(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    init_dev(0)
    net = singlepath(
        use_superkernel=True,
        # width_mult=1.285,
        width_mult=1,
    )
    # state_dict = torch.load('mobilenetv3_small_67.218.pth.tar')
    # net.load_state_dict(state_dict)
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net = net.cuda()
    classifier = nn.Linear(512, 10).cuda()
    opt = torch.optim.SGD(list(net.parameters()) + list(classifier.parameters()), lr=1e-1)
    net.train()

    bs = 16
    input_size = (bs, 3, 112, 112)
    target = to_torch(np.random.randint(low=0, high=10, size=(bs,)), ).cuda()
    x = torch.rand(input_size).cuda()
    grad_min, grad_max = 0, 0
    for i in range(99):
        print(' forward ----- ', i)
        opt.zero_grad()
        out = net(x)
        logits = classifier(out)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), 5)
        for name, p in net.named_parameters():
            if p.grad is not None:
                if p.grad.min().item() < grad_min:
                    grad_min = p.grad.min().item()
                if p.grad.max().item() > grad_max:
                    grad_max = p.grad.max().item()
                if p.grad.min().item() < -5 or p.grad.max().item() > 5:
                    print(name, p.grad.min().item(), p.grad.max().item(), )
                    p.grad.data.clamp_(min=-5, max=5)
        opt.step()
        print(' now loss: ', loss.item())
    print('grad stat', grad_min, grad_max)

    # from thop import profile
    #
    # model = net.to('cuda:0')
    # flops, params = profile(model, input_size=(2, 3, 112, 112),
    #                         device='cuda:0',
    #                         )
    # flops /= 10 ** 9
    # params /= 10 ** 6
    # print(flops, params)

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
    def __init__(self, channel=None):
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
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE', skip_op=False):
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
        if not skip_op:
            self.dw_conv = conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False)
            self.dw_norm = norm_layer(exp)
            self.dw_se = SELayer(exp)
            self.dw_nlin = nlin_layer(inplace=True)
        else:
            self.dw_conv = Identity()
            self.dw_norm = Identity()
            self.dw_se = Identity()
            self.dw_nlin = Identity()
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
            return x + xx
        else:
            return xx
    
    def get_decision(self):
        return (0, 0, 0, 0, 0, 0)


def judgenan(x):
    return not not torch.isnan(x).any().item() or not not torch.isinf(x).any().item()


def Indicator(x):
    sigx = torch.sigmoid(x)
    return ((x >= 0).float() - sigx).detach() + sigx


def my_dropout(x, drop_ratio):
    if drop_ratio != 0.:
        # x = F.dropout(Indicator(x), drop_ratio) * (1 - drop_ratio)
        # x = F.dropout(Indicator(x), drop_ratio)  # note: dropout scale the value
        if np.random.rand() < drop_ratio:
            x = torch.FloatTensor([0]).to(x.device)
        else:
            x = x / (1 - drop_ratio)  # todo whether?
    return x


class SuperKernel(nn.Module):
    def __init__(self, exp, kernel, stride, padding,
                 use_res_connect=True, inp=None, oup=None):
        super(SuperKernel, self).__init__()
        self.padding = padding
        self.stride = stride
        self.exp = exp
        self.use_res_connect = use_res_connect
        self.name = f'{exp}-{randomword(3)}'
        self.t5x5 = nn.Parameter(torch.FloatTensor(1))
        self.t5x5.data.fill_(0)
        self.t50c = nn.Parameter(torch.FloatTensor(1))
        self.t50c.data.fill_(0)
        self.t100c = nn.Parameter(torch.FloatTensor(1))
        self.t100c.data.fill_(0)
        self.weight = nn.Parameter(torch.FloatTensor(exp, 1, 5, 5))  # out, in, k, k
        nn.init.kaiming_normal_(self.weight, mode='fan_out')
        
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
        self.const_zero = to_torch(np.zeros(1, 'float32')).cuda()
        if inp is not None:
            self.R5x5 = self.R100c = 5 ** 2 * exp + inp * exp + exp * oup
            self.R50c = 5 ** 2 * exp / 2 + inp * exp / 2 + exp / 2 * oup
            self.R3x3 = 3 ** 2 * exp + inp * exp + exp * oup
            self.runtimes = [0, self.R3x3, self.R50c, self.R5x5]
        else:
            self.runtimes = None

    def get_decision(self):
        # with torch.no_grad():  # Todo whether
        kernel_3x3 = self.weight * self.mask3x3
        kernel_5x5 = self.weight * self.mask5x5
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
        # return d5x5, d100c, d50c, self.t5x5, self.t100c, self.t50c
        return d5x5.item(), d100c.item(), d50c.item(), self.t5x5.item(), self.t100c.item(), self.t50c.item()

    def build_kernel(self):
        dropout_rate = conf.conv2dmask_drop_ratio
        device = self.weight.device
        kernel_3x3 = self.weight * self.mask3x3.to(device)
        kernel_5x5 = self.weight * self.mask5x5.to(device)
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

        if self.runtimes:
            ratio = self.R3x3 / self.R5x5
            runtime_channels = d50c * (self.R50c + d100c * (self.R100c - self.R50c))
            runtime_reg = runtime_channels * ratio + runtime_channels * (1 - ratio) * d5x5
            # runtime_reg = d50c * (self.R50c + d100c * (self.R100c - self.R50c)) * (ratio + (1 - ratio) * d5x5)
        else:
            runtime_reg = self.const_zero.to(device)
        # logging.info(f'{x5x5.item()} {self.t5x5.item()} {depthwise_kernel_masked.mean().item()} {runtime_reg.item()}')
        return depthwise_kernel_masked, runtime_reg

    def forward(self, x):
        depthwise_kernel_masked, runtime_reg = self.build_kernel()
        output = F.conv2d(
            input=x, weight=depthwise_kernel_masked, stride=self.stride,
            padding=self.padding, groups=self.exp,
        )
        assert not torch.isnan(output).any().item()
        return output, runtime_reg


class MobileBottleneck5x5(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck5x5, self).__init__()

        assert stride in [1, 2]
        # assert kernel in [3, 5]
        assert kernel == 5
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
        self.dw_conv = SuperKernel(exp, kernel, stride, padding, use_res_connect=self.use_res_connect,
                                   inp=inp, oup=oup
                                   )
        self.dw_norm = norm_layer(exp)
        self.dw_se = SELayer(exp)
        self.dw_nlin = nlin_layer(inplace=True)
        self.pwl_conv = conv_layer(exp, oup, 1, 1, 0, bias=False)
        self.pwl_norm = norm_layer(oup)

    def forward(self, x):
        xx = self.pw_conv(x)
        xx = self.pw_norm(xx)
        xx = self.pw_nlin(xx)
        xx, runtime_reg = self.dw_conv(xx)
        xx = self.dw_norm(xx)
        xx = self.dw_se(xx)
        xx = self.dw_nlin(xx)
        xx = self.pwl_conv(xx)
        xx = self.pwl_norm(xx)
        if self.use_res_connect:
            return x + xx, runtime_reg
        else:
            return xx, runtime_reg

    def get_decision(self):
        return self.dw_conv.get_decision()


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
    def __init__(self,
                 width_mult=conf.mbfc_wm,
                 depth_mult=conf.mbfc_dm,
                 use_superkernel=True,
                 build_from_decs=(),
                 ):
        super(SinglePath, self).__init__()
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
        for ind, (k, exp, c, se, nl, s) in enumerate(mobile_setting):
            skip_op = False
            if build_from_decs:
                dec = build_from_decs[ind]
                d5x5, d100c, d50c, t5x5, t100c, t50c = dec
                use_superkernel = False
                if d5x5:
                    k = 5
                else:
                    k = 3
                if d100c and d50c:
                    exp = exp * 2
                elif not d100c:
                    skip_op = True
            if use_superkernel:
                k = 5
                exp = exp * 2
            se = False  # todo
            nl = 'PRE'
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            if use_superkernel and k == 5:
                mb5x5 = MobileBottleneck5x5(input_channel, output_channel, k, s, exp_channel, se, nl)
            else:
                mb5x5 = MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl, skip_op)
            self.features.append(mb5x5)
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

    def forward(self, x, need_runtime_reg=False, *args, **kwargs):
        # bs, nc, nh, nw = x.shape
        ttl_runtime_reg = torch.FloatTensor([0]).to(x.device)
        x = self.head(x)
        # x = self.features(x)
        for feature_op in self.features:
            x = feature_op(x)
            if isinstance(x, tuple):
                x, runtime_reg = x
                ttl_runtime_reg += runtime_reg
        x = self.tail(x)
        x = self.pool(x)
        x = self.flatten(x)
        assert not torch.isnan(x).any().item()
        x = self.linear(x)
        assert not torch.isnan(x).any().item()
        x = self.bn(x)
        # x = F.normalize(x, dim=1)
        if need_runtime_reg:
            return x, ttl_runtime_reg
        else:
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
    
    def load_state_dict_sglpth(self, state_dict):
        my_state_dict = self.state_dict()
        for key in my_state_dict.keys():
            if 'num_batches_tracked' in key: continue
            my_shape = my_state_dict[key].shape
            you_shape = state_dict[key].shape
            if my_shape == you_shape:
                my_state_dict[key] = state_dict[key]
            else:
                if len(my_shape) == 1:
                    my_state_dict[key] = state_dict[key][:my_shape[0]]
                elif len(my_shape) == 4:
                    if my_shape[0] != you_shape[0]:
                        my_state_dict[key] = state_dict[key][:my_shape[0], ...]
                    elif my_shape[1] != you_shape[1]:
                        my_state_dict[key] = state_dict[key][:, :my_shape[1], ...]
                    else:
                        # print(key, my_shape, you_shape)
                        raise ValueError()
                else:
                    # print(key, my_shape, you_shape)
                    raise ValueError()



def singlepath(pretrained=False, **kwargs):
    model = SinglePath(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    from config import conf

    conf.conv2dmask_drop_ratio = 0
    init_dev((3,))
    net = singlepath()
    print('net:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net = nn.DataParallel(net).cuda()
    net.train()
    
    # classifier = nn.Linear(512, 10).cuda()
    # classifier.train()
    # opt = torch.optim.SGD(list(net.parameters()) + list(classifier.parameters()), lr=1e-1)
    #
    # bs = 32
    # input_size = (bs, 3, 112, 112)
    # target = to_torch(np.random.randint(low=0, high=10, size=(bs,)), ).cuda()
    # x = torch.rand(input_size).cuda()
    #
    # for i in range(99):
    #     print(' forward ----- ', i)
    #     with torch.no_grad():
    #         net.module.get_decisions()
    #     opt.zero_grad()
    #     ttl_runtime = 0.452 * 10 ** 6
    #     target_runtime = 2.5 * 10 ** 6
    #     out, runtime_ = net(x, need_runtime_reg=True)
    #     logits = classifier(out)
    #     loss = nn.CrossEntropyLoss()(logits, target)
    #     ttl_runtime += runtime_.mean()
    #     # runtime_reg_loss = 0.1 *10**3 * 10 ** 3 * torch.log(runtime_reg)
    #     if ttl_runtime > target_runtime:
    #         w_ = 1.03
    #     else:
    #         w_ = 0
    #     runtime_regloss = 10 * (ttl_runtime / target_runtime) ** w_
    #     (loss + runtime_regloss).backward()
    #     opt.step()
    #     print(' now loss: ', loss.item(), 'rt ', ttl_runtime.item(), 'rtloss ', runtime_regloss.item())
    #
    # decs = (net.module.get_decisions())
    decs = msgpack_load('/tmp/tmp.pk')
    # msgpack_dump(decs, '/tmp/tmp.pk')
    net2 = singlepath(build_from_decs=decs)
    print(decs)
    net2.load_state_dict_sglpth(net.module.state_dict(), )

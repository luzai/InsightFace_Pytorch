from __future__ import absolute_import
from __future__ import division

import lz
from lz import *
import logging
import torch
import torch.nn as nn
import os

logger = logging.getLogger(__name__)


class OFPenalty(nn.Module):
    _WARNED = False

    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.last_x = None

    def dominant_eigenvalue(self, A):
        B, N, _ = A.size()
        if self.last_x is None:
            with torch.no_grad():
                x = F.normalize(torch.randn(B, N, 1, device='cuda'), dim=1)
                # for _ in range(1):
                #     x = torch.bmm(A, x)
                #     x = F.normalize(x, dim=1)
                self.last_x = x

        x = self.last_x
        for _ in range(9):
            x = torch.bmm(A, x)
            x = F.normalize(x, dim=1)
        self.last_x = x.detach()
        # x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()
        # lz.embed()
        return numerator / denominator

    def get_singular_values(self, A):
        # AAT = torch.bmm(A, A.permute(0, 2, 1))
        AAT = ATA = torch.bmm(A.permute(0, 2, 1), A)  # todo why not same??
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def apply_penalty(self, k='final', x=None, ):
        if isinstance(x, (tuple)):
            if not len(x):
                return 0.
            return sum([self.apply_penalty(k, xx) for xx in x]) / len(x)

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        # todo ?
        # W = W / torch.max(torch.norm(W, dim=1), dim=1)[0].view(batches, 1, 1)
        smallest, largest = self.get_singular_values(W)
        singular_penalty = self.beta * (largest / smallest - 1) ** 2  # (largest - smallest)

        if k == 'intermediate':
            singular_penalty *= 0.01

        return singular_penalty.sum() / (x.size(0))  # Quirk: normalize to 1-batch case

    def forward(self, A):
        return self.apply_penalty("final", A)


of_reger = OFPenalty()


class OFPenaltyOri(nn.Module):
    _WARNED = False

    def __init__(self, args):
        super().__init__()

        self.penalty_position = frozenset(args['of_position'])
        self.beta = args['of_beta']

    def dominant_eigenvalue(self, A):

        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        # x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

        return numerator / denominator

    def get_singular_values(self, A):
        AAT = torch.bmm(A, A.permute(0, 2, 1))
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def apply_penalty(self, k, x):
        if isinstance(x, (tuple)):
            if not len(x):
                return 0.
            return sum([self.apply_penalty(k, xx) for xx in x]) / len(x)

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        singular_penalty = (largest - smallest) * self.beta

        if k == 'intermediate':
            singular_penalty *= 0.01

        return singular_penalty.sum() / (x.size(0) / 32.)  # Quirk: normalize to 32-batch case

    def forward(self, inputs):

        _, y, _, feature_dict = inputs

        logger.debug(str(self.penalty_position))

        existed_positions = frozenset(feature_dict.keys())
        missing = self.penalty_position - existed_positions
        if missing and not self._WARNED:
            self._WARNED = True

            import warnings
            warnings.warn('OF positions {!r} are missing. IGNORED.'.format(list(missing)))

        singular_penalty = sum(
            [self.apply_penalty(k, x) for k, x in feature_dict.items() if k in self.penalty_position])

        logger.debug(str(singular_penalty))
        return singular_penalty


def l2_reg_ortho(W, l2_reg=None):
    cols = W[0].numel()
    rows = W.shape[0]
    w1 = W.view(-1, cols)
    wt = torch.transpose(w1, 0, 1)
    m = torch.matmul(wt, w1)
    ident = (torch.eye(cols, cols))
    ident = ident.cuda()

    w_tmp = (m - ident)
    height = w_tmp.size(0)
    u = F.normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
    v = F.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
    u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
    sigma = torch.dot(u, torch.matmul(w_tmp, v))

    if l2_reg is None:
        l2_reg = (sigma) ** 2
    else:
        l2_reg = l2_reg + (sigma) ** 2

    return l2_reg


if __name__ == '__main__':
    init_dev(2)
    A = torch.rand(512, 49).cuda()
    A = A - 0.5
    reg = l2_reg_ortho(A)
    print(torch.sqrt(reg))

    A.requires_grad_(True)
    for i in range(9999):
        reg = l2_reg_ortho(A)
        reg.backward()
        A.data = A.data - 1e-3 * A.grad.detach()
        A.grad = None
        if i % 99 == 1:
            print(i, torch.sqrt(reg))
    print(A)
    # Ares = A.cpu().detach()
    # Ares = torch.rand(512, 49) - 0.5
    # from scipy.spatial.distance import cdist
    # dist = cdist(Ares, Ares, 'cosine')
    # cos = 1 - dist
    # cos = cos[~np.eye(cos.shape[0],dtype=bool)]
    # stat(cos)
    # np.linalg.norm(Ares, axis=1)
    # np.linalg.svd(Ares, compute_uv=False)

    # torch.manual_seed(1)
    # A = torch.rand(1, 512, 7, 7).cuda()
    # A = torch.ones(1, 512, 7, 7).cuda()
    # A = (A - 0.5)

    # A.requires_grad_(True)
    # for i in range(999):
    #     reg = of_reger.forward(A)
    #     reg.backward()
    #     A.data = A.data - 1e-3 * A.grad.detach()
    #     A.grad = None
    #     if i % 99 == 1:
    #         print(i, reg.item())
    #
    # A = A.view(-1, 512, 49)
    # print(torch.sort(of_reger.get_singular_values(A)[0])[0])
    # print(torch.sort(of_reger.get_singular_values(A)[1])[0])

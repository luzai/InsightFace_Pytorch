# -*- coding: future_fstrings -*-

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
    
    def __repr__(self):
        return f'VATLoss(xi={self.xi}, eps={self.eps}, )'
    
    def forward(self, model, head, x):
        with torch.no_grad():
            fea = model(x)
            logit = head(fea)
            pred = F.softmax(logit, dim=1)
        
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                fea_hat = model(x + self.xi * d)
                pred_hat = head(fea_hat)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                # d = torch.autograd.grad(outputs=adv_distance, inputs=d,
                #                         create_graph=False, retain_graph=False, only_inputs=True)[0].detach()
                # d = _l2_normalize(d)
                adv_distance.backward()
                d = _l2_normalize(d.grad.detach())
                model.zero_grad()
            
            # calc LDS
            if torch.isnan(d).any().item():
                # raise ValueError(f'nan {d}')
                logging.error(f'nan {d}')
                return torch.zeros(1).cuda()
            r_adv = d * self.eps
            fea_hat = model(x + r_adv)
            pred_hat = head(fea_hat)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')
        
        return lds

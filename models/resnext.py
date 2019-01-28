import sys
from collections import OrderedDict
from functools import partial

import torch.nn as nn
import torch
from modules import IdentityResidualBlock, GlobalAvgPool2d, InPlaceABN
from models._util import try_index
from models.model import Linear_block, Flatten, l2_norm


class ResNeXt(nn.Module):
    def __init__(self,
                 structure,
                 groups=64,
                 norm_act=InPlaceABN,
                 input_3x3=True,
                 classes=0,
                 dilation=1,
                 base_channels=(128, 128, 256)):
        """Pre-activation (identity mapping) ResNeXt model

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the four modules of the network.
        groups : int
            Number of groups in each ResNeXt block
        norm_act : callable
            Function to create normalization / activation Module.
        input_3x3 : bool
            If `True` use three `3x3` convolutions in the input module instead of a single `7x7` one.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : list of list of int or list of int or int
            List of dilation factors, or `1` to ignore dilation. For each module, if a single value is given it is
            used for all its blocks, otherwise this expects a value for each block.
        base_channels : list of int
            Channels in the blocks of the first residual module. Each following module will multiply these values by 2.
        """
        super(ResNeXt, self).__init__()
        self.structure = structure
        
        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if dilation != 1 and len(dilation) != 4:
            raise ValueError("If dilation is not 1 it must contain four values")
        
        # Initial layers
        if input_3x3:
            layers = [
                ("conv1", nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ("bn1", norm_act(64)),
                ("conv2", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ("bn2", norm_act(64)),
                ("conv3", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ("pool", nn.MaxPool2d(3, stride=2, padding=1))
            ]
        else:
            layers = [
                ("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
                ("pool", nn.MaxPool2d(3, stride=2, padding=1))
            ]
        self.mod1 = nn.Sequential(OrderedDict(layers))
        
        # Groups of residual blocks
        in_channels = 64
        channels = base_channels
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                s, d = self._stride_dilation(mod_id, block_id, dilation)
                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels, channels, stride=s, norm_act=norm_act, groups=groups, dilation=d)
                ))
                
                # Update channels
                in_channels = channels[-1]
            
            # Create and add module
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            channels = [c * 2 for c in channels]
        
        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        
        self.output_layer = nn.Sequential(
            Linear_block(in_channels, in_channels, groups=in_channels, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            Flatten(),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
        )
        
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))
    
    def forward(self, img, normalize=True, return_norm=False,   mode='train'  ):
        if img.shape[-1] == 112:
            with torch.no_grad():
                img = nn.functional.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
        if mode == 'finetune':
            with torch.no_grad():
                out = self.mod1(img)
                out = self.mod2(out)
                out = self.mod3(out)
                out = self.mod4(out)
                out = self.mod5(out)
        else:
            out = self.mod1(img)
            out = self.mod2(out)
            out = self.mod3(out)
            out = self.mod4(out)
            out = self.mod5(out)
        out = self.bn_out(out)
        out = self.output_layer(out)
        if hasattr(self, "classifier"):
            out = self.classifier(out)
        x = out
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
    
    @staticmethod
    def _stride_dilation(mod_id, block_id, dilation):
        if dilation == 1:
            s = 2 if mod_id > 0 and block_id == 0 else 1
            d = 1
        else:
            if dilation[mod_id] == 1:
                s = 2 if mod_id > 0 and block_id == 0 else 1
                d = 1
            else:
                s = 1
                d = try_index(dilation[mod_id], block_id)
        return s, d


_NETS = {
    "50": {"structure": [3, 4, 6, 3]},
    "101": {"structure": [3, 4, 23, 3]},
    "152": {"structure": [3, 8, 36, 3]},
}

_NETS["100"] = _NETS["101"]

__all__ = ["ResNeXt"]
for name, params in _NETS.items():
    net_name = "net_resnext" + name
    setattr(sys.modules[__name__], net_name, partial(ResNeXt, **params))
    __all__.append(net_name)

if __name__ == '__main__':
    print(__all__)

from __future__ import print_function, division, absolute_import
from models.fbresnet import fbresnet152

from models.cafferesnet import cafferesnet101

from models.bninception import bninception

from models.resnext import resnext101_32x4d
from models.resnext import resnext101_64x4d

from models.inceptionv4 import inceptionv4

from models.inceptionresnetv2 import inceptionresnetv2

from models.nasnet import nasnetalarge

from models.nasnet_mobile import nasnetamobile

from models.torchvision_models import alexnet
from models.torchvision_models import densenet121
from models.torchvision_models import densenet169
from models.torchvision_models import densenet201
from models.torchvision_models import densenet161
from models.torchvision_models import resnet18
from models.torchvision_models import resnet34
from models.torchvision_models import resnet50
from models.torchvision_models import resnet101
from models.torchvision_models import resnet152
from models.torchvision_models import inceptionv3
from models.torchvision_models import squeezenet1_0
from models.torchvision_models import squeezenet1_1
from models.torchvision_models import vgg11
from models.torchvision_models import vgg11_bn
from models.torchvision_models import vgg13
from models.torchvision_models import vgg13_bn
from models.torchvision_models import vgg16
from models.torchvision_models import vgg16_bn
from models.torchvision_models import vgg19_bn
from models.torchvision_models import vgg19

from models.dpn import dpn68
from models.dpn import dpn68b
from models.dpn import dpn92
from models.dpn import dpn98
from models.dpn import dpn131
from models.dpn import dpn107

from models.xception import xception

from models.senet import senet154
from models.senet import se_resnet50
from models.senet import se_resnet101
from models.senet import se_resnet152
from models.senet import se_resnext50_32x4d
from models.senet import se_resnext101_32x4d

from models.pnasnet import pnasnet5large
from models.polynet import polynet

if __name__ == '__main__':
    for model in [
        # resnet50(pretrained=None),  # 1.4k -->1.5k
        nasnetamobile(pretrained=None), # 1.5k --> 1.6k
        # densenet161(pretrained=None), # 2.6k
        # senet154(pretrained=None), # 4.4k   # more than 12k
        # se_resnet101(pretrained=None), # 2.2k
        # se_resnet152(pretrained=None), # 2.9k
        # se_resnext101_32x4d(pretrained=None),# 2.6k
        # polynet(pretrained=None),
        # inceptionresnetv2(pretrained=None),
    ]:
        model = model.cuda()
        print('next')
        import torch
        
        input = torch.autograd.Variable(torch.randn(8, 3, 112, 112)).cuda()
        # input = Variable(torch.randn(2, 3, 224, 224))
        output = model(input)
        output.mean().backward()
        param_mb = sum(p.numel() for p in model.parameters()) / 1000000.0
        print(param_mb)
        import time
        
        time.sleep(10)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.insert(0, '/home/xinglu/prj/InsightFace_Pytorch')
import torch
from torchvision import transforms as trans
import lz
from lz import *
import os
from easydict import EasyDict as edict
import time
import sys
import numpy as np
import argparse
import struct
import cv2, cvbase as cvb
import sklearn
from sklearn.preprocessing import normalize
import mxnet as mx
from mxnet import ndarray as nd

use_mxnet = False
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def read_img(image_path):
    img = cvb.read_img(image_path, )
    return img


def get_feature(imgs, nets):
    count = len(imgs)
    if not use_mxnet:
        data = torch.zeros((count * 2, 3, imgs[0].shape[0], imgs[0].shape[1]))
        for idx, img in enumerate(imgs):
            img = img[:, :, ::-1].copy()  # to rgb
            img = test_transform(img).numpy()
            for flipid in [0, 1]:
                _img = np.copy(img)
                if flipid == 1:
                    _img = _img[:, :, ::-1].copy()
                _img = lz.to_torch(_img)
                data[count * flipid + idx] = _img
        F = []
        for net in nets:
            with torch.no_grad():
                x = net.model(data).cpu().numpy()
                embedding = x[0:count, :] + x[count:, :]
                embedding = sklearn.preprocessing.normalize(embedding)
                F.append(embedding)
        F = np.concatenate(F, axis=1)
        F = sklearn.preprocessing.normalize(F)
    else:
        data = mx.nd.zeros(shape=(count * 2, 3, imgs[0].shape[0], imgs[0].shape[1]))
        for idx, img in enumerate(imgs):
            img = img[:, :, ::-1]  # to rgb
            img = np.transpose(img, (2, 0, 1))
            for flipid in [0, 1]:
                _img = np.copy(img)
                if flipid == 1:
                    _img = _img[:, :, ::-1]
                _img = nd.array(_img)
                data[count * flipid + idx] = _img
        
        F = []
        for net in nets:
            db = mx.io.DataBatch(data=(data,))
            net.model.forward(db, is_train=False)
            x = net.model.get_outputs()[0].asnumpy()
            embedding = x[0:count, :] + x[count:, :]
            embedding = sklearn.preprocessing.normalize(embedding)
            # print('emb', embedding.shape)
            F.append(embedding)
        F = np.concatenate(F, axis=1)
        F = sklearn.preprocessing.normalize(F)
        # print('F', F.shape)
    return F


def write_bin(path, feature):
    feature = np.asarray(feature, dtype=np.float32)
    assert np.isclose((feature ** 2).sum(), 1)
    assert not np.isnan(feature).any()
    # feature += np.random.uniform(-0.001, 0.001, (512,))
    feature = list(feature)
    assert len(feature) == 512
    # rm(path)
    # load_mat(path)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))


def get_and_write(buffer, nets):
    imgs = []
    for k in buffer:
        imgs.append(k[0])
    features = get_feature(imgs, nets)
    # print(np.linalg.norm(feature))
    assert features.shape[0] == len(buffer)
    for ik, k in enumerate(buffer):
        out_path = k[1]
        feature = features[ik].flatten()
        assert feature.shape[0] == 512
        write_bin(out_path, feature)


def main(args):
    print(args)
    gpuid = args.gpu
    ctx = mx.gpu(gpuid)
    nets = []
    image_shape = [int(x) for x in args.image_size.split(',')]
    for model in args.model.split('|'):
        if use_mxnet:
            vec = model.split(',')
            assert len(vec) > 1
            prefix = vec[0]
            epoch = int(vec[1])
            print('loading', prefix, epoch)
            net = edict()
            net.ctx = ctx
            net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
            all_layers = net.sym.get_internals()
            net.sym = all_layers['fc1_output']
            net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
            net.model.bind(data_shapes=[('data', (1, 3, image_shape[1], image_shape[2]))])
            net.model.set_params(net.arg_params, net.aux_params)
            nets.append(net)
        else:
            
            from config import conf
            conf.need_log = False
            conf.batch_size *= 2
           
            from Learner import l2_norm, FaceInfer
            learner = FaceInfer(conf, )
            learner.load_state(
                resume_path=args.model,
                latest=False,
            )
            learner.model.eval()
            nets.append(learner)
    facescrub_out = os.path.join(args.output, 'facescrub')
    megaface_out = os.path.join(args.output, 'megaface')
    
    i = 0
    succ = 0
    buffer = []
    
    imgfns = []
    imgfns += json_load('/data/share/megaface/devkit/templatelists/facescrub_features_list_10000.2.json')['path']
    imgfns += open(args.facescrub_lst, 'r').readlines()
    imgfns = ['/'.join(imgfn.split('/')[-2:]).strip() for imgfn in imgfns]
    imgfns = np.unique(imgfns).tolist()
    # if True:
    #     imgfns = []
    for line in imgfns:
        if i % 1000 == 0:
            print("writing fs", i, succ)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a, b = _path[-2], _path[-1]
        out_dir = os.path.join(facescrub_out, a)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        image_path = os.path.join(args.facescrub_root, image_path)
        img = read_img(image_path)
        assert img is not None, image_path
        assert img.shape == (112, 112, 3), image_path
        # if img is None:
        #     print('read error:', image_path)
        #     continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, nets)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, nets)
        buffer = []
    print('fs stat', i, succ)
    
    # if True:
    #     return
    i = 0
    succ = 0
    buffer = []
    imgfns = []
    for mega_lst in ['/data/share/megaface/devkit/templatelists/megaface_features_list.json_1000000_1',
                     '/data/share/megaface/devkit/templatelists/megaface_features_list.json_100000_1',
                     '/data/share/megaface/devkit/templatelists/megaface_features_list.json_100_1'
                     ]:
        imgfns += json_load(mega_lst)['path']
    imgfns = np.unique(imgfns).tolist()

    for line in imgfns:
        # for line in open(args.megaface_lst, 'r'):
        if i % 1000 == 0:
            print("writing mf", i, succ)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        out_dir = os.path.join(megaface_out, a1, a2)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            # continue
        # print(landmark)
        image_path = os.path.join(args.megaface_root, image_path)
        img = read_img(image_path)
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, nets)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, nets)
        buffer = []
    print('mf stat', i, succ)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    mega_path = '/data2/share/megaface'
    parser.add_argument('--batch_size', type=int, help='', default=32)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--algo', type=str, help='', default='zju.artificial.idiot')
    parser.add_argument('--facescrub-lst', type=str, help='', default=f'{mega_path}/facescrub_lst')
    parser.add_argument('--megaface-lst', type=str, help='', default='/data2/share/megaface/megaface_lst')
    parser.add_argument('--facescrub-root', type=str, help='', default='/data2/share/megaface/facescrub_images.2')
    parser.add_argument('--megaface-root', type=str, help='', default='/data2/share/megaface/megaface_images')
    parser.add_argument('--output', type=str, help='', default='')
    parser.add_argument('--model', type=str, help='',
                        default='/home/xinglu/prj/insightface/logs/MS1MV2-ResNet100-Arcface/model,0')
    parser.set_defaults(
        batch_size=32 if use_mxnet else 128,
        facescrub_root='/data2/share/megaface/facescrub_images.2',
        algo='zju.artificial.idiot',
        output='./feature_out.r152.ada.chkpnt.3',
        # output='./feature_out',
        model=lz.root_path + 'work_space/emore.r152.ada.chkpnt.3/save/',
        # model=lz.root_path + '../insightface/logs/MS1MV2-ResNet100-Arcface/model,0',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    import lz
    # lz.init_dev(lz.get_dev())
    main(parse_arguments(sys.argv[1:]))

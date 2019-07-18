from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lz
from lz import *
import os
from datetime import datetime
import os.path
from easydict import EasyDict as edict
import time
import json
import sys
import numpy as np
import importlib
import itertools
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import mxnet as mx
from mxnet import ndarray as nd
import lmdb, six
from PIL import Image

lz.init_dev((0, 1, 2, 3))
image_shape = (3, 112, 112)
net = None
data_size = 1862120
emb_size = 0
use_flip = True
use_mxnet = False
env = None
glargs = None
xrange = range


def do_flip(data):
    for idx in xrange(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_feature(buffer):
    global emb_size
    if use_flip:
        input_blob = np.zeros((len(buffer) * 2, 3, image_shape[1], image_shape[2]))
    else:
        input_blob = np.zeros((len(buffer), 3, image_shape[1], image_shape[2]))
    idx = 0
    for item in buffer:
        if env is None:
            img = cv2.imread(item)[:, :, ::-1]  # to rgb
        else:
            item = item.replace(glargs.input, '')
            with env.begin(write=False) as txn:
                imgbuf = txn.get(str(item).encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            f = Image.open(buf)
            img = f.convert('RGB')
            img = np.asarray(img)
        img = np.transpose(img, (2, 0, 1))
        attempts = [0, 1] if use_flip else [0]
        for flipid in attempts:
            _img = np.copy(img)
            if flipid == 1:
                do_flip(_img)
            input_blob[idx] = _img
            idx += 1
    input_blob = input_blob.astype('float32')
    if use_mxnet:
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        net.model.forward(db, is_train=False)
        _embedding = net.model.get_outputs()[0].asnumpy()
    else:
        data = input_blob - 127.5
        data /= 127.5
        data = to_torch(data)
        with torch.no_grad():
            _embedding = net.model(data).cpu().numpy()
    if emb_size == 0:
        emb_size = _embedding.shape[1]
        print('set emb_size to ', emb_size)
    if use_flip:
        embedding1 = _embedding[0::2]
        embedding2 = _embedding[1::2]
        embedding = embedding1 + embedding2
        embedding /= 2
    else:
        embedding = _embedding
    embedding = sklearn.preprocessing.normalize(embedding)  # todo
    # print('norm ', np.linalg.norm(embedding, axis=1))
    return embedding


def write_bin(path, m):
    rows, cols = m.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', rows, cols, cols * 4, 5))
        f.write(m.data)


def main(args):
    global image_shape, net, env, glargs
    print(args)
    glargs = args
    env = lmdb.open(args.input + '/imgs_lmdb', readonly=True,
                    # max_readers=1,  lock=False,
                    # readahead=False, meminit=False
                    )
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd) > 0:
        for i in xrange(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))
    image_shape = [int(x) for x in args.image_size.split(',')]
    if use_mxnet:
        net = edict()
        vec = args.model.split(',')
        assert len(vec) > 1
        prefix = vec[0]
        epoch = int(vec[1])
        print('loading', prefix, epoch)
        net.ctx = ctx
        net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
        # net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
        all_layers = net.sym.get_internals()
        net.sym = all_layers['fc1_output']
        net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
        net.model.bind(data_shapes=[('data', (args.batch_size, 3, image_shape[1], image_shape[2]))])
        net.model.set_params(net.arg_params, net.aux_params)
    else:
        # sys.path.insert(0, lz.home_path + 'prj/InsightFace_Pytorch/')
        from config import conf
        from Learner import FaceInfer

        conf.need_log = False
        conf.batch_size = args.batch_size
        conf.fp16 = True
        conf.ipabn = False
        conf.cvt_ipabn = False
        conf.use_chkpnt = False

        net = FaceInfer(conf, gpuid=range(conf.num_devs))
        net.load_state(
            resume_path=args.model,
            latest=False,
        )
        net.model.eval()

    features_all = None

    i = 0
    fstart = 0
    buffer = []
    for line in open(os.path.join(args.input, 'filelist.txt'), 'r'):
        if i % 1000 == 0:
            print("processing ", i, data_size, 1. * i / data_size)
        i += 1
        line = line.strip()
        image_path = os.path.join(args.input, line)
        buffer.append(image_path)
        if len(buffer) == args.batch_size:
            embedding = get_feature(buffer)
            buffer = []
            fend = fstart + embedding.shape[0]
            if features_all is None:
                features_all = np.zeros((data_size, emb_size), dtype=np.float32)
            # print('writing', fstart, fend)
            features_all[fstart:fend, :] = embedding
            fstart = fend
    if len(buffer) > 0:
        embedding = get_feature(buffer)
        fend = fstart + embedding.shape[0]
        print('writing', fstart, fend)
        features_all[fstart:fend, :] = embedding
    write_bin(args.output, features_all)
    # os.system("bypy upload %s"%args.output)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=160 * 4)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--input', type=str, help='', default='')
    parser.add_argument('--output', type=str, help='', default='')
    parser.add_argument('--model', type=str, help='', default='')
    parser.set_defaults(
        input='/data/share/iccv19.lwface/iccv19-challenge-data/',
        # output=lz.work_path + 'mbfc.retina.cl.arc.cotch.bin',
        output=lz.work_path + 'mbfc.cotch.mual.1e-3.bin',
        # model=lz.root_path + '../insightface/logs/r50-arcface-retina/model,16',
        # model=lz.root_path + 'work_space/mbfc.retina.cl.arc.cotch.cont/models',
        model=lz.root_path + 'work_space/mbfc.cotch.mual.1e-3.cont/models',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

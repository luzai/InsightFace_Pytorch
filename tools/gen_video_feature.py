from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lz import *
import os
from datetime import datetime
import os.path
from easydict import EasyDict as edict
import time
import json
import glob
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
import lz
import lmdb, six
from PIL import Image
from config import conf

use_devs = (0, 1, 2, 3,)
lz.init_dev(use_devs)
image_shape = (3, 112, 112)
net = None
data_size = 203848
emb_size = conf.embedding_size
use_flip = True
ctx_num = 1
xrange = range
env = None
glargs = None
use_mxnet = False


def do_flip(data):
    for idx in xrange(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_feature(buffer):
    global emb_size
    input_count = len(buffer)
    if use_flip:
        input_count *= 2
    network_count = input_count
    if input_count % ctx_num != 0:
        network_count = (input_count // ctx_num + 1) * ctx_num

    input_blob = np.zeros((network_count, 3, image_shape[1], image_shape[2]), dtype=np.float32)
    idx = 0
    for item in buffer:
        if env is None:
            img = cv2.imread(item)[:, :, ::-1]  # to rgb
        else:
            item = '/' + item.replace(glargs.input, '').strip('/')
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
    if use_mxnet:
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        net.model.forward(db, is_train=False)
        _embedding = net.model.get_outputs()[0].asnumpy()
        # _embedding = _embedding[0:input_count]
    else:
        data = input_blob - 127.5
        data /= 127.5
        data = to_torch(data)
        with torch.no_grad():
            _embedding = net.model(data).cpu().numpy()
    _embedding = _embedding[0:input_count]
    if emb_size == 0:
        emb_size = _embedding.shape[1]
        print('set emb_size to ', emb_size)
    embedding = np.zeros((len(buffer), emb_size), dtype=np.float32)
    if use_flip:
        embedding1 = _embedding[0::2]
        embedding2 = _embedding[1::2]
        embedding = (embedding1 + embedding2) / 2
    else:
        embedding = _embedding
    embedding = sklearn.preprocessing.normalize(embedding)  # todo
    return embedding


def write_bin(path, m):
    rows, cols = m.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', rows, cols, cols * 4, 5))
        f.write(m.data)


def main_allimg(args):
    global image_shape
    global net
    global ctx_num, env, glargs
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
    ctx_num = len(ctx)
    image_shape = [int(x) for x in args.image_size.split(',')]
    if use_mxnet:
        vec = args.model.split(',')
        assert len(vec) > 1
        prefix = vec[0]
        epoch = int(vec[1])
        print('loading', prefix, epoch)
        net = edict()
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
        lz.init_dev(use_devs)
        conf.need_log = False
        conf.fp16 = True  # maybe faster ?
        conf.ipabn = False
        conf.cvt_ipabn = False
        conf.use_chkpnt = False
        conf.net_mode = 'ir_se'
        conf.net_depth = 100
        conf.input_size = 128
        conf.embedding_size = 512
        from Learner import FaceInfer

        net = FaceInfer(conf,
                        gpuid=range(len(use_devs)),
                        )
        net.load_state(
            resume_path=args.model,
            latest=True,
        )
        net.model.eval()
    features_all = None

    filelist = os.path.join(args.input, 'filelist.txt')
    lines = open(filelist, 'r').readlines()
    buffer_images = []
    buffer_embedding = np.zeros((0, emb_size), dtype=np.float16)
    row_idx = 0
    import h5py
    f = h5py.File(args.output, 'w')
    chunksize = 80 * 10 ** 3
    dst = f.create_dataset("feas", (chunksize, 512), maxshape=(None, emb_size), dtype='f2')
    ind_dst = 0
    vdonm2imgs = lz.msgpack_load(args.input + '/../vdonm2imgs.pk')
    for line in lines:
        if row_idx % 1000 == 0:
            logging.info(f"processing {(row_idx, len(lines), row_idx / len(lines),)}")
        row_idx += 1
        # if row_idx<203000:continue
        # print('stat', i, len(buffer_images), buffer_embedding.shape, aggr_nums, row_idx)
        videoname = line.strip().split()[0]
        # images2 = glob.glob("%s/%s/*.jpg" % (args.input, videoname))
        # images2 = np.sort(images2).tolist()
        images = vdonm2imgs[videoname]
        assert len(images) > 0
        for image_path in images:
            buffer_images.append(image_path)
        while len(buffer_images) >= args.batch_size:
            embedding = get_feature(buffer_images[0:args.batch_size])
            buffer_images = buffer_images[args.batch_size:]
            if ind_dst + args.batch_size > dst.shape[0]:
                dst.resize((dst.shape[0] + chunksize, emb_size), )
            dst[ind_dst:ind_dst + args.batch_size, :] = embedding.astype('float16')
            ind_dst += args.batch_size
            # buffer_embedding = np.concatenate((buffer_embedding, embedding), axis=0).astype('float16')
    if len(buffer_images) != 0:
        embedding = get_feature(buffer_images)
        if ind_dst + args.batch_size > dst.shape[0]:
            dst.resize((dst.shape[0] + chunksize, emb_size), )
        dst[ind_dst:ind_dst + args.batch_size, :] = embedding.astype('float16')
    # lz.save_mat(args.output, buffer_embedding)
    f.flush()
    f.close()


def main(args):
    global image_shape
    global net
    global ctx_num, env, glargs
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
    ctx_num = len(ctx)
    image_shape = [int(x) for x in args.image_size.split(',')]
    if use_mxnet:
        vec = args.model.split(',')
        assert len(vec) > 1
        prefix = vec[0]
        epoch = int(vec[1])
        print('loading', prefix, epoch)
        net = edict()
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
        conf.need_log = False
        conf.fp16 = True  # maybe faster ?
        conf.ipabn = False
        conf.cvt_ipabn = False
        conf.use_chkpnt = False
        # conf.net_mode = 'ir_se'
        # conf.net_depth = 100
        from Learner import FaceInfer

        net = FaceInfer(conf, gpuid=range(conf.num_devs))
        net.load_state(
            resume_path=args.model,
            latest=False,
        )
        net.model.eval()
    features_all = None

    i = 0
    filelist = os.path.join(args.input, 'filelist.txt')
    lines = open(filelist, 'r').readlines()
    buffer_images = []
    buffer_embedding = np.zeros((0, 0), dtype=np.float32)
    aggr_nums = []
    row_idx = 0
    for line in lines:
        # if i < 203000:
        #     i += 1
        #     continue

        if i % 1000 == 0:
            print("processing ", i, len(lines), i / len(lines), )
        i += 1
        # print('stat', i, len(buffer_images), buffer_embedding.shape, aggr_nums, row_idx)
        videoname = line.strip().split()[0]
        images = glob.glob("%s/%s/*.jpg" % (args.input, videoname))
        # images = np.sort(images).tolist()
        assert len(images) > 0
        image_features = []
        for image_path in images:
            buffer_images.append(image_path)
        aggr_nums.append(len(images))
        while len(buffer_images) >= args.batch_size:
            embedding = get_feature(buffer_images[0:args.batch_size])
            buffer_images = buffer_images[args.batch_size:]
            if buffer_embedding.shape[1] == 0:
                buffer_embedding = embedding.copy().astype('float32')
            else:
                buffer_embedding = np.concatenate((buffer_embedding, embedding), axis=0)
        buffer_idx = 0
        acount = 0
        for anum in aggr_nums:
            if buffer_embedding.shape[0] >= anum + buffer_idx:
                image_features = buffer_embedding[buffer_idx:buffer_idx + anum]
                video_feature = np.sum(image_features, axis=0, keepdims=True)
                video_feature = sklearn.preprocessing.normalize(video_feature)
                if features_all is None:
                    features_all = np.zeros((data_size, video_feature.shape[1]), dtype=np.float32)
                # print('write to', row_idx, anum, buffer_embedding.shape)
                features_all[row_idx] = video_feature.flatten()
                row_idx += 1
                buffer_idx += anum
                acount += 1
            else:
                break
        aggr_nums = aggr_nums[acount:]
        buffer_embedding = buffer_embedding[buffer_idx:]

    if len(buffer_images) > 0:
        embedding = get_feature(buffer_images)
        buffer_images = buffer_images[args.batch_size:]
        buffer_embedding = np.concatenate((buffer_embedding, embedding), axis=0)
    buffer_idx = 0
    acount = 0
    for anum in aggr_nums:
        assert buffer_embedding.shape[0] >= anum + buffer_idx
        image_features = buffer_embedding[buffer_idx:buffer_idx + anum]
        video_feature = np.sum(image_features, axis=0, keepdims=True)
        video_feature = sklearn.preprocessing.normalize(video_feature)
        # print('last write to', row_idx, anum, buffer_embedding.shape)
        features_all[row_idx] = video_feature.flatten()
        row_idx += 1
        buffer_idx += anum
        acount += 1

    aggr_nums = aggr_nums[acount:]
    buffer_embedding = buffer_embedding[buffer_idx:]
    # embed()
    assert len(aggr_nums) == 0
    assert buffer_embedding.shape[0] == 0

    write_bin(args.output, features_all)
    print(row_idx, features_all.shape)
    # os.system("bypy upload %s"%args.output)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=175 * len(use_devs))
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--input', type=str, help='', default='')
    parser.add_argument('--output', type=str, help='', default='')
    parser.add_argument('--model', type=str, help='', default='')
    parser.set_defaults(
        input='/data/share/iccv19.lwface/iQIYI-VID-FACE',
        # output=lz.work_path + 'r100.2nrm.bin',
        # output=lz.work_path + 'r100.unrm.allimg.h5',
        output=lz.work_path + 'vdo.mbfc.cotch.mual.1e-3.2nrm.bin',
        # output=lz.work_path + 'mbfc.unrm.allimg.h5',
        # model=lz.root_path + '../insightface/logs/r50-arcface-retina/model,16',
        # model=lz.root_path + './work_space/r100.128.retina.clean.arc/models',
        # model=lz.root_path + './work_space/mbfc.lrg.retina.clean.arc/models',
        model=lz.root_path + 'work_space/mbfc.cotch.mual.1e-3.cont/models',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    # lz.get_dev(1,ok=(3,))
    main(parse_arguments(sys.argv[1:]))
    # main_allimg(parse_arguments(sys.argv[1:]))

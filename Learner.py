import lz
from data.data_pipe import de_preprocess, get_train_loader, get_val_data, get_val_pair
from model import *
from verifacation import evaluate
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras, hflip
from PIL import Image
from torchvision import transforms as trans
import os
import random
import logging
import numbers
import math
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from config import gl_conf
import torch.autograd
import torch.multiprocessing as mp
from models import *
from torch.utils.data.sampler import Sampler
from collections import defaultdict

logger = logging.getLogger()

'''
class MxnetImgIter(io.DataIter):
    def __init__(self, batch_size, data_shape,
                 path_imgrec=None,
                 shuffle=False, aug_list=None, mean=None,
                 rand_mirror=False, cutoff=0,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(MxnetImgIter, self).__init__()
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            self.imgrec = recordio.MXIndexedRecordIO(
                path_imgidx, path_imgrec,
                'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                self.imgidx = list(range(1, int(header.label[0])))
                self.id2range = {}
                self.seq_identity = list(range(int(header.label[0]), int(header.label[1])))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    self.id2range[identity] = (a, b)
                    count = b - a
                print('id2range', len(self.id2range))
            else:
                self.imgidx = list(self.imgrec.keys)
            if shuffle:
                self.seq = self.imgidx
                self.oseq = self.imgidx
                print(len(self.seq))
            else:
                self.seq = None
        
        self.mean = mean
        self.nd_mean = None
        if self.mean:
            self.mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
            self.nd_mean = mx.nd.array(self.mean).reshape((1, 1, 3))
        
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d' % (data_shape[1], data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        self.provide_label = [(label_name, (batch_size,))]
        # print(self.provide_label[0][1])
        self.cur = 0
        self.nbatch = 0
        self.is_init = False
    
    def reset(self):
        """Resets the iterator to the beginning of the data."""
        # print('!! call reset()')
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.seq)
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()
    
    def num_samples(self):
        return len(self.seq)
    
    def next_sample(self):
        """Helper function for reading in next sample."""
        # set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
            while True:
                if self.cur >= len(self.seq):
                    raise StopIteration
                idx = self.seq[self.cur]
                self.cur += 1
                if self.imgrec is not None:
                    s = self.imgrec.read_idx(idx)
                    header, img = recordio.unpack(s)
                    label = header.label
                    if not isinstance(label, numbers.Number):
                        label = label[0]
                    return label, img, None, None
                else:
                    label, fname, bbox, landmark = self.imglist[idx]
                    return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None
    
    def brightness_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        src *= alpha
        return src
    
    def contrast_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        src *= alpha
        src += gray
        return src
    
    def saturation_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        src *= alpha
        src += gray
        return src
    
    def color_aug(self, img, x):
        augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
        random.shuffle(augs)
        for aug in augs:
            # print(img.shape)
            img = aug(img, x)
            # print(img.shape)
        return img
    
    def mirror_aug(self, img):
        _rd = random.randint(0, 1)
        if _rd == 1:
            for c in range(img.shape[2]):
                img[:, :, c] = np.fliplr(img[:, :, c])
        return img
    
    __next__ = next
    
    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        """Returns the next batch of data."""
        # print('in next', self.cur, self.labelcur)
        self.nbatch += 1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample()
                _data = self.imdecode(s)
                if self.rand_mirror:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        _data = mx.ndarray.flip(data=_data, axis=1)
                if self.nd_mean is not None:
                    _data = _data.astype('float32')
                    _data -= self.nd_mean
                    _data *= 0.0078125
                if self.cutoff > 0:
                    centerh = random.randint(0, _data.shape[0] - 1)
                    centerw = random.randint(0, _data.shape[1] - 1)
                    half = self.cutoff // 2
                    starth = max(0, centerh - half)
                    endh = min(_data.shape[0], centerh + half)
                    startw = max(0, centerw - half)
                    endw = min(_data.shape[1], centerw + half)
                    _data = _data.astype('float32')
                    # print(starth, endh, startw, endw, _data.shape)
                    _data[starth:endh, startw:endw, :] = 127.5
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                # print('aa',data[0].shape)
                # data = self.augmentation_transform(data)
                # print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    # print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i < batch_size:
                raise StopIteration
        
        return io.DataBatch([batch_data], [batch_label], batch_size - i)
    
    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')
    
    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')
    
    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)  # mx.ndarray
        return img
    
    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img
    
    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data
    
    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))


class MxnetImgDataset(object):
    def __init__(self, train_iter, batch_size=100, root_path=None):
        # self.transform = trans.Compose([
        #     trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        self.train_iter = train_iter
        # self.train_iter_back = iter(train_iter)
        self.batch_size = batch_size
        self.num_classes = 85164
        self.ind = 0
        self.root_path = Path(root_path).parent
    
    def __len__(self):
        return gl_conf.num_steps_per_epoch
    
    def __getitem__(self, indices, ):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        res = self._get_single_item(indices)
        for k, v in res.items():
            assert (
                    isinstance(v, np.ndarray) or
                    isinstance(v, str) or
                    isinstance(v, int) or
                    isinstance(v, np.int64) or
                    torch.is_tensor(v)
            ), type(v)
        return res
    
    def _get_single_item(self, index):
        try:
            next_data_batch = next(self.train_iter)
        except StopIteration:
            print(f'!!! this batch finish {self.ind} ')
            self.train_iter.reset()
            self.ind = 0
            self.train_iter = iter(self.train_iter)
            next_data_batch = next(self.train_iter)
        self.ind += 1
        imgs = next_data_batch.data[0].asnumpy()
        
        imgs = imgs / imgs.max()
        # simply use 0.5 as mean
        imgs -= 0.5
        imgs /= 0.5
        
        # use per sample mean
        
        # imgs -= imgs.mean(axis=(0, 2, 3), keepdims=True)
        # imgs /= imgs.std(axis=(0, 2, 3), keepdims=True)
        
        # use img mean of pytorch
        
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225])
        labels = next_data_batch.label[0].asnumpy()
        return {'imgs': np.array(imgs, dtype=np.float32), 'labels': np.asarray(labels, dtype=np.int64)}


class MxnetLoader():
    def __init__(self, conf):
        # root_path = lz.work_path + 'faces_small/train.rec'
        root_path = conf.ms1m_folder / 'train.rec'
        root_path = str(root_path)
        train_dataiter = MxnetImgIter(
            batch_size=conf.batch_size,
            data_shape=(3, 112, 112),
            path_imgrec=root_path,
            shuffle=True,
            rand_mirror=True,
            mean=None,
            cutoff=0,
        )
        train_dataiter = mx.io.PrefetchingIter(train_dataiter)
        self.dataset = MxnetImgDataset(train_dataiter, batch_size=conf.batch_size, root_path=root_path)
        train_loader = DataLoader(
            self.dataset, batch_size=1, num_workers=0,
        )
        self.train_loader = train_loader
        self.ind = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # if self.ind < len(self):
        for val in self.train_loader:
            imgs, labels = val['imgs'], val['labels']
            imgs = torch.squeeze(imgs)  # .cuda()
            labels = torch.squeeze(labels).long()  # .cuda()
            assert imgs is not None
            self.ind += 1
            return {'imgs': imgs, 'labels': labels}
        # else:
        if self.int >= len(self):
            self.int = 0
        # raise StopIteration()
    
    def __len__(self):
        return len(self.train_loader)
'''


class TorchDataset(object):
    def __init__(self,
                 path_ms1m
                 ):
        self.path_ms1m = path_ms1m
        self.root_path = Path(path_ms1m)
        path_imgrec = str(path_ms1m) + '/train.rec'
        path_imgidx = path_imgrec[0:-4] + ".idx"
        assert os.path.exists(path_imgidx), path_imgidx
        self.path_imgidx = path_imgidx
        self.path_imgrec = path_imgrec
        self.imgrecs = []
        self.locks = []
        lz.timer.since_last_check('start timer for imgrec')
        for num_rec in range(gl_conf.num_recs):
            if num_rec == 1:
                path_imgrec = path_imgrec.replace('/data2/share/', '/share/data/')
            self.imgrecs.append(
                recordio.MXIndexedRecordIO(
                    path_imgidx, path_imgrec,
                    'r')
            )
            self.locks.append(mp.Lock())
        lz.timer.since_last_check(f'{gl_conf.num_recs} imgrec readers init')  # 27 s / 5 reader
        # try:
        #     self.imgidx, self.ids, self.id2range = lz.msgpack_load(str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        #     self.num_classes = len(self.ids)
        # except:
        lz.timer.since_last_check('start cal dataset info')
        s = self.imgrecs[0].read_idx(0)
        header, _ = recordio.unpack(s)
        assert header.flag > 0, 'ms1m or glint ...'
        print('header0 label', header.label)
        self.header0 = (int(header.label[0]), int(header.label[1]))
        self.id2range = {}
        self.imgidx = []
        self.ids = []
        ids_shif = int(header.label[0])
        for identity in list(range(int(header.label[0]), int(header.label[1]))):
            s = self.imgrecs[0].read_idx(identity)
            header, _ = recordio.unpack(s)
            a, b = int(header.label[0]), int(header.label[1])
            if b - a > gl_conf.cutoff:
                self.id2range[identity] = (a, b)
                self.ids.append(identity)
                self.imgidx += list(range(a, b))
        self.ids = np.asarray(self.ids)
        self.num_classes = len(self.ids)
        self.ids_map = {identity - ids_shif: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
        ids_map_tmp = {identity: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
        self.ids = [ids_map_tmp[id_] for id_ in self.ids]
        self.ids = np.asarray(self.ids)
        self.id2range = {ids_map_tmp[id_]: range_ for id_, range_ in self.id2range.items()}
        
        gl_conf.num_clss = self.num_classes
        gl_conf.dop = np.ones(self.ids.max() + 1, dtype=int) * gl_conf.mining_init
        gl_conf.id2range_dop = {str(id_):
                                    np.ones((range_[1] - range_[0],)) *
                                    gl_conf.mining_init for id_, range_ in
                                self.id2range.items()}
        logging.info(f'update num_clss {gl_conf.num_clss} ')
        lz.msgpack_dump([self.imgidx, self.ids, self.id2range], str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        self.cur = 0
        lz.timer.since_last_check('finish cal dataset info')
    
    def __len__(self):
        return len(self.imgidx)
    
    def __getitem__(self, indices, ):
        # if isinstance(indices, (tuple, list)):
        #     return [self._get_single_item(index) for index in indices]
        res = self._get_single_item(indices)
        for k, v in res.items():
            assert (
                    isinstance(v, np.ndarray) or
                    isinstance(v, str) or
                    isinstance(v, int) or
                    isinstance(v, np.int64) or
                    torch.is_tensor(v)
            ), type(v)
        return res
    
    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)  # mx.ndarray
        return img
    
    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))
    
    def _get_single_item(self, index):
        
        # self.cur += 1
        # index += 1  # noneed,  here it index (imgidx) start from 1,.rec start from 1
        # assert index != 0 and index < len(self) + 1 # index can > len(self)
        succ = False
        index, pid, ind_ind = index
        ## rand until lock
        # while True:
        #     for ind_rec in range(len(self.locks)):
        #         succ = self.locks[ind_rec].acquire(timeout=0)
        #         if succ: break
        #     if succ: break
        
        ##  locality based
        ## todo assumm nrec = 2
        if index < self.imgidx[len(self.imgidx) // 2]:
            ind_rec = 0
        else:
            ind_rec = 1
        succ = self.locks[ind_rec].acquire()
        # logging.info(f'use {ind}')
        
        # ind = index // ((max(self.imgidx) + 1) // len(self.locks))
        
        # for ind in range(len(self.locks)):
        #     succ = self.locks[ind].acquire(timeout=0)
        #     if succ: break
        # if not succ:
        #     print(f'not succ ind is {ind}')
        #     self.locks.append(mp.Lock())
        #     self.imgrecs.append(recordio.MXIndexedRecordIO(
        #         self.path_imgidx, self.path_imgrec, 'r'))
        #     print(f'add a imgrec, ttl num {len(self.locks)}')
        #     ind = len(self.locks) - 1
        #     succ = self.locks[ind].acquire()
        
        # succ = self.lock.acquire(timeout=0)
        
        s = self.imgrecs[ind_rec].read_idx(index)  # from [ 1 to 3804846 ]
        rls_succ = self.locks[ind_rec].release()
        header, img = recordio.unpack(s)  # this is BGR format !
        imgs = self.imdecode(img)
        assert imgs is not None
        label = header.label
        if not isinstance(label, numbers.Number):
            assert label[-1] == 0., f'{label} {index} {imgs.shape}'
            label = label[0]
        label = int(label)
        assert label in self.ids_map
        label = self.ids_map[label]
        assert label == pid
        _rd = random.randint(0, 1)
        if _rd == 1:
            imgs = mx.ndarray.flip(data=imgs, axis=1)
        imgs = imgs.asnumpy()
        imgs = imgs / 255.
        # simply use 0.5 as mean
        imgs -= 0.5
        imgs /= 0.5
        imgs = imgs.transpose((2, 0, 1))
        return {'imgs': np.array(imgs, dtype=np.float32), 'labels': label,
                'ind_inds': ind_ind}


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, path, name, transform=None):
        self.carray, self.issame = get_val_pair(path, name)
        self.transform = transform
    
    def __getitem__(self, index):
        if (self.transform):
            fliped_carray = self.transform(torch.tensor(self.carray[index]))
            return {'carray': self.carray[index], 'issame': 1.0 * self.issame[index], 'fliped_carray': fliped_carray}
        else:
            return {'carray': self.carray[index], 'issame': 1.0 * self.issame[index]}
    
    def __len__(self):
        return len(self.issame)


# improve locality and improve load speed!
class RandomIdSampler(Sampler):
    def __init__(self):
        path_ms1m = gl_conf.use_data_folder
        self.imgidx, self.ids, self.id2range = lz.msgpack_load(str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        # above is the imgidx of .rec file
        # remember -1 to convert to pytorch imgidx
        self.num_instances = gl_conf.instances
        self.batch_size = gl_conf.batch_size
        assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {id: (np.asarray(list(range(idxs[0], idxs[1])))).tolist()
                          for id, idxs in self.id2range.items()}  # it index based on 1
        self.ids = list(self.ids)
        # if gl_conf.mining == 'rand.id' or gl_conf.mining == 'rand.img':
        self.nimgs = np.asarray([
            range_[1] - range_[0] for id_, range_ in self.id2range.items()
        ])
        self.nimgs_normed = self.nimgs / self.nimgs.sum()
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.ids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    
    def __len__(self):
        return self.length
    
    def get_batch_ids(self):
        pids = []
        dop = gl_conf.dop
        if gl_conf.mining == 'rand.id':
            pids = np.random.choice(self.ids,
                                    size=int(self.num_pids_per_batch),
                                    p=self.nimgs_normed,
                                    replace=False)
        elif gl_conf.mining == 'imp':
            # lz.logging.info(f'dop smapler {np.count_nonzero( dop == gl_conf.mining_init)} {dop}') # todo
            pids = np.random.choice(self.ids,
                                    size=int(self.num_pids_per_batch),
                                    p=gl_conf.dop / gl_conf.dop.sum(),
                                    replace=False)
        # todo dio with no replacement
        elif gl_conf.mining == 'dop':
            # lz.logging.info(f'dop smapler {np.count_nonzero( dop ==-1)} {dop}')
            nrand_ids = int(self.num_pids_per_batch * gl_conf.rand_ratio)
            pids_now = np.random.choice(self.ids,
                                        size=nrand_ids,
                                        replace=False)
            pids.append(pids_now)
            for _ in range(int(1 / gl_conf.rand_ratio) - 1):
                pids_next = dop[pids_now]
                pids_next[pids_next == -1] = np.random.choice(self.ids,
                                                              size=len(pids_next[pids_next == -1]),
                                                              replace=False)
                pids.append(pids_next)
                pids_now = pids_next
            pids = np.concatenate(pids)
            pids = np.unique(pids)
            if len(pids) < self.num_pids_per_batch:
                pids_now = np.random.choice(np.setdiff1d(self.ids, pids),
                                            size=self.num_pids_per_batch - len(pids),
                                            replace=False)
                pids = np.concatenate((pids, pids_now))
            else:
                pids = pids[: self.num_pids_per_batch]
        # assert len(pids) == np.unique(pids).shape[0]
        
        return pids
    
    def get_batch_idxs(self):
        pids = self.get_batch_ids()
        cnt = 0
        for pid in pids:
            if gl_conf.mining == 'imp':
                assert len(self.index_dic[pid]) == gl_conf.id2range_dop[str(pid)].shape[0]
                ind_inds = np.random.choice(
                    len(self.index_dic[pid]),
                    size=(self.num_instances,), replace=True,
                    p=gl_conf.id2range_dop[str(pid)] / gl_conf.id2range_dop[str(pid)].sum()
                )
            else:
                ind_inds = np.random.choice(
                    len(self.index_dic[pid]),
                    size=(self.num_instances,), replace=True, )
            
            for ind_ind in ind_inds:
                ind = self.index_dic[pid][ind_ind]
                yield ind, pid, ind_ind
                cnt += 1
                if cnt == self.batch_size:
                    break
            if cnt == self.batch_size:
                break
    
    def __iter__(self):
        if gl_conf.mining == 'rand.img':  # quite slow
            for _ in range(len(self)):
                # lz.timer.since_last_check('s next id iter')
                pid = np.random.choice(
                    self.ids, p=self.nimgs_normed,
                )
                ind_ind = np.random.choice(
                    range(len(self.index_dic[pid])),
                )
                ind = self.index_dic[pid][ind_ind]
                # lz.timer.since_last_check('f next id iter')
                yield ind, pid, ind_ind
        else:
            cnt = 0
            while cnt < len(self):
                # logging.info(f'cnt {cnt}')
                for ind, pid, ind_ind in self.get_batch_idxs():
                    cnt += 1
                    yield (ind, pid, ind_ind)


class SeqSampler(Sampler):
    def __init__(self):
        path_ms1m = gl_conf.use_data_folder
        _, self.ids, self.id2range = lz.msgpack_load(path_ms1m / f'info.{gl_conf.cutoff}.pk')
        # above is the imgidx of .rec file
        # remember -1 to convert to pytorch imgidx
        self.num_instances = gl_conf.instances
        self.batch_size = gl_conf.batch_size
        assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {id: (np.asarray(list(range(idxs[0], idxs[1])))).tolist()
                          for id, idxs in self.id2range.items()}  # it index based on 1
        self.ids = list(self.ids)
        
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.ids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        for pid in self.id2range:
            # range_ = self.id2range[pid]
            for ind_ind in range(len(self.index_dic[pid])):
                ind = self.index_dic[pid][ind_ind]
                yield ind, pid, ind_ind


def update_dop_cls(thetas, labels, dop):
    with torch.no_grad():
        bs = thetas.shape[0]
        thetas[torch.arange(0, bs, dtype=torch.long), labels] = thetas.min()
        dop[labels.cpu().numpy()] = torch.argmax(thetas, dim=1).cpu().numpy()


class face_learner(object):
    def __init__(self, conf, inference=False, ):
        logging.info(f'face learner use {conf}')
        if conf.net_mode == 'mobilefacenet':
            self.model = torch.nn.DataParallel(MobileFaceNet(conf.embedding_size)).cuda()
            print('MobileFaceNet model generated')
        elif conf.net_mode == 'nasnetamobile':
            self.model = nasnetamobile(512)
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif conf.net_mode == 'seresnext50':
            self.model = se_resnext50_32x4d(512, )
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif conf.net_mode == 'seresnext101':
            self.model = se_resnext101_32x4d(512)
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = torch.nn.DataParallel(Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)).cuda()
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        else:
            raise ValueError(conf.net_mode)
        if not inference:
            self.milestones = conf.milestones
            ## torch reader
            self.dataset = TorchDataset(gl_conf.use_data_folder)
            
            self.loader = DataLoader(
                self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                shuffle=False, sampler=RandomIdSampler(), drop_last=True,
                pin_memory=True,
            )
            self.class_num = self.dataset.num_classes
            print(self.class_num, 'classes, load ok ')
            if conf.need_log:
                lz.mkdir_p(conf.log_path, delete=True)
            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            if conf.loss == 'arcface':
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num)
            elif conf.loss == 'softmax':
                self.head = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num)
            else:
                raise ValueError(f'{conf.loss}')
            if conf.head_init:
                kernel = lz.msgpack_load(conf.head_init).astype(np.float32).transpose()
                kernel = torch.from_numpy(kernel)
                assert self.head.kernel.shape == kernel.shape
                self.head.kernel.data = kernel
            self.head = self.head.to(conf.device)
            self.head_triplet = TripletLoss().to(conf.device)
            print('two model heads generated')
            
            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            if conf.use_opt == 'adam':
                self.optimizer = optim.Adam([{'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 0},
                                             {'params': paras_only_bn}, ],
                                            betas=(gl_conf.adam_betas1, gl_conf.adam_betas2),
                                            amsgrad=True,
                                            lr=conf.lr,
                                            )
            elif conf.net_mode == 'mobilefacenet':
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': gl_conf.weight_decay},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            
            print(self.optimizer, 'optimizers generated')
            self.board_loss_every = 100  # len(self.loader) // 100
            self.evaluate_every = len(self.loader) // 3
            self.save_every = len(self.loader) // 3
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
                self.loader.dataset.root_path)  # todo postpone load eval
        else:
            pass
    
    def calc_feature(self, out='t.pk'):
        conf = gl_conf
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True,
        )
        features = np.empty((self.dataset.num_classes, 512))
        import collections
        features_tmp = collections.defaultdict(list)
        features_wei = collections.defaultdict(list)
        for ind_data, data in enumerate(loader):
            if ind_data % 99 == 3:
                logging.info(f'{ind_data} / {len(loader)}')
                # break
            imgs = data['imgs']
            labels = data['labels'].numpy()
            imgs = imgs.to(conf.device)
            labels_unique = np.unique(labels)
            with torch.no_grad():
                embeddings = self.model(imgs, normalize=False).cpu().numpy()
            for la in labels_unique:
                features_tmp[la].append(embeddings[labels == la].mean(axis=0))
                features_wei[la].append(np.count_nonzero(labels == la))
        self.nimgs = np.asarray([
            range_[1] - range_[0] for id_, range_ in self.dataset.id2range.items()
        ])
        self.nimgs_normed = self.nimgs / self.nimgs.sum()
        for ind_fea in features_tmp:
            fea_tmp = features_tmp[ind_fea]
            fea_tmp = np.asarray(fea_tmp)
            fea_wei = features_wei[ind_fea]
            fea_wei = np.asarray(fea_wei)
            fea_wei = fea_wei / fea_wei.sum()
            fea_wei = fea_wei.reshape((-1, 1))
            fea = (fea_tmp * fea_wei).sum(axis=0)
            from sklearn.preprocessing import normalize
            print('how many norm', self.nimgs[ind_fea], np.sqrt((fea ** 2).sum()))
            fea = normalize(fea.reshape(1, -1)).flatten()
            features[ind_fea, :] = fea
        lz.msgpack_dump(features, out)
    
    def calc_importance(self, out):
        conf = gl_conf
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True,
        )
        gl_conf.dop = np.ones(ds.ids.max() + 1, dtype=int) * 1e-8
        gl_conf.id2range_dop = {str(id_):
                                    np.ones((range_[1] - range_[0],)) * 1e-8
                                for id_, range_ in
                                ds.id2range.items()}
        gl_conf.sub_imp_loss = {str(id_):
                                    np.ones((range_[1] - range_[0],)) * 1e-8
                                for id_, range_ in
                                ds.id2range.items()}
        for ind_data, data in enumerate(loader):
            if ind_data % 999 == 0:
                logging.info(f'{ind_data} / {len(loader)}')
            imgs = data['imgs']
            labels_cpu = data['labels']
            ind_inds = data['ind_inds']
            imgs = imgs.to(conf.device)
            labels = labels_cpu.to(conf.device)
            
            with torch.no_grad():
                embeddings = self.model(imgs)
            embeddings.requires_grad_(True)
            thetas = self.head(embeddings, labels)
            losses = nn.CrossEntropyLoss(reduction='none')(thetas, labels)
            loss = losses.mean()
            if gl_conf.tri_wei != 0:
                loss_triplet = self.head_triplet(embeddings, labels)
                loss = ((1 - gl_conf.tri_wei) * loss + gl_conf.tri_wei * loss_triplet) / (1 - gl_conf.tri_wei)
            grad = torch.autograd.grad(loss, embeddings,
                                       retain_graph=True, create_graph=False,
                                       only_inputs=True)[0].detach()
            gi = torch.norm(grad, dim=1)
            for lable_, ind_ind_, gi_, loss_ in zip(labels_cpu.numpy(), ind_inds.numpy(), gi.cpu().numpy(),
                                                    losses.detach().cpu().numpy()):
                gl_conf.id2range_dop[str(lable_)][ind_ind_] = gi_
                gl_conf.sub_imp_loss[str(lable_)][ind_ind_] = loss_
                gl_conf.dop[lable_] = gl_conf.id2range_dop[str(lable_)].mean()
        lz.msgpack_dump({'dop': gl_conf.dop,
                         'id2range_dop': gl_conf.id2range_dop,
                         'sub_imp_loss': gl_conf.sub_imp_loss
                         }, out)
    
    def train(self, conf, epochs):
        self.model.train()
        loader = self.loader
        
        if conf.start_eval:
            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                       self.agedb_30_issame)
            self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
            self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp,
                                                                       self.cfp_fp_issame)
            self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        loss_meter = lz.AverageMeter()
        loss_tri_meter = lz.AverageMeter()
        acc_meter = lz.AverageMeter()
        
        # tau = 0
        # B_multi = 4  # todo monitor time
        # Batch_size = gl_conf.batch_size * B_multi
        # batch_size = gl_conf.batch_size
        # tau_thresh = 1.5
        # tau_thresh = (Batch_size + 3 * batch_size) / (3 * batch_size)
        # alpha_tau = .9
        for e in range(conf.start_epoch, epochs):
            # accuracy = 0
            lz.timer.since_last_check('epoch {} started'.format(e))
            for milestone in self.milestones:
                if e == milestone:
                    self.schedule_lr()
            loader_enum = enumerate(loader)
            
            while True:
                try:
                    ind_data, data = loader_enum.__next__()
                except StopIteration as e:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    break
                data_time.update(
                    lz.timer.since_last_check(f'load data ok ind {ind_data}',
                                              verbose=False  # if ind_data > 2 else True
                                              )
                )
                imgs = data['imgs']
                labels_cpu = data['labels']
                ind_inds = data['ind_inds']
                # todo visualize it
                # import torchvision
                # imgs_thumb = torchvision.utils.make_grid(
                #     to_torch(imgs), normalize=True,
                #     nrow=int(np.sqrt(imgs.shape[0])) //4 * 4 ,
                #     scale_each=True).numpy()
                # imgs_thumb = to_img(imgs_thumb)
                # # imgs_thumb = cvb.resize_keep_ar( imgs_thumb, 1024,1024, )
                # plt_imshow(imgs_thumb)
                # plt.savefig(work_path+'t.png')
                # plt.close()
                # logging.info(f'this batch labes {labels} ')
                imgs = imgs.to(conf.device)
                labels = labels_cpu.to(conf.device)
                self.optimizer.zero_grad()
                
                if not conf.fgg:
                    # if tau > tau_thresh and ind_data < len(loader) - B_multi:  # todo enable it
                    #     logging.info('using sampling')
                    #     imgsl = [imgs]
                    #     labelsl = [labels]
                    #     for _ in range(B_multi - 1):
                    #         ind_data, data = loader_enum.__next__()
                    #         imgs = data['imgs']
                    #         labels = data['labels']
                    #         imgs = imgs.to(conf.device)
                    #         labels = labels.to(conf.device)
                    #         imgsl.append(imgs)
                    #         labelsl.append(labels)
                    #     imgs = torch.cat(imgsl, dim=0)
                    #     labels = torch.cat(labelsl, dim=0)
                    #     with torch.no_grad():
                    #         embeddings = self.model(imgs)
                    #     embeddings.requires_grad_(True)
                    #     thetas = self.head(embeddings, labels)
                    #     loss = conf.ce_loss(thetas, labels)
                    #     grad = torch.autograd.grad(loss, embeddings,
                    #                                retain_graph=False, create_graph=False,
                    #                                only_inputs=True)[0].detach()
                    #     gi = torch.norm(grad, dim=1)
                    #     gi = gi / gi.sum()
                    #     G_ind = torch.multinomial(gi, gl_conf.batch_size, replacement=True)
                    #     imgs = imgs[G_ind]
                    #     labels = labels[G_ind]
                    #     gi = gi[G_ind]  # todo this is unbias
                    #     gi = gi / gi.sum()
                    #     wi = 1 / gl_conf.batch_size * (1 / gi)
                    #     embeddings = self.model(imgs)
                    #     thetas = self.head(embeddings, labels)
                    #     loss = (F.cross_entropy(thetas, labels, reduction='none') * wi).mean()
                    #     loss_meter.update(loss.item())
                    #     loss.backward()
                    # else:
                    if conf.finetune:
                        # todo mode for nas resnext .. only
                        embeddings = self.model(imgs, mode='finetune')
                    else:
                        embeddings = self.model(imgs)
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                    acc_meter.update(acc)
                    loss_meter.update(loss.item())
                    if gl_conf.tri_wei != 0:
                        loss_triplet = self.head_triplet(embeddings, labels)
                        loss_tri_meter.update(loss_triplet.item())
                        loss = ((1 - gl_conf.tri_wei) * loss + gl_conf.tri_wei * loss_triplet) / (1 - gl_conf.tri_wei)
                    if gl_conf.mining == 'imp':
                        grad = torch.autograd.grad(loss, embeddings,
                                                   retain_graph=True, create_graph=False,
                                                   only_inputs=True)[0].detach()
                    loss.backward()
                    if gl_conf.use_opt == 'adam':
                        for group in self.optimizer.param_groups:
                            for param in group['params']:
                                param.data = param.data.add(-gl_conf.weight_decay, group['lr'] * param.data)
                    if gl_conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                    if gl_conf.mining == 'imp':
                        gi = torch.norm(grad, dim=1)
                        for lable_, ind_ind_, gi_ in zip(labels_cpu.numpy(), ind_inds.numpy(), gi.cpu().numpy()):
                            gl_conf.id2range_dop[str(lable_)][ind_ind_] = gl_conf.id2range_dop[str(lable_)][
                                                                              ind_ind_] * 0.9 + 0.1 * gi_
                            gl_conf.dop[lable_] = gl_conf.id2range_dop[str(lable_)].sum()  # todo should be sum?
                    if gl_conf.mining == 'rand.id':
                        gl_conf.dop[labels_cpu.numpy()] = 1
                        #     gi = gi / gi.sum()
                    # tau = alpha_tau * tau + (1 - alpha_tau) * (
                    #         1 -
                    #         (1 / (gi ** 2).sum()).item() *
                    #         (torch.norm(gi - 1 / len(gi), dim=0) ** 2).item()
                    # ) ** (-1 / 2)
                elif conf.fgg == 'g':
                    embeddings_o = self.model(imgs)
                    thetas_o = self.head(embeddings_o, labels)
                    loss_o = conf.ce_loss(thetas_o, labels)
                    grad = torch.autograd.grad(loss_o, embeddings_o,
                                               retain_graph=False, create_graph=False,
                                               only_inputs=True)[0].detach()
                    embeddings = embeddings_o + conf.fgg_wei * grad
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                elif conf.fgg == 'gg':
                    embeddings_o = self.model(imgs)
                    thetas_o = self.head(embeddings_o, labels)
                    loss_o = conf.ce_loss(thetas_o, labels)
                    grad = \
                        torch.autograd.grad(loss_o, embeddings_o,
                                            retain_graph=True, create_graph=True,
                                            only_inputs=True)[
                            0]
                    embeddings = embeddings_o + conf.fgg_wei * grad
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                else:
                    raise ValueError(f'{conf.fgg}')
                self.optimizer.step()
                
                if self.step % 100 == 0:
                    logging.info(f'epoch {e} step {self.step}: ' +
                                 # f'img {imgs.mean()} {imgs.max()} {imgs.min()} ' +
                                 f'loss: {loss.item():.3f} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} acc: {acc_meter.avg:.3e} ' +
                                 f'speed: {gl_conf.batch_size/(data_time.avg+loss_time.avg):.2f} imgs/s')
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    # record lr
                    self.writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('loss/ttl',
                                           ((1 - gl_conf.tri_wei) * loss_meter.avg +
                                            gl_conf.tri_wei * loss_tri_meter.avg) / (1 - gl_conf.tri_wei),
                                           self.step)
                    self.writer.add_scalar('loss/xent', loss_meter.avg, self.step)
                    self.writer.add_scalar('loss/triplet', loss_tri_meter.avg, self.step)
                    self.writer.add_scalar('info/acc', acc_meter.avg, self.step)
                    self.writer.add_scalar('info/speed',
                                           gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    self.writer.add_scalar('info/datatime', data_time.avg, self.step)
                    self.writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    self.writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    self.writer.add_histogram('top_imp', dop, self.step)
                    self.writer.add_scalar('info/doprat',
                                           np.count_nonzero(dop == gl_conf.mining_init) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf,
                                                                                          self.loader.dataset.root_path,
                                                                                          'agedb_30')
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    logging.info(f'validation accuracy on agedb_30 is {accuracy} ')
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf,
                                                                                          self.loader.dataset.root_path,
                                                                                          'lfw')
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    logging.info(f'validation accuracy on lfw is {accuracy} ')
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf,
                                                                                          self.loader.dataset.root_path,
                                                                                          'cfp_fp')
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    logging.info(f'validation accuracy on cfp_fp is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                # if self.step % conf.num_steps_per_epoch == 0 and self.step != 0:
                #     break
                loss_time.update(
                    lz.timer.since_last_check('loss ok', verbose=False)
                )
        
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] = params['lr'] * gl_conf.lr_gamma
        logging.info(f'change lr to {params["lr"]}')
    
    def init_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] = gl_conf.lr
        print(self.optimizer, 'lr', params['lr'])
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        
        lz.mkdir_p(save_path, delete=False)
        
        torch.save(
            self.model.state_dict(),
            save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                          extra)))
        lz.msgpack_dump({'dop': gl_conf.dop,
                         'id2range_dop': gl_conf.id2range_dop,
                         }, str(save_path) + f'/extra_{get_time()}_accuracy:{accuracy}_step:{self.step}_{extra}.pk')
        if not model_only:
            torch.save(
                self.head.state_dict(),
                save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                             extra)))
            torch.save(
                self.optimizer.state_dict(),
                save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                  self.step, extra)))
    
    def save(self, path=work_path + 'twoloss.pth'):
        torch.save(self.model, path)
    
    def list_steps(self, resume_path):
        from pathlib import Path
        save_path = Path(resume_path)
        fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
        steps = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
        steps = np.asarray(steps, int)
        return steps
    
    def load_state_by_step(self, resume_path, step, load_model=True, load_head=False, load_opt=False):
        from pathlib import Path
        save_path = Path(resume_path)
        fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
        steps = self.list_steps(resume_path)
        chs = np.where(steps == step)[0]
        chs = int(chs)
        fixed_str = fixed_strs[chs].replace('model_', '')
        modelp = save_path / 'model_{}'.format(fixed_str)
        if load_model:
            self.model.load_state_dict(torch.load(modelp))
        if load_head:
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
        if load_opt:
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
    
    def load_state(self, conf, fixed_str=None, from_save_folder=False,
                   model_only=False, resume_path=None, load_optimizer=True,
                   latest=True, load_imp=False,
                   ):
        from pathlib import Path
        if resume_path:
            save_path = Path(resume_path)
        elif from_save_folder and osp.exists(conf.save_path):
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        modelp = save_path / '{}'.format(fixed_str)
        if not osp.exists(modelp):
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not os.path.exists(modelp):
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:
                # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        # try:
        #     # todo fx it
        #     logging.info(f'load model from {modelp}')
        #     self.model.module.load_state_dict(torch.load(modelp))
        # except:
        logging.info(f'you are using gpu, load model, {modelp}')
        self.model.load_state_dict(torch.load(modelp), strict=False)
        if not model_only:
            logging.info(f'load head and optimizer from {modelp}')
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
            if load_optimizer:
                self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
        if load_imp:
            extra = lz.msgpack_load(save_path / f'extra_{fixed_str.replace(".pth", ".pk")}')
            gl_conf.dop = extra['dop'].copy()
            gl_conf.id2range_dop = extra['id2range_dop'].copy()
    
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
        
        #         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
        #         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
        #         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
    
    def evaluate_accelerate(self, conf, path, name, nrof_folds=5, tta=False):
        logging.info('start eval')
        self.model.eval()  # set the module in evaluation mode
        idx = 0
        if tta:
            dataset = Dataset_val(path, name, transform=hflip)
        else:
            dataset = Dataset_val(path, name)
        loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                            shuffle=False, pin_memory=True)  # todo why shuffle must false
        length = len(dataset)
        embeddings = np.zeros([length, conf.embedding_size])
        issame = np.zeros(length)
        
        with torch.no_grad():
            for data in loader:
                carray_batch = data['carray']
                issame_batch = data['issame']
                if tta:
                    fliped = data['fliped_carray']
                    emb_batch = self.model(carray_batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                            idx + conf.batch_size)] = l2_norm(emb_batch)
                else:
                    embeddings[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                            idx + conf.batch_size)] = self.model(carray_batch.to(conf.device)).cpu()
                issame[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                        idx + conf.batch_size)] = issame_batch
                idx += conf.batch_size
        
        # tpr/fpr is averaged over various fold division
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        self.model.train()
        logging.info('eval end')
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        # accelerate eval
        logging.info('start eval')
        self.model.eval()  # set the module in evaluation mode
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        self.model.train()
        logging.info('eval end')
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def validate(self, conf, fixed_str, from_save_folder=False, model_only=False):
        self.load_state(conf, fixed_str, from_save_folder=from_save_folder, model_only=model_only)
        
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'agedb_30')
        logging.info(f'validation accuracy on agedb_30 is {accuracy} ')
        
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'lfw')
        logging.info(f'validation accuracy on lfw is {accuracy} ')
        
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'cfp_fp')
        logging.info(f'validation accuracy on cfp_fp is {accuracy} ')
    
    def find_lr(self,
                conf,
                init_value=1e-5,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, data in enumerate(self.loader):
            imgs = data['imgs']
            labels = data['labels']
            if i % 100 == 0:
                logging.info(f'ok {i}')
            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1
            
            self.optimizer.zero_grad()
            
            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)
            if gl_conf.tri_wei != 0:
                loss_triplet = self.head_triplet(embeddings, labels)
                loss = (1 - gl_conf.tri_wei) * loss + gl_conf.tri_wei * loss_triplet  # todo
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss, batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            # Do the SGD step
            # Update the lr for the next step
            
            loss.backward()
            self.optimizer.step()
            
            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                print('finish', batch_num, num)
                plt.plot(log_lrs[10:-5], losses[10:-5])
                plt.show()
                plt.savefig('/tmp/tmp.png')
                from IPython import embed
                embed()
                return log_lrs, losses


if __name__ == '__main__':
    pass
    # # test thread safe
    # ds = TorchDataset(lz.share_path2 + '/glint')
    # print(len(ds))
    
    ds = TorchDataset(lz.share_path2 + '/faces_ms1m_112x112')
    print(len(ds))
    
    # print(len(ds))
    #
    #
    # def read(ds):
    #     ind = random.randint(len(ds) - 10, len(ds) - 1)
    #     # ind = random.randint(1, 2)
    #     data = ds[ind]
    #     # logging.info(data['imgs'].shape)
    #     print(ind, data['imgs'].shape)
    #
    #
    # import torch.multiprocessing as mp
    #
    # ps = []
    # for _ in range(100):
    #     p = mp.Process(target=read, args=(ds))
    #     p.start()
    #     ps.append(p)
    #
    # for p in ps:
    #     p.join()
    
    # # test random id smpler
    # lz.timer.since_last_check('start')
    # smpler = RandomIdSampler()
    # for idx in smpler:
    #     print(idx)
    #     break
    # print(len(smpler))
    # lz.timer.since_last_check('construct ')
    # flag = False
    # for ids in smpler:
    #     # print(ids)
    #     ids = np.asarray(ids)
    #     assert np.min(ids) >= 0
    #     if np.isclose(ids, 0):
    #         flag = True
    # print(flag)

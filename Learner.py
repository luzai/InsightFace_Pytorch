import lz
from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm, MySoftmax
from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans

# os.environ['MXNET_CPU_WORKER_NTHREADS'] = "12"
# os.environ['MXNET_ENGINE_TYPE'] = "ThreadedEnginePerDevice"

import os
import random
import logging
import numbers
import math
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio

import itertools

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from config import gl_conf
import torch.autograd
import torch.multiprocessing as mp

logger = logging.getLogger()


# 85k id, 3.8M imgs

class FaceImageIter(io.DataIter):
    def __init__(self, batch_size, data_shape,
                 path_imgrec=None,
                 shuffle=False, aug_list=None, mean=None,
                 rand_mirror=False, cutoff=0,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
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


class Dataset(object):
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
        return {'imgs': np.array(imgs, dtype=np.float32), 'labels': int(labels)}


class Loader2():
    def __init__(self, conf):
        # root_path = lz.work_path + 'faces_small/train.rec'
        root_path = conf.ms1m_folder / 'train.rec'
        root_path = str(root_path)
        train_dataiter = FaceImageIter(
            batch_size=conf.batch_size,
            data_shape=(3, 112, 112),
            path_imgrec=root_path,
            shuffle=True,
            rand_mirror=True,
            mean=None,
            cutoff=0,
        )
        train_dataiter = mx.io.PrefetchingIter(train_dataiter)
        self.dataset = Dataset(train_dataiter, batch_size=conf.batch_size, root_path=root_path)
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
            return imgs, labels
        # else:
        if self.int >= len(self):
            self.int = 0
        # raise StopIteration()

    def __len__(self):
        return len(self.train_loader)


# todo data aug: face iter --> dataset2
# todo clean up path
# todo more dataset (ms1m, vgg, imdb ... )
class Dataset2(object):
    def __init__(self,
                 path_ms1m=lz.share_path2 + 'faces_ms1m_112x112/',
                 ):
        self.path_ms1m = path_ms1m
        self.root_path = Path(path_ms1m)
        path_imgrec = path_ms1m + '/train.rec'
        path_imgidx = path_imgrec[0:-4] + ".idx"
        assert os.path.exists(path_imgidx)
        self.imgrec = recordio.MXIndexedRecordIO(
            path_imgidx, path_imgrec,
            'r')
        self.imgidx, self.ids, self.id2range = lz.msgpack_load(path_ms1m + '/info.pk')

        self.num_classes = len(self.ids)
        self.cur = 0
        self.lock = mp.Lock()

    def __len__(self):
        return len(self.imgidx)

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
        # index = self.imgidx[index]
        index += 1
        assert index != 0 and index < len(self) + 1
        succ = self.lock.acquire()
        # if succ:
        s = self.imgrec.read_idx(index)  # from [ 1 to 3804846 ]
        rls_succ = self.lock.release()
        header, img = recordio.unpack(s)
        imgs = self.imdecode(img)
        label = header.label
        # else:
        #     s = self.imgrec.read_idx(index)  # from [ 1 to 3804846 ]
        #     rls_succ = self.lock.release()
        #     header, img = recordio.unpack(s)
        #     imgs = self.imdecode(img)
        #     label = header.label

        _rd = random.randint(0, 1)
        if _rd == 1:
            imgs = mx.ndarray.flip(data=imgs, axis=1)
        imgs = imgs.asnumpy()
        imgs = imgs / 255.
        # simply use 0.5 as mean
        imgs -= 0.5
        imgs /= 0.5
        imgs = imgs.transpose((2, 0, 1))
        return {'imgs': np.array(imgs, dtype=np.float32), 'labels': int(label)}


class face_learner(object):
    def __init__(self, conf, inference=False, need_loader=True):
        print(conf)
        if conf.use_mobilfacenet:
            # self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            self.model = torch.nn.DataParallel(MobileFaceNet(conf.embedding_size)).cuda()
            print('MobileFaceNet model generated')
        else:
            # self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            self.model = torch.nn.DataParallel(Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)).cuda()
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        if not inference:
            self.milestones = conf.milestones
            if need_loader:
                # self.loader, self.class_num = get_train_loader(conf)

                self.dataset = Dataset2()
                self.loader = DataLoader(
                    self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                    shuffle=True, pin_memory=True
                )

                # self.loader = Loader2(conf)
                self.class_num = 85164
                print(self.class_num, 'classes, load ok ')
            else:
                import copy
                conf_t = copy.deepcopy(conf)
                conf_t.data_mode = 'emore'
                self.loader, self.class_num = get_train_loader(conf_t)
                print(self.class_num)
                self.class_num = 85164
            lz.mkdir_p(conf.log_path, delete=True
                       )
            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            if conf.loss == 'arcface':
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            elif conf.loss == 'softmax':
                self.head = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            else:
                raise ValueError(f'{conf.loss}')

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
            #             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)
            print('optimizers generated')
            self.board_loss_every = 100  # len(self.loader) // 100
            self.evaluate_every = len(self.loader) // 10
            self.save_every = len(self.loader) // 5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
                self.loader.dataset.root_path)
        else:
            self.threshold = conf.threshold

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.
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

        for e in range(epochs):
            accuracy = 0
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            # todo log lr
            for data in loader:
                imgs = data['imgs']
                labels = data['labels']
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()

                if not conf.fgg:
                    embeddings = self.model(imgs)
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
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

                running_loss += loss.item() / conf.batch_size
                self.optimizer.step()
                # print(self.step)
                if self.step % 100 == 0:
                    logging.info(f'epoch {e} step {self.step}: ' +
                                 f'img {imgs.mean()} {imgs.max()} {imgs.min()} ' +
                                 f'loss: {loss.item()/conf.batch_size} ')

                if not conf.no_eval:
                    if self.step % self.board_loss_every == 0 and self.step != 0:
                        loss_board = running_loss / self.board_loss_every
                        self.writer.add_scalar('train_loss', loss_board, self.step)
                        self.scheduler.step(loss_board)
                        running_loss = 0.
                    if self.step % self.evaluate_every == 0 and self.step != 0:
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                                   self.agedb_30_issame)
                        self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                        self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp,
                                                                                   self.cfp_fp_issame)
                        self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    if self.step % self.save_every == 0 and self.step != 0:
                        self.save_state(conf, accuracy)

                self.step += 1
                if self.step % conf.num_steps_per_epoch == 0 and self.step != 0:
                    # print('! break')
                    break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

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
            self.model.state_dict(), save_path /
                                     ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                   extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                                        ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                     extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                             ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                               self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        self.model.load_state_dict(torch.load(save_path / 'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

        #         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
        #         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
        #         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)

    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        # todo accelerate eval
        logging.info('start eval')
        self.model.eval()
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

    def find_lr(self,
                conf,
                init_value=1e-8,
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
        for i, (imgs, labels) in enumerate(self.loader):
            if i % 100 == 0:
                print(i)
            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)

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
                return log_lrs, losses


if __name__ == '__main__':
    # test thread safe
    ds = Dataset2()
    print(len(ds))


    def read(ds):
        ind = random.randint(len(ds) - 10, len(ds) - 1)
        # ind = random.randint(1, 2)
        data = ds[ind]
        # logging.info(data['imgs'].shape)
        print(ind, data['imgs'].shape)


    import torch.multiprocessing as mp

    ps = []
    for _ in range(100):
        p = mp.Process(target=read, args=(ds))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

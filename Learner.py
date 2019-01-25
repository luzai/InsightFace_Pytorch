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
from config import conf as gl_conf
import torch.autograd
import torch.multiprocessing as mp
from models import *
from torch.utils.data.sampler import Sampler
from collections import defaultdict

logger = logging.getLogger()


def unpack_f64(s):
    from mxnet.recordio import IRHeader, _IR_FORMAT, _IR_SIZE, struct
    header = IRHeader(*struct.unpack(_IR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        header = header._replace(label=np.frombuffer(s, np.float64, header.flag))
        s = s[header.flag * 8:]
    return header, s


def unpack_f32(s):
    from mxnet.recordio import IRHeader, _IR_FORMAT, _IR_SIZE, struct
    header = IRHeader(*struct.unpack(_IR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        header = header._replace(label=np.frombuffer(s, np.float32, header.flag))
        s = s[header.flag * 4:]
    return header, s


def unpack_auto(s, fp):
    if 'f64' not in fp and 'alpha' not in fp:
        return unpack_f32(s)
    else:
        return unpack_f64(s)


class TorchDataset(object):
    def __init__(self,
                 path_ms1m, flip=True
                 ):
        self.flip = flip
        self.path_ms1m = path_ms1m
        self.root_path = Path(path_ms1m)
        path_imgrec = str(path_ms1m) + '/train.rec'
        path_imgidx = path_imgrec[0:-4] + ".idx"
        assert os.path.exists(path_imgidx), path_imgidx
        self.path_imgidx = path_imgidx
        self.path_imgrec = path_imgrec
        self.imgrecs = []
        self.locks = []
        
        if gl_conf.use_redis:
            import redis
            
            self.r = redis.Redis()
        else:
            self.r = None
        
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
        lz.timer.since_last_check('start cal dataset info')
        # try:
        #     self.imgidx, self.ids, self.id2range = lz.msgpack_load(str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        #     self.num_classes = len(self.ids)
        # except:
        s = self.imgrecs[0].read_idx(0)
        header, _ = unpack_auto(s, self.path_imgidx)
        assert header.flag > 0, 'ms1m or glint ...'
        print('header0 label', header.label)
        self.header0 = (int(header.label[0]), int(header.label[1]))
        self.id2range = {}
        self.imgidx = []
        self.ids = []
        ids_shif = int(header.label[0])
        for identity in list(range(int(header.label[0]), int(header.label[1]))):
            s = self.imgrecs[0].read_idx(identity)
            header, _ = unpack_auto(s, self.path_imgidx)
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
        lz.msgpack_dump([self.imgidx, self.ids, self.id2range], str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        
        gl_conf.num_clss = self.num_classes
        gl_conf.explored = np.zeros(self.ids.max() + 1, dtype=int)
        if gl_conf.dop is None:
            if gl_conf.mining == 'dop':
                gl_conf.dop = np.ones(self.ids.max() + 1, dtype=int) * gl_conf.mining_init
                gl_conf.id2range_dop = {str(id_):
                                            np.ones((range_[1] - range_[0],)) *
                                            gl_conf.mining_init for id_, range_ in
                                        self.id2range.items()}
            elif gl_conf.mining == 'imp' or gl_conf.mining == 'rand.id':
                gl_conf.id2range_dop = {str(id_):
                                            np.ones((range_[1] - range_[0],)) *
                                            gl_conf.mining_init for id_, range_ in
                                        self.id2range.items()}
                gl_conf.dop = np.asarray([v.sum() for v in gl_conf.id2range_dop.values()])
        logging.info(f'update num_clss {gl_conf.num_clss} ')
        self.cur = 0
        lz.timer.since_last_check('finish cal dataset info')
    
    def __len__(self):
        if gl_conf.local_rank is not None:
            return len(self.imgidx) // torch.distributed.get_world_size()
        else:
            return len(self.imgidx)
    
    def __getitem__(self, indices, ):
        # if isinstance(indices, (tuple, list)) and len(indices[0])==3:
        #    if self.r:
        #        pass
        #    else:
        #        return [self._get_single_item(index) for index in indices]
        res = self._get_single_item(indices)
        # for k, v in res.items():
        #     assert (
        #             isinstance(v, np.ndarray) or
        #             isinstance(v, str) or
        #             isinstance(v, int) or
        #             isinstance(v, np.int64) or
        #             torch.is_tensor(v)
        #     ), type(v)
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
        if self.r:
            img = self.r.get(f'{gl_conf.dataset_name}/imgs/{index}')
            if img is not None:
                # print('hit! ')
                imgs = self.imdecode(img)
                if self.flip and random.randint(0, 1):
                    imgs = mx.ndarray.flip(data=imgs, axis=1)
                imgs = imgs.asnumpy()
                imgs = imgs / 255.
                # simply use 0.5 as mean
                imgs -= 0.5
                imgs /= 0.5
                imgs = imgs.transpose((2, 0, 1))
                return {'imgs': np.array(imgs, dtype=np.float32), 'labels': pid,
                        'ind_inds': ind_ind}
        ## rand until lock
        while True:
            for ind_rec in range(len(self.locks)):
                succ = self.locks[ind_rec].acquire(timeout=0)
                if succ: break
            if succ: break
        
        ##  locality based
        # if index < self.imgidx[len(self.imgidx) // 2]:
        #     ind_rec = 0
        # else:
        #     ind_rec = 1
        # succ = self.locks[ind_rec].acquire()
        
        s = self.imgrecs[ind_rec].read_idx(index)  # from [ 1 to 3804846 ]
        rls_succ = self.locks[ind_rec].release()
        header, img = unpack_auto(s, self.path_imgidx)  # this is RGB format
        imgs = self.imdecode(img)
        assert imgs is not None
        label = header.label
        if not isinstance(label, numbers.Number):
            assert label[-1] == 0. or label[-1] == 1., f'{label} {index} {imgs.shape}'
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
        
        if gl_conf.use_redis and self.r and lz.get_mem() >= 20:
            self.r.set(f'{gl_conf.dataset_name}/imgs/{index}', img)
        #             if np.random.rand()<0.001:
        #             print('not hit', index)
        
        return {'imgs': np.array(imgs, dtype=np.float32), 'labels': label,
                'ind_inds': ind_ind}


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, path, name, transform=None):
        self.carray, self.issame = get_val_pair(path, name)
        self.carray = self.carray[:, ::-1, :, :]  # BGR 2 RGB!
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
    def __init__(self, imgidx, ids, id2range):
        path_ms1m = gl_conf.use_data_folder
        self.imgidx, self.ids, self.id2range = imgidx, ids, id2range
        # self.imgidx, self.ids, self.id2range = lz.msgpack_load(str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        # above is the imgidx of .rec file
        # remember -1 to convert to pytorch imgidx
        self.num_instances = gl_conf.instances
        self.batch_size = gl_conf.batch_size
        if gl_conf.tri_wei != 0:
            assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {id: (np.asarray(list(range(idxs[0], idxs[1])))).tolist()
                          for id, idxs in self.id2range.items()}  # it index based on 1
        self.ids = list(self.ids)
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
        if gl_conf.local_rank is not None:
            return self.length // torch.distributed.get_world_size()
        else:
            return self.length
    
    # TODO JIT?
    def get_batch_ids(self):
        pids = []
        dop = gl_conf.dop
        if gl_conf.mining == 'imp' or gl_conf.mining == 'rand.id':
            # lz.logging.info(f'dop smapler {np.count_nonzero( dop == gl_conf.mining_init)} {dop}')
            pids = np.random.choice(self.ids,
                                    size=int(self.num_pids_per_batch),
                                    p=gl_conf.dop / gl_conf.dop.sum(),
                                    replace=False)
        # todo dop with no replacement
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
            if len(self.index_dic[pid]) < self.num_instances:
                replace = True
            else:
                replace = False
            if gl_conf.mining == 'imp':
                assert len(self.index_dic[pid]) == gl_conf.id2range_dop[str(pid)].shape[0]
                ind_inds = np.random.choice(
                    len(self.index_dic[pid]),
                    size=(self.num_instances,), replace=replace,
                    p=gl_conf.id2range_dop[str(pid)] / gl_conf.id2range_dop[str(pid)].sum()
                )
            else:
                ind_inds = np.random.choice(
                    len(self.index_dic[pid]),
                    size=(self.num_instances,), replace=replace, )
            if gl_conf.chs_first:
                if gl_conf.dataset_name == 'alpha_jk':
                    ind_inds = np.concatenate(([0],ind_inds))
                    ind_inds = np.unique(ind_inds)[:self.num_instances] # 0 must be chsn
                elif gl_conf.dataset_name == 'alpha_f64':
                    if pid>=112145: #112145 开始是证件-监控
                        ind_inds = np.concatenate(([0],ind_inds))
                        ind_inds = np.unique(ind_inds)[:self.num_instances] # 0 must be chsn
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
        # todo dist
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
import pdb
class FaceInfer():
    def __init__(self,conf,gpuid=0):
        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size)
            print('MobileFaceNet model generated')
        elif conf.net_mode == 'nasnetamobile':
            self.model = nasnetamobile(512)
        # elif conf.net_mode == 'seresnext50':
        #     self.model = se_resnext50_32x4d(512, )
        # elif conf.net_mode == 'seresnext101':
        #     self.model = se_resnext101_32x4d(512)
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
        else:
            raise ValueError(conf.net_mode)
#         embed()
#         pdb.set_trace()
        self.model = self.model.eval()
        dev = torch.device(f'cuda:{gpuid}')
        self.model = torch.nn.DataParallel(self.model,
                                           device_ids=[        dev     ], output_device=dev).to(dev)

    def load_state(self, fixed_str=None,
                   resume_path=None, latest=True,
                   ):
        from pathlib import Path
        save_path = Path(resume_path)
        modelp = save_path / '{}'.format(fixed_str)
        if not osp.exists(modelp):
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not osp.exists(modelp):
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:  # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        logging.info(f'you are using gpu, load model, {modelp}')
        model_state_dict = torch.load(modelp,  map_location=lambda storage , loc:storage )
#         embed()
        model_state_dict = {k:v for k,v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict,strict=True, )  # todo later may upgrade
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True,  )
        
            
class face_learner(object):
    def __init__(self, conf, ):
        logging.info(f'face learner use {conf}')
        self.milestones = conf.milestones
        ## torch reader
        self.dataset = TorchDataset(gl_conf.use_data_folder)
        self.val_loader_cache={}
        self.loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False,
            sampler=RandomIdSampler(self.dataset.imgidx,
                                    self.dataset.ids, self.dataset.id2range),
            drop_last=True,
            pin_memory=True,
        )
        self.class_num = self.dataset.num_classes
        print(self.class_num, 'classes, load ok ')
        if conf.need_log:
            lz.mkdir_p(conf.log_path, delete=True)  # todo delete?
            lz.set_file_logger(conf.log_path)
            lz.set_file_logger_prt(conf.log_path)
        self.writer = SummaryWriter(str(conf.log_path))
        self.step = 0
        
        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size)
            print('MobileFaceNet model generated')
        elif conf.net_mode == 'nasnetamobile':
            self.model = nasnetamobile(512)
        elif conf.net_mode == 'resnext':
            self.model = ResNeXt(**{"structure": [3, 4, 6, 3]})
        elif conf.net_mode == 'csmobilefacenet':
            self.model = CSMobileFaceNet(conf.embedding_size)
            print('CSMobileFaceNet model generated')
            # self.model = net_resnext50()
        # elif conf.net_mode == 'seresnext50':
        #     self.model = se_resnext50_32x4d(512, )
        # elif conf.net_mode == 'seresnext101':
        #     self.model = se_resnext101_32x4d(512)
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        else:
            raise ValueError(conf.net_mode)
        
        if conf.backbone_with_head:
            self.head = None
        else:
            if conf.loss == 'arcface':
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num)
            elif conf.loss == 'softmax':
                self.head = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num)
            else:
                raise ValueError(f'{conf.loss}')
        if conf.local_rank is None:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(),
                                                                   device_ids=[conf.local_rank],
                                                                   output_device=conf.local_rank)
        if conf.head_init:
            kernel = lz.msgpack_load(conf.head_init).astype(np.float32).transpose()
            kernel = torch.from_numpy(kernel)
            assert self.head.kernel.shape == kernel.shape
            self.head.kernel.data = kernel
        if self.head is not None:
            self.head = self.head.cuda()
        if gl_conf.tri_wei != 0:
            self.head_triplet = TripletLoss().cuda()
        print('two model heads generated')
        
        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
        if not gl_conf.backbone_with_head:
            if conf.use_opt == 'adam':
                self.optimizer = optim.Adam([{'params': paras_wo_bn + [*self.head.parameters()], 'weight_decay': 0},
                                             {'params': paras_only_bn}, ],
                                            betas=(gl_conf.adam_betas1, gl_conf.adam_betas2),
                                            amsgrad=True,
                                            lr=conf.lr,
                                            )
            elif conf.net_mode == 'mobilefacenet' or 'csmobilefacenet':
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [*self.head.parameters()], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [*self.head.parameters()], 'weight_decay': gl_conf.weight_decay},
                    {'params': paras_only_bn},
                ], lr=conf.lr, momentum=conf.momentum)
        else:
            self.optimizer = optim.SGD([
                {'params': paras_wo_bn, 'weight_decay': gl_conf.weight_decay},
                {'params': paras_only_bn},
            ], lr=conf.lr, momentum=conf.momentum)
            # self.optimizer = optim.SGD(self.model.parameters(), lr=conf.lr,
            #  momentum=conf.momentum,
            #                             weight_decay=conf.weight_decay)
        print(self.optimizer, 'optimizers generated')
        self.board_loss_every = gl_conf.board_loss_every
        self.evaluate_every = len(self.loader) // 3
        self.save_every = len(self.loader) // 3
        self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
            self.loader.dataset.root_path)  # todo postpone load eval
    
    def push2redis(self, limits=6 * 10 ** 6 // 8):
        conf = gl_conf
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True,
        )
        meter = lz.AverageMeter()
        lz.timer.since_last_check(verbose=False)
        for ind_data, data in enumerate(loader):
            meter.update(lz.timer.since_last_check(verbose=False))
            if ind_data % 99 == 0:
                print(ind_data, meter.avg)
            if ind_data > limits:
                break
    
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
            imgs = imgs.cuda()
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
            imgs = imgs.cuda()
            labels = labels_cpu.cuda()
            
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
    
    def finetune(self, conf, epochs):
        self.writer_ft = SummaryWriter(str(conf.log_path)+'/ft')
        self.model.train()
        loader = self.loader
        
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        opt_time = lz.AverageMeter()
        loss_meter = lz.AverageMeter()
        loss_tri_meter = lz.AverageMeter()
        acc_meter = lz.AverageMeter()
        
        for e in range(conf.start_epoch, epochs):
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
                imgs = data['imgs']
                labels_cpu = data['labels']
                ind_inds = data['ind_inds']
                imgs = imgs.cuda()
                labels = labels_cpu.cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                
                if not gl_conf.backbone_with_head:
                    # todo mode for nas resnext .. only
#                         embeddings = self.model(imgs, mode='finetune')
                    with torch.no_grad():
                        embeddings = self.model(imgs,)
                    thetas = self.head(embeddings, labels)
                else:
                    embeddings, thetas = self.model(imgs, labels=labels, return_logits=True)
                # from IPython import embed;embed()
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
                gl_conf.explored[labels_cpu.numpy()] = 1
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
                 
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.step()
                opt_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e} step {self.step}/{len(loader)}: ' +
                                 # f'img {imgs.mean()} {imgs.max()} {imgs.min()} '+
                                 f'loss: {loss.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'opt time: {opt_time.avg:.2f} ' +
                                 f'acc: {acc_meter.avg:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    self.writer_ft.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    self.writer_ft.add_scalar('loss/ttl',
                                           ((1 - gl_conf.tri_wei) * loss_meter.avg +
                                            gl_conf.tri_wei * loss_tri_meter.avg) / (1 - gl_conf.tri_wei),
                                           self.step)
                    self.writer_ft.add_scalar('loss/xent', loss_meter.avg, self.step)
                    self.writer_ft.add_scalar('loss/triplet', loss_tri_meter.avg, self.step)
                    self.writer_ft.add_scalar('info/acc', acc_meter.avg, self.step)
                    self.writer_ft.add_scalar('info/speed',
                                           gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    self.writer_ft.add_scalar('info/datatime', data_time.avg, self.step)
                    self.writer_ft.add_scalar('info/losstime', loss_time.avg, self.step)
                    self.writer_ft.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    self.writer_ft.add_histogram('top_imp', dop, self.step)
                    self.writer_ft.add_scalar('info/doprat',
                                           np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
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
        
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def train(self, conf, epochs):
        self.model.train()
        loader = self.loader
        
        if conf.start_eval:
            accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.agedb_30,
                                                                       self.agedb_30_issame)
            self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
            accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.lfw, self.lfw_issame)
            self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
            accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.cfp_fp,
                                                                       self.cfp_fp_issame)
            self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        opt_time = lz.AverageMeter()
        loss_meter = lz.AverageMeter()
        loss_tri_meter = lz.AverageMeter()
        acc_meter = lz.AverageMeter()
        accuracy = 0

        # tau = 0
        # B_multi = 4  # todo monitor time
        # Batch_size = gl_conf.batch_size * B_multi
        # batch_size = gl_conf.batch_size
        # tau_thresh = 1.5
        # tau_thresh = (Batch_size + 3 * batch_size) / (3 * batch_size)
        # alpha_tau = .9
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = enumerate(loader)
            
            while True:
                try:
                    ind_data, data = loader_enum.__next__()
                except StopIteration as e:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    break
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
                imgs = imgs.cuda()
                labels = labels_cpu.cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
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
                    #         imgs = imgs.cuda()
                    #         labels = labels.cuda()
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
                    gl_conf.explored[labels_cpu.numpy()] = 1
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
                    if not gl_conf.backbone_with_head:
                        if conf.finetune:
                            # todo mode for nas resnext .. only
                            embeddings_o = self.model(imgs, mode='finetune')
                        else:
                            embeddings_o = self.model(imgs)
                        thetas_o = self.head(embeddings, labels)
                    else:
                        embeddings_o, thetas_o = self.model(imgs, labels=labels, return_logits=True)
                    loss_o = conf.ce_loss(thetas_o, labels)
                    grad = torch.autograd.grad(loss_o, embeddings_o,
                                               retain_graph=False, create_graph=False, allow_unused=True,
                                               # todo unsuse?
                                               only_inputs=True)[0].detach()
                    embeddings = embeddings_o + conf.fgg_wei * grad
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                elif conf.fgg == 'gg':
                    embeddings_o = self.model(imgs)
                    thetas_o = self.head(embeddings_o, labels)
                    loss_o = conf.ce_loss(thetas_o, labels)
                    grad = torch.autograd.grad(loss_o, embeddings_o,
                                               retain_graph=True, create_graph=True,
                                               only_inputs=True)[0]
                    embeddings = embeddings_o + conf.fgg_wei * grad
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                else:
                    raise ValueError(f'{conf.fgg}')
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.step()
                opt_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 # f'img {imgs.mean()} {imgs.max()} {imgs.min()} '+
                                 f'loss: {loss.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'opt time: {opt_time.avg:.2f} ' +
                                 f'acc: {acc_meter.avg:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
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
                                           np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
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
        
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def schedule_lr(self,e=0):
        from bisect import bisect_right
        
        e2lr = {epoch: gl_conf.lr * gl_conf.lr_gamma ** bisect_right(self.milestones, epoch) for epoch in
                range(gl_conf.epochs)}
        lr = e2lr[e]
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        logging.info(f'lr is {lr}')
    
    init_lr = schedule_lr
    
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
                emb = self.model(conf.test_transform(img).cuda().unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).cuda().unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).cuda().unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if gl_conf.local_rank is not None and gl_conf.local_rank != 0:
            return
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        time_now = get_time()
        lz.mkdir_p(save_path, delete=False)
        self.model.cpu()
        torch.save(
            self.model.module.state_dict(),
            save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                          extra)))
        self.model.cuda()
        lz.msgpack_dump({'dop': gl_conf.dop,
                         'id2range_dop': gl_conf.id2range_dop,
                         }, str(save_path) + f'/extra_{time_now}_accuracy:{accuracy}_step:{self.step}_{extra}.pk')
        if not model_only:
            if self.head is not None:
                self.head.cpu()
                torch.save(
                    self.head.state_dict(),
                    save_path /
                    ('head_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                                 extra)))
            self.head.cuda()
            torch.save(
                self.optimizer.state_dict(),
                save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy,
                                                                  self.step, extra)))
    
    # def save(self, path=work_path + 'twoloss.pth'):
    #     torch.save(self.model, path)
    
    def list_steps(self, resume_path):
        from pathlib import Path
        save_path = Path(resume_path)
        fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
        steps = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
        steps = np.asarray(steps, int)
        return steps
   
    @staticmethod
    def try_load(model, state_dict):
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            # logger.info(f'{e}')
            pass
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            # logger.info(f'{e}')
            pass
    
    def load_state(self, fixed_str=None,
                   resume_path=None, latest=True,
                   load_optimizer=False, load_imp=False, load_head=False
                   ):
        from pathlib import Path
        save_path = Path(resume_path)
        modelp = save_path / '{}'.format(fixed_str)
        if not osp.exists(modelp):
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not osp.exists(modelp):
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:  # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        logging.info(f'you are using gpu, load model, {modelp}')
        model_state_dict = torch.load(modelp)
#         embed()
        model_state_dict = {k:v for k,v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict,strict=True)  # todo later may upgrade
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True)
        
        if load_head and osp.exists(save_path / 'head_{}'.format(fixed_str)):
            logging.info(f'load head from {modelp}')
            head_state_dict = torch.load(save_path / 'head_{}'.format(fixed_str))
            if self.head is not None:
                self.try_load(self.head, head_state_dict)
            else:
                self.try_load(self.model.module.head, head_state_dict)
        if load_optimizer:
            logging.info(f'load opt from {modelp}')
            self.try_load(self.optimizer, torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
        if load_imp and osp.exists(save_path / f'extra_{fixed_str.replace(".pth", ".pk")}'):
            extra = lz.msgpack_load(save_path / f'extra_{fixed_str.replace(".pth", ".pk")}')
            gl_conf.dop = extra['dop'].copy()
            gl_conf.id2range_dop = extra['id2range_dop'].copy()
    
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
    
    def evaluate_accelerate(self, conf, path, name, nrof_folds=5, tta=False):
        logging.info('start eval')
        self.model.eval()  # set the module in evaluation mode
        idx = 0
        if name in self.val_loader_cache :
            loader =  self.val_loader_cache[name]
        else:
            if tta:
                dataset = Dataset_val(path, name, transform=hflip)
            else:
                dataset = Dataset_val(path, name)
            loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                shuffle=False, pin_memory=True)  # todo why shuffle must false
            self.val_loader_cache[name]=loader
        length = len(loader.dataset)
        embeddings = np.zeros([length, conf.embedding_size])
        issame = np.zeros(length)
        
        with torch.no_grad():
            for data in loader:
                carray_batch = data['carray']
                issame_batch = data['issame']
                if tta:
                    fliped = data['fliped_carray']
                    emb_batch = self.model(carray_batch.cuda()) + self.model(fliped.cuda())
                    embeddings[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                            idx + conf.batch_size)] = l2_norm(emb_batch)
                else:
                    embeddings[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                            idx + conf.batch_size)] = self.model(carray_batch.cuda()).cpu()
                issame[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                        idx + conf.batch_size)] = issame_batch
                idx += conf.batch_size
        
        # tpr/fpr is averaged over various fold division
        try:
            tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
            buf = gen_plot(fpr, tpr)
            roc_curve = Image.open(buf)
            roc_curve_tensor = trans.ToTensor()(roc_curve)
        except Exception as e:
            logging.error(f'{e}')
            roc_curve_tensor = torch.zeros(3, 100, 100)
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
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.cuda()).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.cuda()).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        self.model.train()
        logging.info('eval end')
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def validate(self, conf, resume_path):
        self.load_state(resume_path=resume_path)
        self.model.eval()
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'agedb_30')
        logging.info(f'validation accuracy on agedb_30 is {accuracy} ')
        
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'lfw')
        logging.info(f'validation accuracy on lfw is {accuracy} ')
        
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'cfp_fp')
        logging.info(f'validation accuracy on cfp_fp is {accuracy} ')
        self.model.train()
        
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
            imgs = imgs.cuda()
            labels = labels.cuda()
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
                # from IPython import embed ;   embed()
                return log_lrs, losses


if __name__ == '__main__':
    pass
    # # test thread safe
    
    ds = TorchDataset(lz.share_path2 + '/faces_ms1m_112x112')
    print(len(ds))
    
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

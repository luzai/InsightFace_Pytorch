# -*- coding: future_fstrings -*-
import lz
from lz import *
from models import Backbone, MobileFaceNet, CSMobileFaceNet, l2_norm, Arcface, MySoftmax, TripletLoss
import models
import numpy as np
from verifacation import evaluate
from torch import optim
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras, hflip
from PIL import Image
from torchvision import transforms as trans
import os, random, logging, numbers, math
from pathlib import Path
from torch.utils.data import DataLoader
from config import conf
import torch.autograd
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torch.nn import functional as F

try:
    import mxnet as mx
    from mxnet import ndarray as nd
    from mxnet import recordio
except ImportError:
    logging.warning('if want to train, install mxnet for read rec data')
    conf.training = False

if conf.fp16:
    try:
        # from apex.parallel import DistributedDataParallel as DDP
        # from apex.fp16_utils import *
        from apex import amp
        # amp.register_half_function(torch.nn, 'PReLU')
        # amp.register_half_function(torch.nn.functional, 'prelu')
    except ImportError:
        logging.warning(
            "if want to use fp16, install apex from https://www.github.com/nvidia/apex to run this example.")
        conf.fp16 = False


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


# todo merge this
class Obj():
    pass


def get_rec(p='/data2/share/faces_emore/train.idx'):
    self = Obj()
    self.imgrecs = []
    path_imgidx = p
    path_imgrec = path_imgidx.replace('.idx', '.rec')
    self.imgrecs.append(
        recordio.MXIndexedRecordIO(
            path_imgidx, path_imgrec,
            'r')
    )
    self.lock = mp.Lock()
    self.imgrec = self.imgrecs[0]
    s = self.imgrec.read_idx(0)
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
        #     if b - a > conf.cutoff:
        self.id2range[identity] = (a, b)
        self.ids.append(identity)
        self.imgidx += list(range(a, b))
    self.ids = np.asarray(self.ids)
    self.num_classes = len(self.ids)
    # self.ids_map = {identity - ids_shif: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
    ids_map_tmp = {identity: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
    self.ids = [ids_map_tmp[id_] for id_ in self.ids]
    self.ids = np.asarray(self.ids)
    self.id2range = {ids_map_tmp[id_]: range_ for id_, range_ in self.id2range.items()}
    return self


class DatasetCasia(torch.utils.data.Dataset):
    def __init__(self, path=None):
        self.file = '/data2/share/casia_landmark.txt'
        self.lines = open(self.file).readlines()
        df = pd.read_csv(self.file, sep='\t', header=None)
        self.num_classes = np.unique(df.iloc[:, 1]).shape[0]
        self.root_path = Path('/data2/share/faces_emore/')
        self.cache = {}

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item].split()
        path = '/data2/share/casia/' + line[0]
        if item in self.cache:
            img = self.cache[item]
        else:
            img = open(path, 'rb').read()
            self.cache[item] = img
        fbs = np.asarray(bytearray(img), dtype=np.uint8)
        img = cv2.imdecode(fbs, 1)
        cid = int(line[1])
        kpts = line[2:]
        kpts = np.asarray(kpts, int).reshape(-1, 2)
        warp = lz.preprocess(img, kpts, )
        warp = cvb.bgr2rgb(warp)
        if random.randint(0, 1) == 1:
            warp = warp[:, ::-1, :]  # for 112,112,3 second 112 is left right
        # plt_imshow(img, inp_mode='bgr'); plt.show()
        # plt_imshow(warp, inp_mode = 'bgr'); plt.show()
        warp = warp / 255.
        warp -= 0.5
        warp /= 0.5
        warp = np.array(warp, dtype=np.float32)
        warp = warp.transpose((2, 0, 1))
        return {'imgs': warp, 'labels': int(cid), 'indexes': item + 1,
                'ind_inds': -1, 'is_trains': False}


class DatasetIJBC2(torch.utils.data.Dataset):
    def __init__(self, ):
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        IJBC_path = '/data1/share/IJB_release/' if 'amax' in lz.hostname() else '/home/zl/zl_data/IJB_release/'
        img_list_path = IJBC_path + './IJBC/meta/ijbc_name_5pts_score.txt'
        img_list = open(img_list_path)
        files = img_list.readlines()
        img_path = '/share/data/loose_crop'
        if not osp.exists(img_path):
            img_path = IJBC_path + './IJBC/loose_crop'
        self.img_path = img_path
        self.IJBC_path = IJBC_path
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        import cvbase as cvb
        from PIL import Image
        img_index = item
        each_line = self.files[img_index]
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(self.img_path, name_lmk_score[0])
        img = cvb.read_img(img_name)
        img = cvb.bgr2rgb(img)  # this is RGB
        assert img is not None, img_name
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        warp_img = lz.preprocess(img, landmark=lmk)
        warp_img = Image.fromarray(warp_img)
        img = self.test_transform(warp_img)
        return img


class MegaFaceDisDS(torch.utils.data.Dataset):
    def __init__(self):
        megaface_path = '/data1/share/megaface/'
        # files_scrub = open(f'{megaface_path}/facescrub_lst') .readlines()
        # files_scrub = [f'{megaface_path}/facescrub_images/{f}' for f in files_scrub]
        files_scrub = []
        files_dis = open('f{megaface_path}/megaface_lst').readlines()
        files_dis = [f'{megaface_path}/facescrub_images/{f}' for f in files_dis]
        self.files = files_dis + files_scrub
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        import cvbase as cvb
        img = cvb.read_img(self.files[item])
        img = cvb.bgr2rgb(img)  # this is RGB
        img = self.test_transform(img)
        return img


class DatasetCfpFp(torch.utils.data.Dataset):
    def __init__(self):
        from data.data_pipe import get_val_pair
        carray, issame = get_val_pair('/data2/share/faces_emore', 'cfp_fp')
        carray = np.array(carray, np.float32)
        carray = carray[:, ::-1, :, :].copy()  # to rgb
        self.carray = carray

    def __len__(self):
        return len(self.carray)

    def __getitem__(self, item):
        img = self.carray[item]
        if random.randint(0, 1) == 1:
            img = img[:, :, ::-1].copy()
        return img


class TestDataset(object):
    def __init__(self):
        assert conf.use_test
        if conf.use_test == 'ijbc':
            self.rec_test = DatasetIJBC2()
            self.imglen = len(self.rec_test)
        elif conf.use_test == 'glint':
            self.rec_test = get_rec('/data2/share/glint_test/train.idx')
            self.imglen = max(self.rec_test.imgidx) + 1
        elif conf.use_test == 'cfp_fp':
            self.rec_test = DatasetCfpFp()
            self.imglen = len(self.rec_test)
        else:
            raise ValueError(f'{conf.use_test}')

    def _get_single_item(self, index):
        if conf.use_test == 'ijbc':
            imgs = self.rec_test[index]
        elif conf.use_test == 'cfp_fp':
            imgs = self.rec_test[index]
        elif conf.use_test == 'glint':
            self.rec_test.lock.acquire()
            s = self.rec_test.imgrec.read_idx(index)
            self.rec_test.lock.release()
            header, img = unpack_auto(s, 'glint_test')
            imgs = self.imdecode(img)
            imgs = self.preprocess_img(imgs)
        else:
            raise ValueError(f'{conf.use_test}')
        return {'imgs': np.array(imgs, dtype=np.float32), 'labels': -1, 'indexes': index,
                'ind_inds': -1, 'is_trains': False}

    def __len__(self):
        return self.imglen

    def __getitem__(self, indices, ):
        res = self._get_single_item(indices)
        return res


rec_cache = {}


class TorchDataset(object):
    def __init__(self,
                 path_ms1m,
                 ):
        self.flip = conf.flip
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
        for num_rec in range(conf.num_recs):
            self.imgrecs.append(
                recordio.MXIndexedRecordIO(
                    path_imgidx, path_imgrec,
                    'r')
            )
            self.locks.append(mp.Lock())
        lz.timer.since_last_check(f'{conf.num_recs} imgrec readers init')
        lz.timer.since_last_check('start cal dataset info')
        with self.locks[0]:
            s = self.imgrecs[0].read_idx(0)
        header, _ = unpack_auto(s, self.path_imgidx)
        assert header.flag > 0, 'ms1m or glint ...'
        logging.info(f'header0 label {header.label}')
        self.header0 = (int(header.label[0]), int(header.label[1]))
        self.id2range = {}
        self.idx2id = {}
        self.imgidx = []
        self.ids = []
        ids_shif = int(header.label[0])
        for identity in list(range(int(header.label[0]), int(header.label[1]))):
            s = self.imgrecs[0].read_idx(identity)
            header, _ = unpack_auto(s, self.path_imgidx)
            a, b = int(header.label[0]), int(header.label[1])
            self.id2range[identity] = (a, b)
            self.ids.append(identity)
            self.imgidx += list(range(a, b))
        self.ids = np.asarray(self.ids)
        self.num_classes = len(self.ids)
        self.ids_map = {identity - ids_shif: id2 for identity, id2 in
                        zip(self.ids, range(self.num_classes))}  # now cutoff==0, this is identitical
        ids_map_tmp = {identity: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
        self.ids = np.asarray([ids_map_tmp[id_] for id_ in self.ids])
        self.id2range = {ids_map_tmp[id_]: range_ for id_, range_ in self.id2range.items()}
        for id_, range_ in self.id2range.items():
            for idx_ in range(range_[0], range_[1]):
                self.idx2id[idx_] = id_
        # lz.msgpack_dump([self.imgidx, self.ids, self.id2range], str(path_ms1m) + f'/info.{conf.cutoff}.pk')
        conf.num_clss = self.num_classes
        conf.explored = np.zeros(self.ids.max() + 1, dtype=int)
        if conf.dop is None:
            if conf.mining == 'dop':
                conf.dop = np.ones(self.ids.max() + 1, dtype=int) * conf.mining_init
                conf.id2range_dop = {str(id_):
                                         np.ones((range_[1] - range_[0],)) *
                                         conf.mining_init for id_, range_ in
                                     self.id2range.items()}
            elif conf.mining == 'imp' or conf.mining == 'rand.id':
                conf.id2range_dop = {str(id_):
                                         np.ones((range_[1] - range_[0],)) *
                                         conf.mining_init for id_, range_ in
                                     self.id2range.items()}
                conf.dop = np.asarray([v.sum() for v in conf.id2range_dop.values()])
        logging.info(f'update num_clss {conf.num_clss} ')
        self.cur = 0
        lz.timer.since_last_check('finish cal dataset info')
        if conf.kd and conf.sftlbl_from_file:  # todo deprecated
            self.teacher_embedding_db = lz.Database('work_space/teacher_embedding.h5', 'r')
        if conf.cutoff:
            assert conf.clean_ids is not None, 'not all option combination is implemented'
            id2nimgs = collections.defaultdict(int)
            for id_ in conf.clean_ids:
                id2nimgs[id_] += 1
            abadon_ids_hand = [83192, 47005]
            abadon_ids = np.where(np.array(list(id2nimgs.values())) <= 10)[0]
            abadon_ids = abadon_ids.tolist() + abadon_ids_hand
            ids_remap = {}
            new_id = -1
            for id_ in np.unique(conf.clean_ids):
                if id_ in abadon_ids:
                    ids_remap[id_] = -1
                else:
                    ids_remap[id_] = new_id
                    new_id += 1
            new_ids = []
            for id_ in conf.clean_ids:
                new_ids.append(ids_remap[id_])
            new_ids = np.array(new_ids)
            conf.clean_ids = new_ids
        if conf.clean_ids is not None:
            conf.num_clss = self.num_classes = np.unique(conf.clean_ids).shape[0] - 1

        global rec_cache
        self.rec_cache = rec_cache
        if conf.fill_cache:
            self.fill_cache()

    def fill_cache(self):
        start, stop = min(self.imgidx), max(self.imgidx) + 1
        if isinstance(conf.fill_cache, float):
            stop = start + (stop - start) * conf.fill_cache
            stop = int(stop)
        for index in range(start, stop):
            if index % 9999 == 1:
                logging.info(f'loading {index} ')
            self.rec_cache[index] = self.imgrecs[0].read_idx(index)

    def __len__(self):
        if conf.local_rank is not None:
            return len(self.imgidx) // torch.distributed.get_world_size()
        else:
            return len(self.imgidx) // conf.epoch_less_iter

    def __getitem__(self, indices, ):
        res = self._get_single_item(indices)
        return res

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)  # mx.ndarray
        return img

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))

    def preprocess_img(self, imgs):
        if self.flip and random.randint(0, 1) == 1:
            imgs = mx.ndarray.flip(data=imgs, axis=1)
        imgs = imgs.asnumpy()
        if not conf.fast_load:
            imgs = imgs / 255.
            imgs -= 0.5  # simply use 0.5 as mean
            imgs /= 0.5
            imgs = np.array(imgs, dtype=np.float32)
        imgs = imgs.transpose((2, 0, 1))
        return imgs

    def _get_single_item(self, index):
        # global rec_cache
        if isinstance(index, tuple):
            # assert not isinstance(index, tuple)
            index, pid, ind_ind = index
            index -= 1
        if conf.clean_ids is not None:
            lbl = conf.clean_ids[index]
            if lbl == -1: return self._get_single_item(np.random.randint(low=0, high=len(self)))
        index += 1  # 1 based!
        if index in self.rec_cache:
            s = self.rec_cache[index]
        else:
            with self.locks[0]:
                s = self.imgrecs[0].read_idx(index)  # from [ 1 to 3804846 ]
            # rec_cache[index] = s
        header, img = unpack_auto(s, self.path_imgidx)  # this is RGB format
        imgs = self.imdecode(img)
        assert imgs is not None
        label = header.label
        if not isinstance(label, numbers.Number):
            assert label[-1] == 0. or label[-1] == 1., f'{label} {index} {imgs.shape}'
            label = label[0]
        label = int(label)
        imgs = self.preprocess_img(imgs)
        assert label == int(self.idx2id[index]), f'{label} {self.idx2id[index]}'
        assert label in self.ids_map
        label = self.ids_map[label]
        label = int(label)
        if conf.clean_ids is not None:
            label = lbl
        res = {'imgs': imgs, 'labels': label,
               'indexes': index, 'ind_inds': -1,
               }
        return res


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, path, name, transform=None):
        from data.data_pipe import get_val_pair
        self.carray, self.issame = get_val_pair(path, name)
        self.carray = self.carray[:, ::-1, :, :]  # BGR 2 RGB!
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            fliped_carray = self.transform(torch.tensor(self.carray[index]))
            return {'carray': self.carray[index], 'issame': 1.0 * self.issame[index], 'fliped_carray': fliped_carray}
        else:
            return {'carray': self.carray[index], 'issame': 1.0 * self.issame[index]}

    def __len__(self):
        return len(self.issame)


class RandomIdSampler(Sampler):
    def __init__(self, imgidx, ids, id2range):
        path_ms1m = conf.use_data_folder
        self.imgidx, self.ids, self.id2range = imgidx, ids, id2range
        # above is the imgidx of .rec file
        # remember -1 to convert to pytorch imgidx
        self.num_instances = conf.instances
        self.batch_size = conf.batch_size
        if conf.tri_wei != 0:
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
        if conf.local_rank is not None:
            return self.length // torch.distributed.get_world_size()
        else:
            return self.length

    def get_batch_ids(self):
        pids = []
        dop = conf.dop
        if conf.mining == 'imp' or conf.mining == 'rand.id':
            # lz.logging.info(f'dop smapler {np.count_nonzero( dop == conf.mining_init)} {dop}')
            pids = np.random.choice(self.ids,
                                    size=int(self.num_pids_per_batch),
                                    p=conf.dop / conf.dop.sum(),
                                    # comment, diff smpl diff wei; the smpl in few-smpl cls will more likely to smpled
                                    # not comment, diff smpl same wei;
                                    replace=False
                                    )
        # todo dop with no replacement
        elif conf.mining == 'dop':
            # lz.logging.info(f'dop smapler {np.count_nonzero( dop ==-1)} {dop}')
            nrand_ids = int(self.num_pids_per_batch * conf.rand_ratio)
            pids_now = np.random.choice(self.ids,
                                        size=nrand_ids,
                                        replace=False)
            pids.append(pids_now)
            for _ in range(int(1 / conf.rand_ratio) - 1):
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
            if conf.mining == 'imp':
                assert len(self.index_dic[pid]) == conf.id2range_dop[str(pid)].shape[0]
                ind_inds = np.random.choice(
                    len(self.index_dic[pid]),
                    size=(self.num_instances,), replace=replace,
                    p=conf.id2range_dop[str(pid)] / conf.id2range_dop[str(pid)].sum()
                )
            else:
                ind_inds = np.random.choice(
                    len(self.index_dic[pid]),
                    size=(self.num_instances,), replace=replace, )
            if conf.chs_first:
                if conf.dataset_name == 'alpha_jk':
                    ind_inds = np.concatenate(([0], ind_inds))
                    ind_inds = np.unique(ind_inds)[:self.num_instances]  # 0 must be chsn
                elif conf.dataset_name == 'alpha_f64':
                    if pid >= 112145:  # 112145 开始是证件-监控
                        ind_inds = np.concatenate(([0], ind_inds))
                        ind_inds = np.unique(ind_inds)[:self.num_instances]  # 0 must be chsn
            for ind_ind in ind_inds:
                ind = self.index_dic[pid][ind_ind]
                yield ind, pid, ind_ind
                cnt += 1
                if cnt == self.batch_size:
                    break
            if cnt == self.batch_size:
                break

    def __iter__(self):
        if conf.mining == 'rand.img':  # quite slow
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
        # todo update code
        path_ms1m = conf.use_data_folder
        # _, self.ids, self.id2range = lz.msgpack_load(path_ms1m / f'info.{conf.cutoff}.pk')
        # above is the imgidx of .rec file
        # remember -1 to convert to pytorch imgidx
        self.num_instances = conf.instances
        self.batch_size = conf.batch_size
        assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {id: (np.asarray(list(range(idxs[0], idxs[1])))).tolist()
                          for id, idxs in self.id2range.items()}  # it index based on 1
        self.ids = list(self.ids)

        # estimate number of examples in an epoch
        self.nimgs_lst = [len(idxs) for idxs in self.index_dic.values()]
        self.length = sum(self.nimgs_lst)

    def __len__(self):
        return self.length

    def __iter__(self):
        for pid in self.id2range:
            for ind_ind in range(len(self.index_dic[pid])):
                ind = self.index_dic[pid][ind_ind]
                yield ind, pid, ind_ind


def update_dop_cls(thetas, labels, dop):
    with torch.no_grad():
        bs = thetas.shape[0]
        # logging.info(f'min is {thetas.min()}')
        thetas[torch.arange(0, bs, dtype=torch.long), labels] = -1e4
        dop[labels.cpu().numpy()] = torch.argmax(thetas, dim=1).cpu().numpy()


class FaceInfer():
    def __init__(self, conf=conf, gpuid=(0,)):
        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size)
            print('MobileFaceNet model generated')
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
        elif conf.net_mode == 'mbv3':
            self.model = models.mobilenetv3(mode=conf.mb_mode, width_mult=conf.mb_mult)
        elif conf.net_mode == 'hrnet':
            self.model = models.get_cls_net()
        elif conf.net_mode == 'effnet':
            name = conf.eff_name
            self.model = models.EfficientNet.from_name(name)
            imgsize = models.EfficientNet.get_image_size(name)
            assert conf.input_size == imgsize, imgsize
        else:
            raise ValueError(conf.net_mode)
        self.model = self.model.eval()
        self.model = torch.nn.DataParallel(self.model,
                                           device_ids=list(gpuid), output_device=gpuid[0]).to(gpuid[0])

    def load_model_only(self, fpath):
        model_state_dict = torch.load(fpath, map_location=lambda storage, loc: storage)
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict, strict=True, )
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True, )

    def load_state(self, fixed_str=None,
                   resume_path=None, latest=True,
                   ):
        from pathlib import Path
        save_path = Path(resume_path)
        modelp = save_path / '{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not (modelp).exists() or not modelp.is_file():
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:  # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            assert step.shape[0] > 0, f"{resume_path} chk!"
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        logging.info(f'you are using gpu, load model, {modelp}')
        model_state_dict = torch.load(str(modelp), map_location=lambda storage, loc: storage)
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if conf.cvt_ipabn:
            import copy
            model_state_dict2 = copy.deepcopy(model_state_dict)
            for k in model_state_dict2.keys():
                if 'running_mean' in k:
                    name = k.replace('running_mean', 'weight')
                    model_state_dict2[name] = torch.abs(model_state_dict[name])
            model_state_dict = model_state_dict2
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict, strict=True, )
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True, )


if conf.fast_load:
    class data_prefetcher():
        def __init__(self, loader):
            self.loader = iter(loader)
            self.stream = torch.cuda.Stream()
            self.mean = torch.tensor([.5 * 255, .5 * 255, .5 * 255]).cuda().view(1, 3, 1, 1)
            self.std = torch.tensor([.5 * 255, .5 * 255, .5 * 255]).cuda().view(1, 3, 1, 1)
            # With Amp, it isn't necessary to manually convert data to half.
            # Type conversions are done internally on the fly within patched torch functions.
            if conf.fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
            self.buffer = []
            self.preload()

        def preload(self):
            try:
                ind_loader, data_loader = next(self.loader)
                with torch.cuda.stream(self.stream):
                    data_loader['imgs'] = data_loader['imgs'].cuda()
                    data_loader['labels_cpu'] = data_loader['labels']
                    data_loader['labels'] = data_loader['labels'].cuda()
                    if conf.fp16:
                        data_loader['imgs'] = data_loader['imgs'].half()
                    else:
                        data_loader['imgs'] = data_loader['imgs'].float()
                    data_loader['imgs'] = data_loader['imgs'].sub_(self.mean).div_(self.std)
                    self.buffer.append((ind_loader, data_loader))
            except StopIteration:
                self.buffer.append((None, None))
                return

        def next(self):
            torch.cuda.current_stream().wait_stream(self.stream)
            self.preload()
            res = self.buffer.pop(0)
            return res

        __next__ = next

        def __iter__(self):
            while True:
                ind, data = self.next()
                if ind is None:
                    raise StopIteration
                yield ind, data
else:
    data_prefetcher = lambda x: x


def fast_collate(batch):
    imgs = [img['imgs'] for img in batch]
    targets = torch.tensor([target['labels'] for target in batch], dtype=torch.int64)
    ind_inds = torch.tensor([target['ind_inds'] for target in batch], dtype=torch.int64)
    indexes = torch.tensor([target['indexes'] for target in batch], dtype=torch.int64)
    is_trains = torch.tensor([target['is_trains'] for target in batch], dtype=torch.int64)
    w = imgs[0].shape[1]
    h = imgs[0].shape[2]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        # nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)

    return {'imgs': tensor, 'labels': targets,
            'ind_inds': ind_inds, 'indexes': indexes,
            'is_trains': is_trains}


class face_learner(object):
    def __init__(self, conf=conf, ):
        self.milestones = conf.milestones
        self.val_loader_cache = {}
        ## torch reader
        if conf.dataset_name == 'webface' or conf.dataset_name == 'casia':
            file = '/data2/share/casia_landmark.txt'
            df = pd.read_csv(file, sep='\t', header=None)
            id2nimgs = {}
            for key, frame in df.groupby(1):
                id2nimgs[key] = frame.shape[0]

            nimgs = list(id2nimgs.values())
            nimgs = np.array(nimgs)
            nimgs = nimgs.sum() / nimgs
            id2wei = {ind: wei for ind, wei in enumerate(nimgs)}
            weis = [id2wei[id_] for id_ in np.array(df.iloc[:, 1])]
            weis = np.asarray(weis)
            weis = np.ones((weis.shape[0]))  # todo
            self.dataset = DatasetCasia(conf.use_data_folder, )
            self.loader = DataLoader(self.dataset, batch_size=conf.batch_size,
                                     num_workers=conf.num_workers,
                                     # sampler=torch.utils.data.sampler.WeightedRandomSampler(
                                     #     weis, weis.shape[0],
                                     #     replacement=True
                                     # ),
                                     shuffle=True,
                                     # todo if no-replacement, suddenly large loss on new epoch
                                     # todo whether rebalance
                                     drop_last=True, pin_memory=True, )
            self.class_num = conf.num_clss = self.dataset.num_classes
            conf.explored = np.zeros(self.class_num, dtype=int)
            conf.dop = np.ones(self.class_num, dtype=int) * conf.mining_init
        else:
            self.dataset = TorchDataset(conf.use_data_folder)
            self.loader = DataLoader(
                self.dataset, batch_size=conf.batch_size,
                num_workers=conf.num_workers,
                # sampler=RandomIdSampler(self.dataset.imgidx,
                #                         self.dataset.ids, self.dataset.id2range),
                shuffle=True,
                drop_last=True, pin_memory=conf.pin_memory,
                collate_fn=torch.utils.data.dataloader.default_collate if not conf.fast_load else fast_collate
            )
            self.class_num = self.dataset.num_classes
        logging.info(f'{self.class_num} classes, load ok ')
        if conf.need_log:
            if torch.distributed.is_initialized():
                lz.set_file_logger(str(conf.log_path) + f'/proc{torch.distributed.get_rank()}')
                lz.set_file_logger_prt(str(conf.log_path) + f'/proc{torch.distributed.get_rank()}')
                lz.mkdir_p(conf.log_path, delete=False)
            else:
                lz.mkdir_p(conf.log_path, delete=False)
                lz.set_file_logger(str(conf.log_path))
                lz.set_file_logger_prt(str(conf.log_path))
        if conf.need_tb and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            self.writer = SummaryWriter(str(conf.log_path))
            conf.writer = self.writer
            self.writer.add_text('conf', f'{conf}', 0)  # todo to markdown
        else:
            self.writer = None
        self.step = 0
        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size)
            logging.info('MobileFaceNet model generated')
        elif conf.net_mode == 'effnet':
            name = conf.eff_name
            self.model = models.EfficientNet.from_name(name)
            imgsize = models.EfficientNet.get_image_size(name)
            assert conf.input_size == imgsize, imgsize
        elif conf.net_mode == 'mbfc':
            self.model = models.mbfc()
        elif conf.net_mode == 'sglpth':
            self.model = models.singlepath()
        elif conf.net_mode == 'mbv3':
            self.model = models.mobilenetv3(mode=conf.mb_mode, width_mult=conf.mb_mult)
        elif conf.net_mode == 'hrnet':
            self.model = models.get_cls_net()
        elif conf.net_mode == 'nasnetamobile':
            self.model = models.nasnetamobile(512)
        elif conf.net_mode == 'resnext':
            self.model = models.ResNeXt(**models.resnext._NETS[str(conf.net_depth)])
        elif conf.net_mode == 'csmobilefacenet':
            self.model = CSMobileFaceNet()
            logging.info('CSMobileFaceNet model generated')
        elif conf.net_mode == 'densenet':
            self.model = models.DenseNet(**models.densenet._NETS[str(conf.net_depth)])
        elif conf.net_mode == 'widerresnet':
            self.model = models.WiderResNet(**models.wider_resnet._NETS[str(conf.net_depth)])
            # self.model = models.WiderResNetA2(**models.wider_resnet._NETS[str(conf.net_depth)])
        # elif conf.net_mode == 'seresnext50':
        #     self.model = se_resnext50_32x4d(512, )
        # elif conf.net_mode == 'seresnext101':
        #     self.model = se_resnext101_32x4d(512)
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            logging.info('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        else:
            raise ValueError(conf.net_mode)
        if conf.kd:
            save_path = Path('work_space/emore.r152.cont/save/')
            fixed_str = [t.name for t in save_path.glob('model*_*.pth')][0]
            if not conf.sftlbl_from_file:
                self.teacher_model = Backbone(152, conf.drop_ratio, 'ir_se')
                self.teacher_model = torch.nn.DataParallel(self.teacher_model).cuda()
                modelp = save_path / fixed_str
                model_state_dict = torch.load(modelp)
                model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
                if list(model_state_dict.keys())[0].startswith('module'):
                    self.teacher_model.load_state_dict(model_state_dict, strict=True)
                else:
                    self.teacher_model.module.load_state_dict(model_state_dict, strict=True)
                self.teacher_model.eval()
            self.teacher_head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).cuda()
            head_state_dict = torch.load(save_path / 'head_{}'.format(fixed_str.replace('model_', '')))
            self.teacher_head.load_state_dict(head_state_dict)
        if conf.loss == 'arcface':
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'softmax':
            self.head = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'arcfaceneg':
            from models.model import ArcfaceNeg
            self.head = ArcfaceNeg(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'cosface':
            from models.model import CosFace
            self.head = CosFace(conf.embedding_size, self.class_num)
        elif conf.loss == 'arcface2':
            from models.model import Arcface2
            self.head = Arcface2(conf.embedding_size, self.class_num)
        elif conf.loss == 'adacos':
            from models.model import AdaCos
            self.head = AdaCos(num_classes=self.class_num)
        else:
            raise ValueError(f'{conf.loss}')
        self.model.cuda()

        if conf.head_init:
            kernel = lz.msgpack_load(conf.head_init).astype(np.float32).transpose()
            kernel = torch.from_numpy(kernel)
            assert self.head.kernel.shape == kernel.shape
            self.head.kernel.data = kernel
        if self.head is not None:
            self.head = self.head.cuda()
        if conf.tri_wei != 0:
            self.head_triplet = TripletLoss().cuda()
        logging.info(' model heads generated')

        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
        if conf.use_opt == 'adam':  # todo deprecated
            self.optimizer = optim.Adam([{'params': paras_wo_bn + [*self.head.parameters()], 'weight_decay': 0},
                                         {'params': paras_only_bn}, ],
                                        betas=(conf.adam_betas1, conf.adam_betas2),
                                        amsgrad=True,
                                        lr=conf.lr,
                                        )
        elif conf.use_opt == 'sgd':
            ## wdecay
            # self.optimizer = optim.SGD([
            #     {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},  # this is mobilenet wdecay
            #     {'params': [paras_wo_bn[-1]] + [*self.head.parameters()], 'weight_decay': 4e-4},
            #     {'params': paras_only_bn}], lr=conf.lr, momentum=conf.momentum)

            ## normal
            # self.optimizer = optim.SGD([
            #     {'params': paras_wo_bn + [*self.head.parameters()], 'weight_decay': conf.weight_decay},
            #     {'params': paras_only_bn},
            # ], lr=conf.lr, momentum=conf.momentum)

            ## not fastfc exactly
            # self.optimizer = optim.SGD([
            #     {'params': paras_wo_bn[:-1], 'weight_decay': conf.weight_decay},
            #     {'params': [paras_wo_bn[-1]] + [*self.head.parameters()], 'weight_decay': conf.weight_decay,
            #      'lr_mult': 10},
            #     {'params': paras_only_bn, },
            # ], lr=conf.lr, momentum=conf.momentum, )

            ## fastfc: only head mult 10
            self.optimizer = optim.SGD([
                {'params': paras_wo_bn, 'weight_decay': conf.weight_decay},
                {'params': [*self.head.parameters()], 'weight_decay': conf.weight_decay, 'lr_mult': 10},
                {'params': paras_only_bn, },
            ], lr=conf.lr, momentum=conf.momentum, )

        elif conf.use_opt == 'adabound':
            from tools.adabound import AdaBound
            self.optimizer = AdaBound([
                {'params': paras_wo_bn + [*self.head.parameters()],
                 'weight_decay': conf.weight_decay},
                {'params': paras_only_bn},
            ], lr=conf.lr, betas=(conf.adam_betas1, conf.adam_betas2),
                gamma=1e-3, final_lr=conf.final_lr,
            )
        else:
            raise ValueError(f'{conf.use_opt}')
        if conf.fp16:
            if conf.use_test:
                nloss = 2  # todo
            else:
                nloss = 1
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        logging.info(f'optimizers generated {self.optimizer}')

        if conf.local_rank is None:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(),
                                                                   device_ids=[conf.local_rank],
                                                                   output_device=conf.local_rank)

        self.board_loss_every = conf.board_loss_every
        self.head.train()
        self.model.train()

    def count_flops(self):
        from thop import profile
        flops, params = profile(self.model, input_size=(1, 3, 112, 112))
        return flops, params

    def train_dist(self, conf, epochs):
        self.model.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        dist_need_log = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        if conf.start_eval:
            for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                if dist_need_log:
                    self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                    logging.info(f'validation accuracy on {ds} is {accuracy} ')
        if dist_need_log:
            self.save_state(conf, 0)
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = conf.ce_loss(thetas, labels)
                if conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                if dist_need_log:
                    with torch.no_grad():
                        if conf.mining == 'dop':
                            update_dop_cls(thetas, labels_cpu, conf.dop)
                        if conf.mining == 'rand.id':
                            conf.dop[labels_cpu.numpy()] = 1
                    conf.explored[labels_cpu.numpy()] = 1
                    with torch.no_grad():
                        acc_t = (thetas.argmax(dim=1) == labels)
                        acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if dist_need_log and self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        # writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        if dist_need_log:
                            self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                            logging.info(f'validation accuracy on {ds} is {accuracy} ')

                if dist_need_log and self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def warmup(self, conf, epochs):
        if epochs == 0: return
        self.model.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = SummaryWriter(str(conf.log_path) + '/warmup')
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        e = 0
        ttl_batch = int(epochs * len(loader))
        loader_enum = data_prefetcher(enumerate(loader))
        while True:
            now_lr = conf.lr * (self.step + 1) / ttl_batch
            for params in self.optimizer.param_groups:
                params['lr'] = now_lr
            ind_data, data = next(loader_enum)
            if ind_data is None:
                logging.info(f'one epoch finish {e} {ind_data}')
                loader_enum = data_prefetcher(enumerate(loader))
                ind_data, data = next(loader_enum)
            if (self.step + 1) % len(loader) == 0 or self.step > ttl_batch:
                self.step += 1
                break
            imgs = data['imgs'].cuda()
            assert imgs.max() < 2
            if 'labels_cpu' in data:
                labels_cpu = data['labels_cpu'].cpu()
            else:
                labels_cpu = data['labels'].cpu()
            labels = data['labels'].cuda()
            data_time.update(
                lz.timer.since_last_check(verbose=False)
            )
            self.optimizer.zero_grad()
            embeddings = self.model(imgs, )
            thetas = self.head(embeddings, labels)
            loss_xent = conf.ce_loss(thetas, labels)
            if conf.fp16:
                with amp.scale_loss(loss_xent, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_xent.backward()
            with torch.no_grad():
                if conf.mining == 'dop':
                    update_dop_cls(thetas, labels_cpu, conf.dop)
                if conf.mining == 'rand.id':
                    conf.dop[labels_cpu.numpy()] = 1
            conf.explored[labels_cpu.numpy()] = 1
            with torch.no_grad():
                acc_t = (thetas.argmax(dim=1) == labels)
                acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
            self.optimizer.step()
            loss_time.update(
                lz.timer.since_last_check(verbose=False)
            )
            if self.step % self.board_loss_every == 0:
                logging.info(f'epoch {e}/{epochs} lr {now_lr}' +
                             f'step {self.step}/{len(loader)}: ' +
                             f'xent: {loss_xent.item():.2e} ' +
                             f'data time: {data_time.avg:.2f} ' +
                             f'loss time: {loss_time.avg:.2f} ' +
                             f'acc: {acc:.2e} ' +
                             f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                writer.add_scalar('info/acc', acc, self.step)
                writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                writer.add_scalar('info/datatime', data_time.avg, self.step)
                writer.add_scalar('info/losstime', loss_time.avg, self.step)
                writer.add_scalar('info/epoch', e, self.step)
                dop = conf.dop
                if dop is not None:
                    writer.add_histogram('top_imp', dop, self.step)
                    writer.add_scalar('info/doprat',
                                      np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

            if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                        conf,
                        self.loader.dataset.root_path,
                        ds)
                    self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                    logging.info(f'validation accuracy on {ds} is {accuracy} ')
            if self.step % self.save_every == 0 and self.step != 0:
                self.save_state(conf, accuracy)
            self.step += 1
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def get_loader_enum(self, loader=None):
        import gc
        loader = loader or self.loader
        succ = False
        while not succ:
            try:
                loader_enum = data_prefetcher(enumerate(loader))
                succ = True
            except Exception as e:
                try:
                    del loader_enum
                except:
                    pass
                gc.collect()
                logging.info(f'err is {e}')
                time.sleep(10)
        return loader_enum

    def train_simple(self, conf, epochs):
        self.model.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        if conf.start_eval:
            for ds in ['cfp_fp', ]:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                logging.info(f'validation accuracy on {ds} is {accuracy} ')
        for e in range(conf.start_epoch, epochs):
            if e >= 6:  # todo
                conf.conv2dmask_drop_ratio = 0.
                lambda_runtime_reg = conf.lambda_runtime_reg
            else:  # 0 1 2 3 4 5
                lambda_runtime_reg = 0
            lz.timer.since_last_check('epoch {} started'.format(e))
            # self.schedule_lr(e)
            loader_enum = self.get_loader_enum()
            acc_grad_cnt = 0
            while True:
                self.schedule_lr(step=self.step)
                try:
                    ind_data, data = next(loader_enum)
                except StopIteration as err:
                    logging.info(f'one epoch finish {e} err is {err}')
                    loader_enum = self.get_loader_enum()
                    ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = self.get_loader_enum()
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if acc_grad_cnt == 0:
                    self.optimizer.zero_grad()

                if conf.net_mode == 'sglpth':
                    ttl_runtime = 0.452 * 10 ** 6  # todo
                    target_runtime = 2.5 * 10 ** 6
                    embeddings = self.model(imgs, need_runtime_reg=True)
                    embeddings, runtime_ = embeddings
                    ttl_runtime += runtime_.mean()
                    # runtime_regloss = lambda_runtime_reg * torch.log(ttl_runtime)
                    if ttl_runtime.item() > target_runtime:
                        w_ = 1.03  # todo
                    else:
                        w_ = 0
                    runtime_regloss = lambda_runtime_reg * (ttl_runtime / target_runtime) ** w_
                else:
                    embeddings = self.model(imgs, )
                    ttl_runtime = runtime_regloss = torch.FloatTensor([0]).cuda()

                assert not torch.isnan(embeddings).any().item()
                thetas = self.head(embeddings, labels)
                # loss_xent_all = F.cross_entropy(thetas , labels , reduction='none')
                # loss_xent = loss_xent_all.mean()
                loss_xent = F.cross_entropy(thetas, labels, )
                if conf.fp16:
                    with amp.scale_loss((loss_xent + runtime_regloss) / conf.acc_grad,
                                        self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    ((loss_xent + runtime_regloss) / conf.acc_grad).backward()
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                if acc_grad_cnt == conf.acc_grad - 1:
                    self.optimizer.step()
                    acc_grad_cnt = 0
                else:
                    acc_grad_cnt += 1
                assert acc_grad_cnt < conf.acc_grad, acc_grad_cnt
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'ttlrt: {ttl_runtime.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                if writer and self.step % self.board_loss_every == 0:
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('loss/runtime_regloss', runtime_regloss.item(), self.step)
                    writer.add_scalar('loss/ttl_runtime', ttl_runtime.item(), self.step)
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)

                    if conf.net_mode == 'sglpth':
                        with torch.no_grad():
                            for depth, dec in enumerate(self.model.module.get_decisions()):
                                d5x5, d100c, d50c, t5x5, t100c, t50c = dec
                                self.writer.add_scalar(f'd5x5/{depth}', d5x5, global_step=self.step)
                                self.writer.add_scalar(f'd50c/{depth}', d50c, global_step=self.step)
                                self.writer.add_scalar(f'd100c/{depth}', d100c, global_step=self.step)
                                self.writer.add_scalar(f't100c/{depth}', t100c, global_step=self.step)
                                self.writer.add_scalar(f't50c/{depth}', t50c, global_step=self.step)

                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def train_ghm(self, conf, epochs):
        self.ghm_mom = 0.75
        self.gmax = 350
        self.ginterv = 30
        self.bins = int(self.gmax / self.ginterv) + 1
        self.gmax = self.bins * self.ginterv
        self.edges = np.asarray([self.ginterv * x for x in range(self.bins + 1)])
        self.acc_sum = np.zeros(self.bins)

        self.model.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = nn.CrossEntropyLoss(reduction='none')(thetas, labels)
                grad_xent = torch.autograd.grad(loss_xent.sum(),
                                                embeddings,
                                                retain_graph=True,
                                                create_graph=False, only_inputs=True,
                                                allow_unused=True)[0].detach()
                weights = torch.zeros_like(loss_xent)
                with torch.no_grad():
                    gnorm = grad_xent.norm(dim=1).cpu()
                    tot = grad_xent.shape[0]
                    n_valid_bins = 0
                    for i in range(self.bins):
                        inds = (gnorm >= self.edges[i]) & (gnorm < self.edges[i + 1])
                        num_in_bin = inds.sum().item()
                        if num_in_bin > 0:
                            # self.ghm_mom = 0
                            if self.ghm_mom > 0:
                                self.acc_sum[i] = self.ghm_mom * self.acc_sum[i] \
                                                  + (1 - self.ghm_mom) * num_in_bin
                                weights[inds] = tot / self.acc_sum[i]
                            else:
                                weights[inds] = tot / num_in_bin
                            n_valid_bins += 1
                    if n_valid_bins > 0:
                        weights /= n_valid_bins
                weights /= weights.sum()
                loss_xent = (weights * loss_xent).sum()
                if conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def train_with_wei(self, conf, epochs):
        self.model.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        d = msgpack_load('work_space/wei.pk')
        weis = d['weis']
        edges = d['edges']
        iwidth = edges[1] - edges[0]
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = nn.CrossEntropyLoss(reduction='none')(thetas, labels)
                grad_xent = torch.autograd.grad(loss_xent.sum(),
                                                embeddings,
                                                retain_graph=True,
                                                create_graph=False, only_inputs=True,
                                                allow_unused=True)[0].detach()
                with torch.no_grad():
                    gnorm = grad_xent.norm(dim=1).cpu().numpy()
                    locs = np.ceil((gnorm - edges[0]) / iwidth)
                    locs = np.asarray(locs, int)
                    locs[locs > 99] = 99
                    weis_batch = weis[locs]
                    weis_batch += 1e-5
                    weis_batch /= weis_batch.sum()
                    # plt.plot(weis_batch,); plt.show()
                    weis_batch = to_torch(np.asarray(weis_batch, dtype=np.float32)).cuda()
                loss_xent = (weis_batch * loss_xent).sum()

                if conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def train_use_test(self, conf, epochs):
        self.model.train()
        loader = self.loader
        if conf.use_test:
            loader_test = DataLoader(
                TestDataset(), batch_size=conf.batch_size,
                num_workers=conf.num_workers, shuffle=True,
                drop_last=True, pin_memory=True,
                collate_fn=torch.utils.data.dataloader.default_collate if not conf.fast_load else fast_collate
            )
            loader_test_enum = data_prefetcher(enumerate(loader))
        self.evaluate_every = conf.other_every or len(loader) // (3 * conf.epoch_less_iter)
        self.save_every = conf.other_every or len(loader) // (3 * conf.epoch_less_iter)
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0

        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            if e <= 1:
                conf.use_test = False
            else:
                conf.use_test = 'cfp_fp'  # todo
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ## get data
                try:
                    ind_data, data = next(loader_enum)
                except StopIteration as err:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if conf.use_test:
                    try:
                        ind_data, imgs_test_data = next(loader_test_enum)
                    except StopIteration as err:
                        logging.info(f'test finish {e} {ind_data}')
                        loader_test_enum = data_prefetcher(enumerate(loader_test))
                        ind_data, imgs_test_data = next(loader_enum)
                    if ind_data is None:
                        logging.info(f'test finish {e} {ind_data}')
                        loader_test_enum = data_prefetcher(enumerate(loader_test))
                        ind_data, imgs_test_data = next(loader_enum)
                    imgs_test = imgs_test_data['imgs'].cuda()
                    assert imgs_test.max() < 2
                ## get loss and backward
                self.optimizer.zero_grad()  # todo why must put here # I do not know, but order matters!
                if conf.use_test:
                    loss_vat = conf.vat_loss_func(self.model, self.head, imgs_test)
                    if loss_vat != 0:
                        if conf.fp16:
                            with self.optimizer.scale_loss(loss_vat) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss_vat.backward()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = conf.ce_loss(thetas, labels)
                if conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()  # todo skip error
                else:
                    loss_xent.backward()
                ## post process
                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    if conf.use_test:
                        logging.info(f'vat: {loss_vat.item():.2e} ')
                        writer.add_scalar('loss/vat', loss_vat.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    # todo train_tri
    def train(self, conf, epochs, mode='train', name=None):
        self.model.train()
        if mode == 'train':
            loader = self.loader
            self.evaluate_every = conf.other_every or len(loader) // 3
            self.save_every = conf.other_every or len(loader) // 3
        elif mode == 'finetune':
            loader = DataLoader(
                self.dataset, batch_size=conf.batch_size * conf.ftbs_mult,
                num_workers=conf.num_workers,
                sampler=RandomIdSampler(self.dataset.imgidx,
                                        self.dataset.ids, self.dataset.id2range),
                drop_last=True, pin_memory=True,
                collate_fn=torch.utils.data.dataloader.default_collate if not conf.fast_load else fast_collate
            )
            self.evaluate_every = conf.other_every or len(loader) // 3
            self.save_every = conf.other_every or len(loader) // 3
        else:
            raise ValueError(mode)
        self.step = conf.start_step
        if name is None:
            writer = self.writer
        else:
            writer = SummaryWriter(str(conf.log_path) + '/ft')

        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        tau = 0
        B_multi = 2
        Batch_size = conf.batch_size * B_multi
        batch_size = conf.batch_size
        tau_thresh = 1.2  # todo mv to conf
        #         tau_thresh = (Batch_size + 3 * batch_size) / (3 * batch_size)
        alpha_tau = .9
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))

            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs']
                labels_cpu = data['labels_cpu']
                labels = data['labels']
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
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                #                 if not conf.fgg:
                #                 if np.random.rand()>0.5:
                writer.add_scalar('info/tau', tau, self.step)
                if conf.online_imp and tau > tau_thresh and ind_data < len(loader) - B_multi:
                    writer.add_scalar('info/sampl', 1, self.step)
                    imgsl = [imgs]
                    labelsl = [labels]
                    for _ in range(B_multi - 1):
                        ind_data, data = next(loader_enum)
                        imgs = data['imgs'].cuda()
                        labels = data['labels'].cuda()
                        imgsl.append(imgs)
                        labelsl.append(labels)
                    imgs = torch.cat(imgsl, dim=0)
                    labels = torch.cat(labelsl, dim=0)
                    with torch.no_grad():
                        embeddings = self.model(imgs)
                    embeddings.requires_grad_(True)
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    grad = torch.autograd.grad(loss, embeddings,
                                               retain_graph=False, create_graph=False,
                                               only_inputs=True)[0].detach()
                    grad.requires_grad_(False)
                    with torch.no_grad():
                        gi = torch.norm(grad, dim=1)
                        gi /= gi.sum()
                        G_ind = torch.multinomial(gi, conf.batch_size, replacement=True)
                        imgs = imgs[G_ind]
                        labels = labels[G_ind]
                        gi_b = gi[G_ind]  # todo this is unbias
                        gi_b = gi_b / gi_b.sum()
                        wi = 1 / conf.batch_size * (1 / gi_b)
                    embeddings = self.model(imgs)
                    thetas = self.head(embeddings, labels)
                    loss = (F.cross_entropy(thetas, labels, reduction='none') * wi).mean()
                    if conf.fp16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                else:
                    writer.add_scalar('info/sampl', 0, self.step)
                    embeddings = self.model(imgs, mode=mode)
                    thetas = self.head(embeddings, labels)
                    if not conf.kd:
                        loss = conf.ce_loss(thetas, labels)
                    else:
                        alpha = conf.alpha
                        T = conf.temperature
                        outputs = thetas
                        with torch.no_grad():
                            if not conf.sftlbl_from_file:
                                teachers_embedding = self.teacher_model(imgs, )
                            else:
                                teachers_embedding = data['teacher_embedding']
                            teacher_outputs = self.teacher_head(teachers_embedding, labels)
                        # loss = -(F.softmax(teacher_outputs / T, dim=1) * F.log_softmax(outputs / T, dim=1)).sum(     dim=1).mean() *alpha
                        loss = F.kl_div(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (
                                alpha * T * T)  # todo batchmean
                        loss += F.cross_entropy(outputs, labels) * (1. - alpha)
                    if conf.tri_wei != 0:
                        loss_triplet, info = self.head_triplet(embeddings, labels, return_info=True)
                        grad_tri = torch.autograd.grad(loss_triplet, embeddings, retain_graph=True, create_graph=False,
                                                       only_inputs=True)[0].detach()

                        writer.add_scalar('info/grad_tri', torch.norm(grad_tri, dim=1).mean().item(), self.step)
                        grad_xent = torch.autograd.grad(loss, embeddings, retain_graph=True, create_graph=False,
                                                        only_inputs=True)[0].detach()
                        writer.add_scalar('info/grad_xent', torch.norm(grad_xent, dim=1).mean().item(), self.step)
                        loss = ((1 - conf.tri_wei) * loss + conf.tri_wei * loss_triplet) / (1 - conf.tri_wei)
                    # if conf.online_imp:
                    #     # todo the order not correct
                    #     grad = torch.autograd.grad(loss, embeddings,
                    #                                retain_graph=True, create_graph=False,
                    #                                only_inputs=True)[0].detach()
                    #     grad.requires_grad_(False)
                    #     gi = torch.norm(grad, dim=1)
                    #     gi /= gi.sum()
                    if conf.fp16:
                        with  self.optimizer.scale_loss(loss) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    with torch.no_grad():
                        if conf.mining == 'dop':
                            update_dop_cls(thetas, labels_cpu, conf.dop)
                        #  if conf.mining == 'imp' :
                        #  for lable_, ind_ind_, gi_ in zip(labels_cpu.numpy(), ind_inds.numpy(), gi.cpu().numpy()):
                        #      conf.id2range_dop[str(lable_)][ind_ind_] = conf.id2range_dop[str(lable_)][
                        #      ind_ind_] * 0.9 + 0.1 * gi_
                        #                             conf.dop[lable_] = conf.id2range_dop[str(lable_)].sum()  # todo should be sum?
                        if conf.mining == 'rand.id':
                            conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                if conf.online_imp:
                    tau = alpha_tau * tau + \
                          (1 - alpha_tau) * (1 - (1 / (gi ** 2).sum()).item() * (
                            torch.norm(gi - 1 / len(gi), dim=0) ** 2).item()) ** (-1 / 2)

                #                 elif conf.fgg == 'g':
                #                     embeddings_o = self.model(imgs)
                #                     thetas_o = self.head(embeddings, labels)
                #                     loss_o = conf.ce_loss(thetas_o, labels)
                #                     grad = torch.autograd.grad(loss_o, embeddings_o,
                #                                                retain_graph=False, create_graph=False, allow_unused=True,
                #                                                only_inputs=True)[0].detach()
                #                     embeddings = embeddings_o + conf.fgg_wei * grad
                #                     thetas = self.head(embeddings, labels)
                #                     loss = conf.ce_loss(thetas, labels)
                #                     loss.backward()
                #                 elif conf.fgg == 'gg':
                #                     embeddings_o = self.model(imgs)
                #                     thetas_o = self.head(embeddings_o, labels)
                #                     loss_o = conf.ce_loss(thetas_o, labels)
                #                     grad = torch.autograd.grad(loss_o, embeddings_o,
                #                                                retain_graph=True, create_graph=True,
                #                                                only_inputs=True)[0]
                #                     embeddings = embeddings_o + conf.fgg_wei * grad
                #                     thetas = self.head(embeddings, labels)
                #                     loss = conf.ce_loss(thetas, labels)
                #                     loss.backward()
                #                 else:
                #                     raise ValueError(f'{conf.fgg}')
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 # f'img {imgs.mean()} {imgs.max()} {imgs.min()} '+
                                 f'loss: {loss.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss.item(), self.step)
                    if conf.tri_wei != 0:
                        writer.add_scalar('loss/triplet', loss_triplet.item(), self.step)
                        writer.add_scalar('loss/dap', info['dap'], self.step)
                        writer.add_scalar('loss/dan', info['dan'], self.step)

                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed',
                                      conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    writer.add_histogram('top_imp', dop, self.step)
                    writer.add_scalar('info/doprat',
                                      np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def train_fgg(self, conf, epochs, mode='train', name=None):
        self.model.train()
        if mode == 'train':
            loader = self.loader
            self.evaluate_every = conf.other_every or len(loader) // 3
            self.save_every = conf.other_every or len(loader) // 3
        elif mode == 'finetune':
            loader = DataLoader(
                self.dataset, batch_size=conf.batch_size * conf.ftbs_mult,
                num_workers=conf.num_workers,
                sampler=RandomIdSampler(self.dataset.imgidx,
                                        self.dataset.ids, self.dataset.id2range),
                drop_last=True, pin_memory=True,
                collate_fn=torch.utils.data.dataloader.default_collate if not conf.fast_load else fast_collate
            )
            self.evaluate_every = conf.other_every or len(loader) // 3
            self.save_every = conf.other_every or len(loader) // 3
        else:
            raise ValueError(mode)
        self.step = conf.start_step
        if name is None:
            writer = self.writer
        else:
            writer = SummaryWriter(str(conf.log_path) + '/ft')

        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))

            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs']
                labels_cpu = data['labels_cpu']
                labels = data['labels']
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode=mode)
                thetas = self.head(embeddings, labels)
                if not conf.fgg:
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                elif conf.fgg == 'g':
                    embeddings_o = self.model(imgs)
                    thetas_o = self.head(embeddings, labels)
                    loss_o = conf.ce_loss(thetas_o, labels)
                    grad = torch.autograd.grad(loss_o, embeddings_o,
                                               retain_graph=False, create_graph=False, allow_unused=True,
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

                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1

                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]

                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 # f'img {imgs.mean()} {imgs.max()} {imgs.min()} '+
                                 f'loss: {loss.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed',
                                      conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    writer.add_histogram('top_imp', dop, self.step)
                    writer.add_scalar('info/doprat',
                                      np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def set_lr(self, lr):
        for params in self.optimizer.param_groups:
            if 'lr_mult' in params:
                params['lr'] = lr * params['lr_mult']
            else:
                params['lr'] = lr

    def schedule_lr_mstep(self, e=0, step=0):
        from bisect import bisect_right
        steps_per_epoch = len(self.loader)
        if step != 0:
            e = (step + 1) // steps_per_epoch
        e2lr = {epoch: conf.lr * conf.lr_gamma ** bisect_right(self.milestones, epoch) for epoch in
                range(conf.epochs)}
        # logging.info(f'map e to lr is {e2lr}')
        lr = e2lr[e]
        for params in self.optimizer.param_groups:
            if 'lr_mult' in params:
                params['lr'] = lr * params['lr_mult']
            else:
                params['lr'] = lr
        # logging.info(f'lr is {lr}')

    def schedule_lr_cosanl(self, e=0, step=0):
        assert e == 0, 'not implement yet'
        steps_per_epoch = len(self.loader)
        ttl_steps = conf.epochs * steps_per_epoch
        base_lr = conf.lr
        # min_lr = base_lr/10 # defalt use 0
        lr = 1 / 2 * (base_lr) * (1 + np.cos(step / ttl_steps * np.pi))
        for params in self.optimizer.param_groups:
            if 'lr_mult' in params:
                params['lr'] = lr * params['lr_mult']
            else:
                params['lr'] = lr

    # schedule_lr =init_lr = schedule_lr_mstep
    schedule_lr = init_lr = schedule_lr_cosanl

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        # if conf.local_rank is not None and conf.local_rank != 0:
        #     return
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        time_now = get_time()
        lz.mkdir_p(save_path, delete=False)
        # self.model.cpu()
        torch.save(
            self.model.module.state_dict(),
            save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                          extra)))
        # self.model.cuda()
        lz.msgpack_dump({'dop': conf.dop,
                         'id2range_dop': conf.id2range_dop,
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

    def list_steps(self, resume_path):
        from pathlib import Path
        save_path = Path(resume_path)
        fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
        steps = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
        steps = np.asarray(steps, int)
        return steps

    def list_fixed_strs(self, resume_path):
        from pathlib import Path

        save_path = Path(resume_path)
        fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
        steps = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
        steps = np.asarray(steps, int)
        fixed_strs = np.asarray(fixed_strs)
        fixed_strs = fixed_strs[np.argsort(steps)]
        fixed_strs = [fx.replace('model_', '') for fx in fixed_strs]
        return fixed_strs

    def load_state(self, fixed_str='',
                   resume_path=None, latest=True,
                   load_optimizer=False, load_imp=False, load_head=False
                   ):
        from pathlib import Path
        save_path = Path(resume_path)
        modelp = save_path / '{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:  # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            assert step.shape[0] > 0, f"{resume_path} chk!"
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        logging.info(f'you are using gpu, load model, {modelp}')
        model_state_dict = torch.load(modelp)
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if conf.cvt_ipabn:
            import copy
            model_state_dict2 = copy.deepcopy(model_state_dict)
            for k in model_state_dict2.keys():
                if 'running_mean' in k:
                    name = k.replace('running_mean', 'weight')
                    model_state_dict2[name] = torch.abs(model_state_dict[name])
            model_state_dict = model_state_dict2
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict, strict=True)
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True)

        if load_head:
            assert osp.exists(save_path / 'head_{}'.format(fixed_str))
            logging.info(f'load head from {modelp}')
            head_state_dict = torch.load(save_path / 'head_{}'.format(fixed_str))
            self.head.load_state_dict(head_state_dict)
        if load_optimizer:
            logging.info(f'load opt from {modelp}')
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
        if load_imp and (save_path / f'extra_{fixed_str.replace(".pth", ".pk")}').exists():
            extra = lz.msgpack_load(save_path / f'extra_{fixed_str.replace(".pth", ".pk")}')
            conf.dop = extra['dop'].copy()
            conf.id2range_dop = extra['id2range_dop'].copy()

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor, writer=None):
        writer = writer or self.writer
        if writer:
            writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
            writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
            # writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    def evaluate(self, conf, path, name, nrof_folds=10, tta=True):
        # from utils import ccrop_batch
        self.model.eval()
        roc_curve_tensor = None
        idx = 0
        if name in self.val_loader_cache:
            carray, issame = self.val_loader_cache[name]
        else:
            from data.data_pipe import get_val_pair
            carray, issame = get_val_pair(path, name)
            self.val_loader_cache[name] = carray, issame
        carray = carray[:, ::-1, :, :].copy()  # BGR 2 RGB!
        embeddings = np.zeros([len(carray), conf.embedding_size])
        try:
            with torch.no_grad():
                while idx + conf.batch_size <= len(carray):
                    batch = torch.tensor(carray[idx:idx + conf.batch_size])
                    if tta:
                        # batch = ccrop_batch(batch)
                        fliped = hflip_batch(batch)
                        emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                        emb_batch = emb_batch.cpu()
                        embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                    else:
                        embeddings[idx:idx + conf.batch_size] = self.model(batch.cuda()).cpu()
                    idx += conf.batch_size
                if idx < len(carray):
                    batch = torch.tensor(carray[idx:])
                    if tta:
                        # batch = ccrop_batch(batch)
                        fliped = hflip_batch(batch)
                        emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                        emb_batch = emb_batch.cpu()
                        embeddings[idx:] = l2_norm(emb_batch)
                    else:
                        embeddings[idx:] = self.model(batch.cuda()).cpu()
            tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
            res = accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
            # buf = gen_plot(fpr, tpr)
            # roc_curve = Image.open(buf)
            # roc_curve_tensor = trans.ToTensor()(roc_curve)
        except Exception as e:
            logging.info(f'exp is {e}')
            res = 0, 0, None
        self.model.train()
        return res

    evaluate_accelerate = evaluate

    # todo this evaluate is depracated
    def evaluate_accelerate_dingyi(self, conf, path, name, nrof_folds=10, tta=True):
        lz.timer.since_last_check('start eval')
        self.model.eval()  # set the module in evaluation mode
        idx = 0
        if name in self.val_loader_cache:
            loader = self.val_loader_cache[name]
        else:
            if tta:
                dataset = Dataset_val(path, name, transform=hflip)
            else:
                dataset = Dataset_val(path, name)
            loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=0,
                                shuffle=False, pin_memory=False)
            self.val_loader_cache[name] = loader  # because we have limited memory
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
                    emb_batch = emb_batch.cpu()
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
        #     buf = gen_plot(fpr, tpr)
        #     roc_curve = Image.open(buf)
        #     roc_curve_tensor = trans.ToTensor()(roc_curve)
        except Exception as e:
            logging.error(f'{e}')
        roc_curve_tensor = torch.zeros(3, 100, 100)
        self.model.train()
        lz.timer.since_last_check('eval end')
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def validate_ori(self, conf, resume_path=None,
                     valds_names=('lfw', 'agedb_30', 'cfp_fp',
                                  'cfp_ff', 'calfw', 'cplfw', 'vgg2_fp',)):
        res = {}
        if resume_path is not None:
            self.load_state(resume_path=resume_path)
        self.model.eval()
        for ds in valds_names:
            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.loader.dataset.root_path,
                                                                       ds)
            logging.info(f'validation accuracy on {ds} is {accuracy} ')
            res[ds] = accuracy

        self.model.train()
        return res

    # todo deprecated
    def validate_dingyi(self, conf, resume_path=None):
        if resume_path is not None:
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

    def push2redis(self, limits=6 * 10 ** 6 // 8):
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True, collate_fn=torch.utils.data.default_collate if not conf.fast_load else fast_collate
        )
        meter = lz.AverageMeter()
        lz.timer.since_last_check(verbose=False)
        for ind_data, data in enumerate(loader):
            meter.update(lz.timer.since_last_check(verbose=False))
            if ind_data % 99 == 0:
                print(ind_data, meter.avg)
            if ind_data > limits:
                break

    def calc_teacher_logits(self, out='work_space/teacher_embedding.h5'):
        db = lz.Database(out, 'a')
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True,
            collate_fn=torch.utils.data.dataloader.default_collate if not conf.fast_load else fast_collate
        )
        for ind_data, data in data_prefetcher(enumerate(loader)):
            if ind_data % 99 == 3:
                logging.info(f'{ind_data} / {len(loader)}')
            indexes = data['indexes'].numpy()
            imgs = data['imgs']
            labels = data['labels']
            with torch.no_grad():
                if str(indexes.max()) in db:
                    print('skip', indexes.max())
                    continue
                embeddings = self.teacher_model(imgs, )
                outputs = embeddings.cpu().numpy()
                for index, output in zip(indexes, outputs):
                    db[str(index)] = output
        db.flush()
        db.close()

    def calc_img_feas(self, out='t.h5'):
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=conf.batch_size,
                            num_workers=conf.num_workers, shuffle=False,
                            drop_last=False, pin_memory=True, )
        self.class_num = conf.num_clss = self.dataset.num_classes
        import h5py
        from sklearn.preprocessing import normalize

        f = h5py.File(out, 'w')
        chunksize = 80 * 10 ** 3
        dst = f.create_dataset("feas", (chunksize, 512), maxshape=(None, 512), dtype='f2')
        dst_gtri = f.create_dataset("gtri", (chunksize, 512), maxshape=(None, 512), dtype='f2')
        dst_gxent = f.create_dataset("gxent", (chunksize, 512), maxshape=(None, 512), dtype='f2')
        dst_tri = f.create_dataset("tri", (chunksize,), maxshape=(None,), dtype='f2')
        dst_xent = f.create_dataset("xent", (chunksize,), maxshape=(None,), dtype='f2')
        dst_gtri_norm = f.create_dataset("gtri_norm", (chunksize,), maxshape=(None,), dtype='f2')
        dst_gxent_norm = f.create_dataset("gxent_norm", (chunksize,), maxshape=(None,), dtype='f2')
        dst_img = f.create_dataset("img", (chunksize, 3, 112, 112), maxshape=(None, 3, 112, 112), dtype='f2')
        dst_inds = f.create_dataset("inds", (chunksize,), maxshape=(None,), dtype='f4')
        ind_dst = 0
        for ind_data, data in data_prefetcher(enumerate(loader)):
            imgs = data['imgs'].cuda()
            labels = data['labels'].cuda()
            bs = imgs.shape[0]
            if ind_dst + bs > dst.shape[0]:
                dst.resize((dst.shape[0] + chunksize, 512), )
                dst_gxent.resize((dst.shape[0] + chunksize, 512), )
                dst_xent.resize((dst.shape[0] + chunksize,), )
                dst_xent[dst.shape[0]:dst.shape[0] + chunksize] = -1
                dst_gxent_norm.resize((dst.shape[0] + chunksize,), )
                dst_inds.resize((dst.shape[0] + chunksize,))
                # dst_img.resize((dst.shape[0] + chunksize, 3, 112, 112))
            # assert (data['indexes'].numpy() == np.arange(ind_dst + 1, ind_dst + bs + 1)).all()
            dst_inds[ind_dst:ind_dst + bs] = data['indexes'].numpy()

            with torch.no_grad():
                embeddings = self.model(imgs, )
            embeddings.requires_grad_(True)
            thetas = self.head(embeddings, labels)
            loss_xent = nn.CrossEntropyLoss(reduction='sum')(thetas, labels)  # for grad of each sample
            grad_xent = torch.autograd.grad(loss_xent,
                                            embeddings,
                                            retain_graph=False,
                                            create_graph=False, only_inputs=True,
                                            allow_unused=True)[0].detach()

            with torch.no_grad():
                dst[ind_dst:ind_dst + bs, :] = (embeddings.cpu().numpy()).astype(np.float16)
                # dst_img[ind_dst:ind_dst + bs, :, :, :] = imgs.cpu().numpy()
                dst_gxent[ind_dst:ind_dst + bs, :] = grad_xent.cpu().numpy().astype(np.float16)
                dst_gxent_norm[ind_dst:ind_dst + bs] = grad_xent.norm(dim=1).cpu().numpy()
                dst_xent[ind_dst:ind_dst + bs] = nn.CrossEntropyLoss(reduction='none')(thetas, labels).cpu().numpy()
            ind_dst += bs

            if ind_data % 99 == 0:
                logging.info(f'{ind_data} / {len(loader)}, {loss_xent.item()} {grad_xent.norm(dim=1)[0].item()}')
                # break
        f.flush()
        f.close()
        self.model.train()

    def find_lr(self,
                init_value=1e-5,
                final_value=100.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.set_lr(lr)
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        loader_enum = self.get_loader_enum()
        acc_grad_cnt = 0
        while True:
            try:
                ind_data, data = next(loader_enum)
            except StopIteration as err:
                logging.info(f'one epoch finish err is {err}')
                loader_enum = self.get_loader_enum()
                ind_data, data = next(loader_enum)
            imgs = data['imgs'].cuda()
            labels = data['labels'].cuda()
            if batch_num % 100 == 0:
                logging.info(f'ok {batch_num}/{num} lr {lr} {acc_grad_cnt}')
            if acc_grad_cnt == 0:
                self.optimizer.zero_grad()
            embeddings = self.model(imgs)
            if torch.isnan(embeddings).any().item():
                break
            thetas = self.head(embeddings, labels)
            if not conf.kd:
                loss = conf.ce_loss(thetas, labels)
            else:
                alpha = conf.alpha
                T = conf.temperature
                outputs = thetas
                with torch.no_grad():
                    teachers_embedding = self.teacher_model(imgs, )
                    teacher_outputs = self.head(teachers_embedding, labels)
                loss = -(F.softmax(teacher_outputs / T, dim=1) * F.log_softmax(outputs / T, dim=1)).sum(
                    dim=1).mean() * T * T * alpha + \
                       F.cross_entropy(outputs, labels) * (1. - alpha)  # todo wrong here
            if conf.tri_wei != 0:
                loss_triplet = self.head_triplet(embeddings, labels)
                loss = (1 - conf.tri_wei) * loss + conf.tri_wei * loss_triplet  # todo
            # Do the SGD step
            # Update the lr for the next step
            if torch.isnan(loss).any():
                break
            if conf.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if acc_grad_cnt == conf.acc_grad - 1:
                self.optimizer.step()
                acc_grad_cnt = 0
                batch_num += 1
                # Compute the smoothed loss
                self.writer.add_scalar('lr/loss', loss.item(), batch_num)
                avg_loss = beta * avg_loss + (1 - beta) * loss.item()
                self.writer.add_scalar('lr/avg_loss', avg_loss, batch_num)
                smoothed_loss = avg_loss / (1 - beta ** batch_num)
                self.writer.add_scalar('lr/smoothed_loss', smoothed_loss, batch_num)
                # Stop if the loss is exploding
                # if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                #     logging.info('exited with best_loss at {}'.format(best_loss))
                #     plt.plot(log_lrs[10:-5], losses[10:-5])
                #     plt.show()
                #     return log_lrs, losses
                # Record the best loss
                if smoothed_loss < best_loss or batch_num == 1:
                    best_loss = smoothed_loss
                # Store the values
                losses.append(smoothed_loss)
                log_lrs.append(math.log10(lr))
                self.writer.add_scalar('lr/log_lr', math.log10(lr), batch_num)
                self.writer.add_scalar('lr/lr', lr, batch_num)
                lr *= mult
                self.set_lr(lr)
            else:
                acc_grad_cnt += 1
            assert acc_grad_cnt < conf.acc_grad, acc_grad_cnt
            if batch_num > num:
                break
        print('finish', batch_num, num)
        plt.plot(log_lrs[10:-5], losses[10:-5])
        plt.show()
        plt.savefig('/tmp/tmp.png')
        return log_lrs, losses


class face_cotching(face_learner):
    def __init__(self, conf=conf, ):
        self.milestones = conf.milestones
        self.val_loader_cache = {}
        ## torch reader
        if conf.dataset_name == 'webface' or conf.dataset_name == 'casia':
            file = '/data2/share/casia_landmark.txt'
            df = pd.read_csv(file, sep='\t', header=None)
            id2nimgs = {}
            for key, frame in df.groupby(1):
                id2nimgs[key] = frame.shape[0]

            nimgs = list(id2nimgs.values())
            nimgs = np.array(nimgs)
            nimgs = nimgs.sum() / nimgs
            id2wei = {ind: wei for ind, wei in enumerate(nimgs)}
            weis = [id2wei[id_] for id_ in np.array(df.iloc[:, 1])]
            weis = np.asarray(weis)

            self.dataset = DatasetCasia(conf.use_data_folder, )
            self.loader = DataLoader(self.dataset, batch_size=conf.batch_size,
                                     num_workers=conf.num_workers,
                                     # sampler=torch.utils.data.sampler.WeightedRandomSampler(weis, weis.shape[0]),
                                     shuffle=True,
                                     drop_last=True, pin_memory=True, )
            self.class_num = conf.num_clss = self.dataset.num_classes
            conf.explored = np.zeros(self.class_num, dtype=int)
            conf.dop = np.ones(self.class_num, dtype=int) * conf.mining_init
        else:
            self.dataset = TorchDataset(conf.use_data_folder)
            self.loader = DataLoader(
                self.dataset, batch_size=conf.batch_size,
                num_workers=conf.num_workers,
                # sampler=RandomIdSampler(self.dataset.imgidx,
                #                         self.dataset.ids, self.dataset.id2range),
                shuffle=True,
                drop_last=True, pin_memory=True,
                collate_fn=torch.utils.data.dataloader.default_collate if not conf.fast_load else fast_collate
            )
            self.class_num = self.dataset.num_classes
        logging.info(f'{self.class_num} classes, load ok ')
        if conf.need_log:
            if torch.distributed.is_initialized():
                lz.set_file_logger(str(conf.log_path) + f'/proc{torch.distributed.get_rank()}')
                lz.set_file_logger_prt(str(conf.log_path) + f'/proc{torch.distributed.get_rank()}')
                lz.mkdir_p(conf.log_path, delete=False)
            else:
                lz.mkdir_p(conf.log_path, delete=False)
                lz.set_file_logger(str(conf.log_path))
                lz.set_file_logger_prt(str(conf.log_path))
                # todo why no log?
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.writer = SummaryWriter(str(conf.log_path))
        else:
            self.writer = None
        conf.writer = self.writer
        self.step = 0

        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size)
            self.model2 = MobileFaceNet(conf.embedding_size)
            logging.info('MobileFaceNet model generated')
        elif conf.net_mode == 'sglpth':
            self.model = models.singlepath()
        elif conf.net_mode == 'mbv3':
            self.model = models.mobilenetv3()
        elif conf.net_mode == 'nasnetamobile':
            self.model = models.nasnetamobile(512)
        elif conf.net_mode == 'resnext':
            self.model = models.ResNeXt(**models.resnext._NETS[str(conf.net_depth)])
        elif conf.net_mode == 'csmobilefacenet':
            self.model = CSMobileFaceNet(conf.embedding_size)
            logging.info('CSMobileFaceNet model generated')
        elif conf.net_mode == 'densenet':
            self.model = models.DenseNet(**models.densenet._NETS[str(conf.net_depth)])
        elif conf.net_mode == 'widerresnet':
            self.model = models.WiderResNet(**models.wider_resnet._NETS[str(conf.net_depth)])
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            self.model2 = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            logging.info('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        elif conf.net_mode == 'effnet':
            name = conf.eff_name
            self.model = models.EfficientNet.from_name(name)
            imgsize = models.EfficientNet.get_image_size(name)
            assert conf.input_size == imgsize, imgsize
            self.model2 = models.EfficientNet.from_name(name)
        else:
            raise ValueError(conf.net_mode)
        if conf.loss == 'arcface':
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num)
            self.head2 = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'softmax':
            self.head = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num)
            self.head2 = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'arcfaceneg':
            from models.model import ArcfaceNeg
            self.head = ArcfaceNeg(embedding_size=conf.embedding_size, classnum=self.class_num)
        else:
            raise ValueError(f'{conf.loss}')

        self.model.to(conf.model1_dev[0])
        self.model2.to(conf.model2_dev[0])

        if self.head is not None:
            self.head = self.head.to(device=conf.model1_dev[0])
            self.head2 = self.head2.to(device=conf.model2_dev[0])
            self.head.update_mrg()
            self.head2.update_mrg()

        if conf.tri_wei != 0:
            self.head_triplet = TripletLoss().cuda()
        logging.info(' model heads generated')

        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
        paras_only_bn2, paras_wo_bn2 = separate_bn_paras(self.model2)
        if conf.use_opt == 'sgd':
            ## wdecay
            # self.optimizer = optim.SGD([
            #     {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
            #     {'params': [paras_wo_bn[-1]] + [*self.head.parameters()], 'weight_decay': 4e-4},
            #     {'params': paras_only_bn}], lr=conf.lr, momentum=conf.momentum)

            ## normal
            # self.optimizer = optim.SGD([
            #     {'params': paras_wo_bn + [*self.head.parameters()],
            #      'weight_decay': conf.weight_decay},
            #     {'params': paras_only_bn},
            # ], lr=conf.lr, momentum=conf.momentum)
            # self.optimizer2 = optim.SGD([
            #     {'params': paras_wo_bn2 + [*self.head2.parameters()],
            #      'weight_decay': conf.weight_decay},
            #     {'params': paras_only_bn2},
            # ], lr=conf.lr, momentum=conf.momentum)

            ## fastfc truly
            self.optimizer = optim.SGD([
                {'params': paras_wo_bn, 'weight_decay': conf.weight_decay},
                {'params': [*self.head.parameters()], 'weight_decay': conf.weight_decay, 'lr_mult': 10},
                {'params': paras_only_bn, },
            ], lr=conf.lr, momentum=conf.momentum, )
            self.optimizer2 = optim.SGD([
                {'params': paras_wo_bn2, 'weight_decay': conf.weight_decay},
                {'params': [*self.head2.parameters()], 'weight_decay': conf.weight_decay, 'lr_mult': 10},
                {'params': paras_only_bn2},
            ], lr=conf.lr, momentum=conf.momentum)
        else:
            raise ValueError(f'{conf.use_opt}')
        if conf.fp16:
            keep_batchnorm_fp32 = True if conf.opt_level == 'O3' else None
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=conf.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32)
            self.model2, self.optimizer2 = amp.initialize(
                self.model2, self.optimizer2, opt_level=conf.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32)

        logging.info(f'optimizers generated {self.optimizer}')

        self.model = torch.nn.DataParallel(self.model,
                                           device_ids=conf.model1_dev,
                                           output_device=conf.model1_dev[0]
                                           )
        self.model2 = torch.nn.DataParallel(self.model2,
                                            device_ids=conf.model2_dev,
                                            output_device=conf.model2_dev[0]
                                            )

        self.board_loss_every = conf.board_loss_every
        self.head.train()
        self.model.train()
        self.head2.train()
        self.model2.train()

    def schedule_lr_mstep(self, e=0, step=0):
        from bisect import bisect_right
        e2lr = {epoch: conf.lr * conf.lr_gamma ** bisect_right(self.milestones, epoch) for epoch in
                range(conf.epochs)}
        logging.info(f'map e to lr is {e2lr}')
        lr = e2lr[e]
        for params in self.optimizer.param_groups:
            if 'lr_mult' in params:
                params['lr'] = lr * params['lr_mult']
            else:
                params['lr'] = lr
        for params in self.optimizer2.param_groups:
            if 'lr_mult' in params:
                params['lr'] = lr * params['lr_mult']
            else:
                params['lr'] = lr
        logging.info(f'lr is {lr}')

    def schedule_lr_cosanl(self, e=0, step=0):
        assert e == 0, 'not implement yet'
        steps_per_epoch = len(self.loader)
        ttl_steps = conf.epochs * steps_per_epoch
        base_lr = conf.lr
        # min_lr = base_lr/10 # defalt use 0
        lr = 1 / 2 * (base_lr) * (1 + np.cos(step / ttl_steps * np.pi))
        for params in self.optimizer.param_groups:
            if 'lr_mult' in params:
                params['lr'] = lr * params['lr_mult']
            else:
                params['lr'] = lr

        for params in self.optimizer2.param_groups:
            if 'lr_mult' in params:
                params['lr'] = lr * params['lr_mult']
            else:
                params['lr'] = lr

    # init_lr = schedule_lr = schedule_lr_mstep
    schedule_lr = init_lr = schedule_lr_cosanl

    def evaluate(self, conf, path, name, nrof_folds=10, tta=True, ensemble=False):
        # from utils import ccrop_batch
        self.model.eval()
        self.model2.eval()
        idx = 0
        if name in self.val_loader_cache:
            carray, issame = self.val_loader_cache[name]
        else:
            from data.data_pipe import get_val_pair
            carray, issame = get_val_pair(path, name)
            carray = carray[:, ::-1, :, :].copy()  # BGR 2 RGB!
            self.val_loader_cache[name] = carray, issame
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    # batch = ccrop_batch(batch)
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    if ensemble:
                        emb_batch2 = self.model2(batch.cuda()) + self.model2(fliped.cuda())
                        emb_batch = emb_batch.cpu() + emb_batch2.cpu()
                    else:
                        emb_batch = emb_batch.cpu()
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.cuda()).cpu()
                    # todo ...
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    # batch = ccrop_batch(batch)
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    if ensemble:
                        emb_batch2 = self.model2(batch.cuda()) + self.model2(fliped.cuda())
                        emb_batch = emb_batch.cpu() + emb_batch2.cpu()
                    else:
                        emb_batch = emb_batch.cpu()
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.cuda()).cpu()
                    # todo ...
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        roc_curve_tensor = None
        # buf = gen_plot(fpr, tpr)
        # roc_curve = Image.open(buf)
        # roc_curve_tensor = trans.ToTensor()(roc_curve)
        self.model.train()
        self.model2.train()
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    evaluate_accelerate = evaluate

    def validate_ori(self, *args):
        res = {}
        self.model.eval()
        for ds in ['cfp_fp']:  # ['lfw', 'agedb_30', 'cfp_fp', 'cfp_ff', 'calfw', 'cplfw', 'vgg2_fp', ]
            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.loader.dataset.root_path,
                                                                       ds)
            logging.info(f'validation accuracy on {ds} is {accuracy} ')
            res[ds] = accuracy

        self.model.train()
        return res

    def train_cotching(self, conf, epochs):
        self.model.train()
        self.model2.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        if conf.start_eval:
            for ds in ['cfp_fp', ]:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                logging.info(f'validation accuracy on {ds} is {accuracy} ')
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                try:
                    ind_data, data = next(loader_enum)
                except StopIteration as err:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].to(device=conf.model1_dev[0])
                # imgs.requires_grad_(True)
                imgs2 = imgs
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].to(device=conf.model1_dev[0])
                labels2 = labels
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )

                embeddings = self.model(imgs, mode='train')
                embeddings2 = self.model2(imgs2, mode='train')
                thetas = self.head(embeddings, labels)
                thetas2 = self.head2(embeddings2, labels2)
                pred = thetas.argmax(dim=1)
                pred2 = thetas2.argmax(dim=1)
                disagree = pred != pred2
                if disagree.sum().item() == 0:
                    logging.info(f'disagree is zero!')
                    disagree = to_torch(np.random.randint(0, 1, disagree.shape)).type_as(disagree)  # todo
                loss_xent = F.cross_entropy(thetas[disagree], labels[disagree], reduction='none')
                loss_xent2 = F.cross_entropy(thetas2[disagree], labels2[disagree], reduction='none')
                ind_sorted = loss_xent.argsort()
                ind2_sorted = loss_xent2.argsort()
                num_disagree = labels[disagree].shape[0]
                assert num_disagree == disagree.sum().item()
                tau = conf.tau
                Ek = len(loader)
                Emax = len(loader) * conf.epochs
                lambda_e = 1 - min(self.step / Ek * tau, (1 + (self.step - Ek) / (Emax - Ek)) * tau)
                num_remember = max(int(round(num_disagree * lambda_e)), 1)
                ind_update = ind_sorted[:num_remember]
                ind2_update = ind2_sorted[:num_remember]
                loss_xent = loss_xent[ind2_update].mean()
                loss_xent2 = loss_xent2[ind_update].mean()

                self.optimizer.zero_grad()
                if conf.fp16:
                    with amp.scale_loss(loss_xent, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                self.optimizer.step()

                self.optimizer2.zero_grad()
                if conf.fp16:
                    with amp.scale_loss(loss_xent2, self.optimizer2) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent2.backward()
                self.optimizer2.step()

                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                    acc2 = ((thetas.argmax(dim=1) == labels).sum().item() + 0.0) / labels.shape[0]
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'xent2: {loss_xent2.item():.2e} ' +
                                 f'dsgr: {disagree.sum().item()} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/disagree', disagree.sum().item(), self.step)
                    writer.add_scalar('info/remenber', num_remember, self.step)
                    writer.add_scalar('info/lambda_e', lambda_e, self.step)
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('loss/xent2', loss_xent2.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/acc2', acc2, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        # if conf.local_rank is not None and conf.local_rank != 0:
        #     return
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        time_now = get_time()
        lz.mkdir_p(save_path, delete=False)
        # self.model.cpu()
        torch.save(
            self.model.module.state_dict(),
            save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                          extra)))
        torch.save(
            self.model2.module.state_dict(),
            save_path /
            ('model2_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                           extra)))
        # self.model.cuda()
        lz.msgpack_dump({'dop': conf.dop,
                         'id2range_dop': conf.id2range_dop,
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

                self.head2.cpu()
                torch.save(
                    self.head2.state_dict(),
                    save_path /
                    ('head2_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                                  extra)))
                self.head2.cuda()

            torch.save(
                self.optimizer.state_dict(),
                save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy,
                                                                  self.step, extra)))
            torch.save(
                self.optimizer2.state_dict(),
                save_path /
                ('optimizer2_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy,
                                                                   self.step, extra)))

    def load_state(self, fixed_str='',
                   resume_path=None, latest=True,
                   load_optimizer=False, load_imp=False, load_head=False,
                   load_model2=False,
                   ):
        from pathlib import Path
        save_path = Path(resume_path)
        modelp = save_path / '{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:  # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            assert step.shape[0] > 0, f"{resume_path} chk!"
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '').replace('model2_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        logging.info(f'you are using gpu, load model, {modelp}')
        model_state_dict = torch.load(modelp)
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if load_model2:
            model_state_dict2 = torch.load(str(modelp).replace('model_', 'model2_'))
            model_state_dict2 = {k: v for k, v in model_state_dict2.items() if 'num_batches_tracked' not in k}
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict, strict=True)
            if load_model2:
                self.model2.load_state_dict(model_state_dict2, strict=True)
            else:
                self.model2.load_state_dict(model_state_dict, strict=True)
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True)
            if load_model2:
                self.model2.module.load_state_dict(model_state_dict2, strict=True)
            else:
                self.model2.module.load_state_dict(model_state_dict, strict=True)

        if load_head:
            assert osp.exists(save_path / 'head_{}'.format(fixed_str))
            logging.info(f'load head from {modelp}')
            head_state_dict = torch.load(save_path / 'head_{}'.format(fixed_str))
            self.head.load_state_dict(head_state_dict)
            if load_model2:
                head_state_dict2 = torch.load(save_path / 'head2_{}'.format(fixed_str))
                self.head2.load_state_dict(head_state_dict2)
            else:
                self.head2.load_state_dict(head_state_dict)
        if load_optimizer:
            logging.info(f'load opt from {modelp}')
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
            if load_model2:
                self.optimizer2.load_state_dict(torch.load(save_path / 'optimizer2_{}'.format(fixed_str)))
            else:
                self.optimizer2.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
        if load_imp and (save_path / f'extra_{fixed_str.replace(".pth", ".pk")}').exists():
            extra = lz.msgpack_load(save_path / f'extra_{fixed_str.replace(".pth", ".pk")}')
            conf.dop = extra['dop'].copy()
            conf.id2range_dop = extra['id2range_dop'].copy()

    def train_cotching_accbs(self, conf, epochs):
        self.model.train()
        self.model2.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        imgs_l = []
        labels_l = []
        imgs2_l = []
        labels2_l = []
        accuracy = 0
        step_det = 0
        if conf.start_eval:
            for ds in ['cfp_fp', ]:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                logging.info(f'validation accuracy on {ds} is {accuracy} ')
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                try:
                    ind_data, data = next(loader_enum)
                except StopIteration as err:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                if not imgs_l or sum([imgs.shape[0] for imgs in imgs_l]) < conf.batch_size:
                    imgs = data['imgs'].to(device=conf.model1_dev[0])
                    assert imgs.max() < 2
                    labels = data['labels'].to(device=conf.model1_dev[0])
                    data_time.update(
                        lz.timer.since_last_check(verbose=False)
                    )
                    with torch.no_grad():
                        embeddings = self.model(imgs, )
                        embeddings2 = self.model2(imgs, )
                        thetas = self.head(embeddings, labels)
                        thetas2 = self.head2(embeddings2, labels)
                        pred = thetas.argmax(dim=1)
                        pred2 = thetas2.argmax(dim=1)
                        disagree = pred != pred2
                        if disagree.sum().item() == 0:
                            continue  # this assert acc can finally reach bs
                        loss_xent = F.cross_entropy(thetas[disagree], labels[disagree], reduction='none')
                        loss_xent2 = F.cross_entropy(thetas2[disagree], labels[disagree], reduction='none')
                        ind_sorted = loss_xent.argsort()
                        ind2_sorted = loss_xent2.argsort()
                        num_disagree = labels[disagree].shape[0]
                        tau = conf.tau
                        Ek = len(loader)
                        Emax = len(loader) * conf.epochs
                        lambda_e = 1 - min(self.step / Ek * tau, (1 + (self.step - Ek) / (Emax - Ek)) * tau)
                        num_remember = max(int(round(num_disagree * lambda_e)), 1)
                        ind_update = ind_sorted[:num_remember]
                        ind2_update = ind2_sorted[:num_remember]
                        imgs_l.append(imgs[disagree][ind_update].cpu())
                        ## original
                        # labels_l.append(labels[ind_update].cpu())
                        # imgs2_l.append(imgs[ind2_update].cpu())
                        # labels2_l.append(labels[ind2_update].cpu())
                        ##after
                        labels_l.append(labels[disagree][ind_update].cpu())
                        imgs2_l.append(imgs[disagree][ind2_update].cpu())
                        labels2_l.append(labels[disagree][ind2_update].cpu())

                    continue
                else:
                    imgs_new = torch.cat(imgs_l, dim=0)
                    imgs = imgs_new[:conf.batch_size].to(device=conf.model1_dev[0])
                    labels_new = torch.cat(labels_l, dim=0)
                    labels = labels_new[:conf.batch_size].to(device=conf.model1_dev[0])
                    # imgs_l = [imgs_new[conf.batch_size:]]  # whether this right
                    # labels_l = [labels_new[conf.batch_size:]]
                    imgs_l = []
                    labels_l = []
                    labels_cpu = labels.cpu()
                    embeddings2 = self.model2(imgs, )  # model1 --> imgs1 --> model2
                    thetas2 = self.head2(embeddings2, labels)
                    loss_xent2 = F.cross_entropy(thetas2, labels)
                    self.optimizer2.zero_grad()
                    if conf.fp16:
                        with amp.scale_loss(loss_xent2, self.optimizer2) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_xent2.backward()
                    self.optimizer2.step()

                    imgs2_new = torch.cat(imgs2_l, dim=0)
                    imgs2 = imgs2_new[:conf.batch_size].to(device=conf.model1_dev[0])
                    labels2_new = torch.cat(labels2_l, dim=0)
                    labels2 = labels2_new[:conf.batch_size].to(device=conf.model1_dev[0])
                    # imgs2_l = [imgs2_new[conf.batch_size:]]
                    # labels2_l = [labels2_new[conf.batch_size:]]
                    # logging.info(f'{imgs2_new.shape[0]} {step_det}')
                    step_det += imgs2_new[conf.batch_size:].shape[0] / conf.batch_size
                    if step_det > 1:
                        self.step -= 1
                        step_det -= 1
                    imgs2_l = []
                    labels2_l = []

                    embeddings = self.model(imgs2, )
                    thetas = self.head(embeddings, labels2)
                    loss_xent = F.cross_entropy(thetas, labels2)
                    self.optimizer.zero_grad()
                    if conf.fp16:
                        with amp.scale_loss(loss_xent, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_xent.backward()
                    self.optimizer.step()

                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/disagree', disagree.sum().item(), self.step)
                    writer.add_scalar('info/disagree_ratio', num_disagree / conf.batch_size, self.step)
                    writer.add_scalar('info/remenber', num_remember, self.step)
                    writer.add_scalar('info/remenber_ratio', num_remember / conf.batch_size, self.step)
                    writer.add_scalar('info/lambda_e', lambda_e, self.step)
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('loss/xent2', loss_xent2.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def get_loader_enum(self, loader=None):
        import gc
        loader = loader or self.loader
        succ = False
        while not succ:
            try:
                loader_enum = data_prefetcher(enumerate(loader))
                succ = True
            except Exception as e:
                try:
                    del loader_enum
                except:
                    pass
                gc.collect()
                logging.info(f'err is {e}')
                time.sleep(10)
        return loader_enum

    def train_mual(self, conf, epochs):
        self.model.train()
        self.model2.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        target_bs = conf.batch_size * conf.acc_grad
        now_bs = 0
        accuracy = 0
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        if conf.start_eval:
            for ds in ['cfp_fp', ]:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                logging.info(f'validation accuracy on {ds} is {accuracy} ')
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            # self.schedule_lr(e)
            loader_enum = self.get_loader_enum()
            while True:
                self.schedule_lr(step=self.step)
                try:
                    ind_data, data = next(loader_enum)
                except StopIteration as err:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = self.get_loader_enum()
                    ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = self.get_loader_enum()
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                if now_bs >= target_bs:
                    now_bs = 0
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.optimizer2.step()
                    self.optimizer2.zero_grad()
                imgs = data['imgs'].to(device=conf.model1_dev[0])
                imgs2 = data['imgs'].to(device=conf.model2_dev[0])
                assert imgs.max() < 2
                labels = data['labels'].to(conf.model1_dev[0])
                labels2 = data['labels'].to(conf.model2_dev[0])
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                embeddings = self.model(imgs)
                embeddings2 = self.model2(imgs2)
                thetas = self.head(embeddings, labels)
                thetas2 = self.head2(embeddings2, labels2)

                loss_xent = F.cross_entropy(thetas, labels, )
                loss_xent2 = F.cross_entropy(thetas2, labels2)

                if conf.mutual_learning:
                    mual1 = F.kl_div(F.log_softmax(thetas, dim=1),
                                     F.softmax(thetas2.detach().to(conf.model1_dev[0]), dim=1),
                                     reduction='batchmean',
                                     )
                    loss_xent = loss_xent + mual1 * conf.mutual_learning
                    mual2 = F.kl_div(F.log_softmax(thetas2, dim=1),
                                     F.softmax(thetas.detach().to(conf.model2_dev[0]), dim=1),
                                     reduction='batchmean',
                                     )  # todo temparature?
                    loss_xent2 = loss_xent2 + mual2 * conf.mutual_learning
                if conf.fp16:
                    with amp.scale_loss(loss_xent2 / conf.acc_grad, self.optimizer2) as scaled_loss:
                        scaled_loss.backward()
                else:
                    (loss_xent2 / conf.acc_grad).backward()
                if conf.fp16:
                    with amp.scale_loss(loss_xent / conf.acc_grad, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    (loss_xent / conf.acc_grad).backward()
                # lz.clip_grad_value_(self.model.parameters(), )
                # lz.clip_grad_value_(self.model2.parameters(), )

                now_bs += conf.batch_size

                with torch.no_grad():
                    labels_cpu = labels.cpu()
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('loss/xent2', loss_xent2.item(), self.step)
                    if conf.mutual_learning:
                        writer.add_scalar('loss/mual1', mual1.item(), self.step)
                        writer.add_scalar('loss/mual2', mual2.item(), self.step)

                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >=len(loader)//2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def train_cotching_accbs_v2(self, conf, epochs):
        self.model.train()
        self.model2.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        target_bs = conf.batch_size * conf.acc_grad
        now_bs = 0
        accuracy = 0
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        if conf.start_eval:
            for ds in ['cfp_fp', ]:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                logging.info(f'validation accuracy on {ds} is {accuracy} ')
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            # self.schedule_lr(e)
            loader_enum = self.get_loader_enum()
            while True:
                self.schedule_lr(step=self.step)
                try:
                    ind_data, data = next(loader_enum)
                except StopIteration as err:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = self.get_loader_enum()
                    ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = self.get_loader_enum()
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                if now_bs >= target_bs:
                    now_bs = 0
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.optimizer2.step()
                    self.optimizer2.zero_grad()
                imgs = data['imgs'].to(device=conf.model1_dev[0])
                assert imgs.max() < 2
                labels = data['labels'].to(device=conf.model1_dev[0])
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                embeddings = self.model(imgs, )
                embeddings2 = self.model2(imgs, )
                thetas = self.head(embeddings, labels)
                thetas2 = self.head2(embeddings2, labels)
                pred = thetas.argmax(dim=1)
                pred2 = thetas2.argmax(dim=1)
                # disagree = pred != pred2
                disagree = (pred != pred2) | ((pred == pred2) & (pred != labels))
                num_disagree = disagree.sum().item()

                if num_disagree == 0:
                    continue  # this assert acc can finally reach bs
                loss_xent = F.cross_entropy(thetas[disagree], labels[disagree], reduction='none')
                loss_xent2 = F.cross_entropy(thetas2[disagree], labels[disagree], reduction='none')
                ind_sorted = loss_xent.argsort()
                ind2_sorted = loss_xent2.argsort()
                tau = conf.tau
                Ek = len(loader)  # todo
                Emax = len(loader) * conf.epochs
                lambda_e = 1 - min(self.step / Ek * tau, (1 + (self.step - Ek) / (Emax - Ek)) * tau)
                num_remember = max(int(round(num_disagree * lambda_e)), 1)
                ind_update = ind_sorted[:num_remember]
                ind2_update = ind2_sorted[:num_remember]
                loss_xent_rmbr = loss_xent[ind2_update].mean()  # this is where exchange
                loss_xent2_rmbr = loss_xent2[ind_update].mean()

                if conf.mutual_learning:
                    mual1 = F.kl_div(F.log_softmax(thetas[disagree], dim=1),
                                     F.softmax(thetas2[disagree].detach(), dim=1),
                                     reduction='batchmean',
                                     )  # todo may [ind2_update]
                    loss_xent_rmbr += mual1 * conf.mutual_learning
                    mual2 = F.kl_div(F.log_softmax(thetas2[disagree], dim=1),
                                     F.softmax(thetas[disagree].detach(), dim=1),
                                     reduction='batchmean',
                                     )  # todo batchmean or mean
                    loss_xent2_rmbr += mual2 * conf.mutual_learning
                rmbr_rat = num_remember / conf.batch_size
                if conf.fp16:
                    with amp.scale_loss(loss_xent2_rmbr * rmbr_rat, self.optimizer2) as scaled_loss:
                        scaled_loss.backward()
                else:
                    (loss_xent2_rmbr * rmbr_rat).backward()
                if conf.fp16:
                    with amp.scale_loss(loss_xent_rmbr * rmbr_rat, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    (loss_xent_rmbr * rmbr_rat).backward()
                now_bs += num_remember

                with torch.no_grad():
                    labels_cpu = labels.cpu()
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent_rmbr.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/disagree', num_disagree, self.step)
                    writer.add_scalar('info/disagree_ratio', num_disagree / conf.batch_size, self.step)
                    writer.add_scalar('info/remenber', num_remember, self.step)
                    writer.add_scalar('info/remenber_ratio', num_remember / conf.batch_size, self.step)
                    writer.add_scalar('info/lambda_e', lambda_e, self.step)
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent_rmbr.item(), self.step)
                    writer.add_scalar('loss/xent2', loss_xent2_rmbr.item(), self.step)
                    if conf.mutual_learning:
                        writer.add_scalar('loss/mual1', mual1.item(), self.step)
                        writer.add_scalar('loss/mual2', mual2.item(), self.step)

                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')


class face_cotching_head(face_learner):
    def __init__(self, conf=conf, ):
        self.milestones = conf.milestones
        self.val_loader_cache = {}
        ## torch reader
        if conf.dataset_name == 'webface' or conf.dataset_name == 'casia':
            file = '/data2/share/casia_landmark.txt'
            df = pd.read_csv(file, sep='\t', header=None)
            id2nimgs = {}
            for key, frame in df.groupby(1):
                id2nimgs[key] = frame.shape[0]

            nimgs = list(id2nimgs.values())
            nimgs = np.array(nimgs)
            nimgs = nimgs.sum() / nimgs
            id2wei = {ind: wei for ind, wei in enumerate(nimgs)}
            weis = [id2wei[id_] for id_ in np.array(df.iloc[:, 1])]
            weis = np.asarray(weis)

            self.dataset = DatasetCasia(conf.use_data_folder, )
            self.loader = DataLoader(self.dataset, batch_size=conf.batch_size,
                                     num_workers=conf.num_workers,
                                     sampler=torch.utils.data.sampler.WeightedRandomSampler(weis, weis.shape[0]),
                                     drop_last=True, pin_memory=True, )
            self.class_num = conf.num_clss = self.dataset.num_classes
            conf.explored = np.zeros(self.class_num, dtype=int)
            conf.dop = np.ones(self.class_num, dtype=int) * conf.mining_init
        else:
            self.dataset = TorchDataset(conf.use_data_folder)
            self.loader = DataLoader(
                self.dataset, batch_size=conf.batch_size,
                num_workers=conf.num_workers,
                sampler=RandomIdSampler(self.dataset.imgidx,
                                        self.dataset.ids, self.dataset.id2range),
                drop_last=True, pin_memory=True,
                collate_fn=torch.utils.data.dataloader.default_collate if not conf.fast_load else fast_collate
            )
            self.class_num = self.dataset.num_classes
        logging.info(f'{self.class_num} classes, load ok ')
        if conf.need_log:
            if torch.distributed.is_initialized():
                lz.set_file_logger(str(conf.log_path) + f'/proc{torch.distributed.get_rank()}')
                lz.set_file_logger_prt(str(conf.log_path) + f'/proc{torch.distributed.get_rank()}')
                lz.mkdir_p(conf.log_path, delete=False)
            else:
                lz.mkdir_p(conf.log_path, delete=False)
                lz.set_file_logger(str(conf.log_path))
                lz.set_file_logger_prt(str(conf.log_path))
                # todo why no log?
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.writer = SummaryWriter(str(conf.log_path))
        else:
            self.writer = None
        self.step = 0

        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size)
            logging.info('MobileFaceNet model generated')
        elif conf.net_mode == 'nasnetamobile':
            self.model = models.nasnetamobile(512)
        elif conf.net_mode == 'resnext':
            self.model = models.ResNeXt(**models.resnext._NETS[str(conf.net_depth)])
        elif conf.net_mode == 'csmobilefacenet':
            self.model = CSMobileFaceNet(conf.embedding_size)
            logging.info('CSMobileFaceNet model generated')
        elif conf.net_mode == 'densenet':
            self.model = models.DenseNet(**models.densenet._NETS[str(conf.net_depth)])
        elif conf.net_mode == 'widerresnet':
            self.model = models.WiderResNet(**models.wider_resnet._NETS[str(conf.net_depth)])
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            logging.info('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        else:
            raise ValueError(conf.net_mode)
        if conf.loss == 'arcface':
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num)
            self.head2 = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'softmax':
            self.head = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num)
            self.head2 = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'arcfaceneg':
            from models.model import ArcfaceNeg
            self.head = ArcfaceNeg(embedding_size=conf.embedding_size, classnum=self.class_num)
        else:
            raise ValueError(f'{conf.loss}')

        self.model.cuda()

        if self.head is not None:
            self.head = self.head.to(device=conf.model2_dev[0])
            self.head2 = self.head2.to(device=conf.model2_dev[0])

        if conf.tri_wei != 0:
            self.head_triplet = TripletLoss().cuda()
        logging.info(' model heads generated')

        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
        if conf.use_opt == 'adam':  # todo deprecated
            self.optimizer = optim.Adam([{'params': paras_wo_bn + [*self.head.parameters()], 'weight_decay': 0},
                                         {'params': paras_only_bn}, ],
                                        betas=(conf.adam_betas1, conf.adam_betas2),
                                        amsgrad=True,
                                        lr=conf.lr,
                                        )
        elif conf.net_mode == 'mobilefacenet' or conf.net_mode == 'csmobilefacenet':
            if conf.use_opt == 'sgd':
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [*self.head.parameters()], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}], lr=conf.lr, momentum=conf.momentum)
                # embed()
            elif conf.use_opt == 'adabound':
                from tools.adabound import AdaBound
                self.optimizer = AdaBound([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [*self.head.parameters()], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, betas=(conf.adam_betas1, conf.adam_betas2),
                    gamma=1e-3, final_lr=conf.final_lr, )
        elif conf.use_opt == 'sgd':
            self.optimizer = optim.SGD([
                {'params': paras_wo_bn + [*self.head.parameters()] + [*self.head2.parameters()],
                 'weight_decay': conf.weight_decay},
                {'params': paras_only_bn},
            ], lr=conf.lr, momentum=conf.momentum)
        elif conf.use_opt == 'adabound':
            from tools.adabound import AdaBound
            self.optimizer = AdaBound([
                {'params': paras_wo_bn + [*self.head.parameters()],
                 'weight_decay': conf.weight_decay},
                {'params': paras_only_bn},
            ], lr=conf.lr, betas=(conf.adam_betas1, conf.adam_betas2),
                gamma=1e-3, final_lr=conf.final_lr,
            )
        else:
            raise ValueError(f'{conf.use_opt}')
        if conf.fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        logging.info(f'optimizers generated {self.optimizer}')

        self.model = torch.nn.DataParallel(self.model,
                                           device_ids=conf.model1_dev,
                                           output_device=conf.model1_dev[0]
                                           ).cuda()

        self.board_loss_every = conf.board_loss_every
        self.head.train()
        self.model.train()
        self.head2.train()

    def schedule_lr(self, e=0):
        from bisect import bisect_right

        e2lr = {epoch: conf.lr * conf.lr_gamma ** bisect_right(self.milestones, epoch) for epoch in
                range(conf.epochs)}
        logging.info(f'map e to lr is {e2lr}')
        lr = e2lr[e]
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        logging.info(f'lr is {lr}')

    init_lr = schedule_lr

    def train_cotching(self, conf, epochs):
        self.model.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        if conf.start_eval:
            for ds in ['cfp_fp', ]:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                logging.info(f'validation accuracy on {ds} is {accuracy} ')
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                try:
                    ind_data, data = next(loader_enum)
                except StopIteration as err:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].to(device=conf.model1_dev[0])
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].to(device=conf.model1_dev[0])
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )

                embeddings = self.model(imgs, mode='train')
                embeddings = rescale(embeddings)
                thetas = self.head(embeddings, labels)
                thetas2 = self.head2(embeddings, labels)
                pred = thetas.argmax(dim=1)
                pred2 = thetas2.argmax(dim=1)
                disagree = pred != pred2
                if disagree.sum().item() == 0:
                    logging.info(f'disagree is zero!')
                    disagree = to_torch(np.random.randint(0, 1, disagree.shape)).type_as(disagree)  # todo
                loss_xent = F.cross_entropy(thetas[disagree], labels[disagree], reduction='none')
                loss_xent2 = F.cross_entropy(thetas2[disagree], labels[disagree], reduction='none')
                ind_sorted = loss_xent.argsort()
                ind2_sorted = loss_xent2.argsort()
                num_disagree = labels[disagree].shape[0]
                assert num_disagree == disagree.sum().item()
                tau = conf.tau
                Ek = len(loader)
                Emax = len(loader) * conf.epochs
                lambda_e = 1 - min(self.step / Ek * tau, (1 + (self.step - Ek) / (Emax - Ek)) * tau)
                num_remember = max(int(round(num_disagree * lambda_e)), 1)
                ind_update = ind_sorted[:num_remember]
                ind2_update = ind2_sorted[:num_remember]
                loss_xent = loss_xent[ind2_update].mean()
                loss_xent2 = loss_xent2[ind_update].mean()

                loss_xent = loss_xent + loss_xent2

                self.optimizer.zero_grad()
                if conf.fp16:
                    with amp.scale_loss(loss_xent, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                self.optimizer.step()

                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                    acc2 = ((thetas.argmax(dim=1) == labels).sum().item() + 0.0) / labels.shape[0]
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'xent2: {loss_xent2.item():.2e} ' +
                                 f'dsgr: {disagree.sum().item()} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/disagree', disagree.sum().item(), self.step)
                    writer.add_scalar('info/remenber', num_remember, self.step)
                    writer.add_scalar('info/lambda_e', lambda_e, self.step)
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('loss/xent2', loss_xent2.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/acc2', acc2, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step >= len(loader) // 2:
                    break
            if conf.prof and e > 1:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        # if conf.local_rank is not None and conf.local_rank != 0:
        #     return
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        time_now = get_time()
        lz.mkdir_p(save_path, delete=False)
        # self.model.cpu()
        torch.save(
            self.model.module.state_dict(),
            save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                          extra)))
        # self.model.cuda()
        lz.msgpack_dump({'dop': conf.dop,
                         'id2range_dop': conf.id2range_dop,
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

                self.head2.cpu()
                torch.save(
                    self.head2.state_dict(),
                    save_path /
                    ('head2_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                                  extra)))
                self.head2.cuda()

            torch.save(
                self.optimizer.state_dict(),
                save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy,
                                                                  self.step, extra)))

    def load_state(self, fixed_str='',
                   resume_path=None, latest=True,
                   load_optimizer=False, load_imp=False, load_head=False,
                   load_model2=False,
                   ):
        from pathlib import Path
        save_path = Path(resume_path)
        modelp = save_path / '{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:  # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            assert step.shape[0] > 0, f"{resume_path} chk!"
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        logging.info(f'you are using gpu, load model, {modelp}')
        model_state_dict = torch.load(modelp)
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict, strict=True)
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True)

        if load_head:
            assert osp.exists(save_path / 'head_{}'.format(fixed_str))
            logging.info(f'load head from {modelp}')
            head_state_dict = torch.load(save_path / 'head_{}'.format(fixed_str))
            self.head.load_state_dict(head_state_dict)
            if load_model2:
                head_state_dict2 = torch.load(save_path / 'head2_{}'.format(fixed_str))
                self.head2.load_state_dict(head_state_dict2)
            else:
                self.head2.load_state_dict(head_state_dict)
        if load_optimizer:
            logging.info(f'load opt from {modelp}')
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
        if load_imp and (save_path / f'extra_{fixed_str.replace(".pth", ".pk")}').exists():
            extra = lz.msgpack_load(save_path / f'extra_{fixed_str.replace(".pth", ".pk")}')
            conf.dop = extra['dop'].copy()
            conf.id2range_dop = extra['id2range_dop'].copy()


class ReScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / 2  # todo 2 head here


rescale = ReScale.apply

if __name__ == '__main__':
    exit()
    # ds = DatasetCfpFp()
    # ds = DatasetCasia()
    conf.fill_cache = 0
    ds = TorchDataset(conf.use_data_folder)
    imgs = []
    for i in np.random.randint(0, 100, (10,)):
        img = ds[i]['imgs']
    #     imgs.append(img)
    # plt_imshow_tensor(imgs)
    # plt.show()

    loader = DataLoader(
        ds, batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        sampler=RandomIdSampler(ds.imgidx, ds.ids, ds.id2range),
        # shuffle=True,
        drop_last=True, pin_memory=True,
        collate_fn=torch.utils.data.dataloader.default_collate if not conf.fast_load else fast_collate
    )
    for ind, data in enumerate(loader):
        if ind >= 10: break
    class_num = ds.num_classes
    print(class_num)

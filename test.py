import lz
from lz import *

from config import get_config
from Learner import face_learner
import argparse
import mxnet as mx
import torch


def main1():
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=8, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se',
                        type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]", default='emore',
                        type=str)
    parser.set_defaults(
        epochs=8,
        net='ir_se',
        net_depth='50',
        lr=1e-3,
        batch_size=96,
        num_workers=3,
        data_mode="ms1m",
    )
    args = parser.parse_args()

    conf = get_config(training=True)

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth

    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode

    learner = face_learner(conf, inference=False, need_loader=False)
    # print(learner.find_lr(conf, ))
    # learner.train(conf, args.epochs)

    for i in range(1):
        for imgs, labels in learner.loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            print('ok', imgs.shape, labels.shape)
            embeddings = learner.model(imgs)
            thetas = learner.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)
            import torch.autograd
            # fgg g
            grad = torch.autograd.grad(loss, embeddings, retain_graph=False, create_graph=False, only_inputs=True)[
                0].detach()
            embeddings_adv = embeddings + 0.01 * grad
            thetas_adv = learner.head(embeddings_adv, labels)
            loss_adv = conf.ce_loss(thetas_adv, labels)
            loss_adv.backward()
            # fgg gg
            # grad = torch.autograd.grad(loss, embeddings, retain_graph=True, create_graph=True, only_inputs=True)[
            #     0]
            # embeddings_adv = embeddings + 0.01 * grad
            # thetas_adv = learner.head(embeddings_adv, labels)
            # loss_adv = conf.ce_loss(thetas_adv, labels)
            # loss_adv.backward()


def main2():
    fname = '/home/xinglu/work/faces_small/train'
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(fname_idx,
                                           fname_rec, 'w')
    img_files = glob.glob('/home/xinglu/work/faces_small/*/*')

    header = mx.recordio.IRHeader(0, [len(img_files), len(img_files)], 0, 0)
    s = mx.recordio.pack_img(header, 0)
    record.write_idx(0, s)

    for ind, img_file in enumerate(img_files):
        ind = ind + 1
        img = cv2.imread(img_file)
        label = img_file.split('/')[-2]
        header = mx.recordio.IRHeader(0, label, ind, 0)
        s = mx.recordio.pack_img(header, img)
        record.write_idx(ind, s)


def main3():
    from data.data_pipe import load_mx_rec
    load_mx_rec(lz.work_path + 'faces_small/')


import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio


# from config import gl_conf
def extract_ms1m_info():
    self = edict()
    path_ms1m = lz.share_path2 + 'faces_ms1m_112x112/'
    path_imgrec = lz.share_path2 + 'faces_ms1m_112x112/train.rec'
    path_imgidx = path_imgrec[0:-4] + ".idx"
    self.imgrec = recordio.MXIndexedRecordIO(
        path_imgidx, path_imgrec,
        'r')
    s = self.imgrec.read_idx(0)
    header, _ = recordio.unpack(s)
    self.header0 = (int(header.label[0]), int(header.label[1]))
    # assert(header.flag==1)
    self.imgidx = list(range(1, int(header.label[0])))
    id2range = dict()
    self.seq_identity = list(range(int(header.label[0]), int(header.label[1])))
    for identity in self.seq_identity:
        s = self.imgrec.read_idx(identity)
        header, _ = recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        id2range[(identity - 3804847)] = (a, b)
        count = b - a
    self.seq = self.imgidx
    self.seq_identity = [int(t) - 3804847 for t in self.seq_identity]
    lz.msgpack_dump([self.imgidx, self.seq_identity, id2range], path_ms1m + '/info.pk')


def load_ms1m_info():
    self = edict()
    path_ms1m = lz.share_path2 + 'faces_ms1m_112x112/'
    path_imgrec = lz.share_path2 + 'faces_ms1m_112x112/train.rec'
    path_imgidx = path_imgrec[0:-4] + ".idx"
    self.imgrec = recordio.MXIndexedRecordIO(
        path_imgidx, path_imgrec,
        'r')

    imgidx, ids, id2range = lz.msgpack_load(path_ms1m + '/info.pk')
    print(len(imgidx), len(ids), len(id2range))
    # while True:
    #     time.sleep(10)
    # for indt in range(1):
    #     id1 = ids[0]
    #     imgid = id2range[id1][0]
    #     s = self.imgrec.read_idx(imgid)
    #     header, img = recordio.unpack(s)
    #     print(header.label, id1)
    imgidx, ids = np.array(imgidx), np.array(ids)
    print(stat_np(imgidx))
    print(stat_np(ids))
    # (1, 1902423.5, 1902423.5, 3804846)
    # (0, 42581.5, 42581.5, 85163)

if __name__ == '__main__':
    # main3()
    # extract_ms1m_info()
    load_ms1m_info()

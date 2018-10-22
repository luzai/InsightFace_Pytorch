import lz
from lz import *

from config import get_config
from Learner import face_learner
import argparse
import mxnet as mx


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

    import torch

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
    try:
        import Queue as queue
    except ImportError:
        import queue
    q_out = queue.Queue()
    fname = '/home/xinglu/work/faces_small/train'
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(fname_idx,
                                           fname_rec, 'w')
    cnt = 0
    pre_time = time.time()
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


if __name__ == '__main__':
    main3()

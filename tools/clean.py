import lz
from lz import *

from config import conf
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
        img = cv2.imread(img_file)  # rec is BGR format!
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
    self.imgidx = list(range(1, int(header.label[0])))
    id2range = dict()
    self.seq_identity = list(range(int(header.label[0]), int(header.label[1])))
    self.ids = self.seq_identity
    print(f'{min(self.ids)}')
    for identity in self.seq_identity:
        s = self.imgrec.read_idx(identity - 1)
        header, _ = recordio.unpack(s)
        header.label
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
    for indt in range(1):
        id1 = ids[0]
        imgid = id2range[id1][0]
        img_info = self.imgrec.read_idx(imgid)
        header, img = recordio.unpack_img(img_info)
        print(header.label, id1)

        plt_imshow(img)  # on rec  is BGR format !
        plt.show()
    imgidx, ids = np.array(imgidx), np.array(ids)
    # (1, 1902423.5, 1902423.5, 3804846)
    # (0, 42581.5, 42581.5, 85163)


def cleanup(p=root_path + 'work_space'):
    for nowp, containp, containf in os.walk(p):
        flag = False
        for f in containf:
            if '.pth' in f and 'ir' not in f:
                flag = True
                break
        if flag:
            from pathlib import Path
            save_path = Path(nowp)
            fixed_strs = [t.name for t in save_path.glob('*_*.pth')] + [t.name for t in save_path.glob('*_*.pk')]
            steps = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            steps = np.asarray(steps, int)
            max_step = max(steps)
            for f in fixed_strs:
                step = f.split('_')[-2].split(':')[-1]
                step = int(step)
                if step != max_step:
                    fp = nowp + '/' + f
                    rm(fp, remove=True)


def anaylze_imp(p='glint.bs.cont'):
    rp = root_path + 'work_space'
    path = rp + '/' + p + '/models/'
    from pathlib import Path
    assert osp.exists(path), path
    fixed_strs = [t.name for t in Path(path).glob('extra_*')]
    steps = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
    steps = np.asarray(steps)
    ind_min, ind_max = steps.argmin(), steps.argmax()
    fn1, fn2 = fixed_strs[ind_min], fixed_strs[ind_max]
    fn1 = path + fn1
    fn2 = path + fn2

    plt.figure()
    res = lz.msgpack_load(fn1)
    top_imp = res['dop']
    sub_imp = res['id2range_dop']
    plt.plot(top_imp)
    plt.yscale('log')

    res = lz.msgpack_load(fn2)
    top_imp = res['dop']
    sub_imp = res['id2range_dop']
    plt.plot(top_imp, alpha=0.9)
    plt.show()

    # plt.figure()
    # plt.plot()
    # plt.show()
    # todo divided by nimgs

    plt.figure()
    for i in range(1000):
        _ = plt.plot(sub_imp[str(i)])
    plt.yscale('log')
    plt.show()

    plt.figure()
    max_key = np.asarray(list(sub_imp.keys()), dtype=int).max()
    for i in range(max_key - 1000, max_key):
        _ = plt.plot(sub_imp[str(i)])
    plt.yscale('log')
    plt.show()

    all_sub_imp = np.concatenate(list(sub_imp.values()), )
    cnt = np.count_nonzero(
        all_sub_imp == -1
    )
    print(cnt / all_sub_imp.shape[0])
    plt.figure()
    plt.hist(all_sub_imp)
    plt.show()

    cnt = np.count_nonzero(
        top_imp == -1
    )
    print(cnt / top_imp.shape[0])
    plt.figure()
    plt.hist(top_imp)
    plt.show()


def my_worker_loop(dataset, index_queue, data_queue, done_event, collate_fn, seed, init_fn, worker_id):
    from torch.utils.data.dataloader import _set_worker_signal_handlers, ManagerWatchdog, MP_STATUS_CHECK_INTERVAL, \
        ExceptionWrapper
    import queue
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    try:
        global _use_shared_memory
        _use_shared_memory = True

        # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal happened again already.
        # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
        _set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        data_queue.cancel_join_thread()

        if init_fn is not None:
            init_fn(worker_id)

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                # Received the final signal
                assert done_event.is_set()
                return
            elif done_event.is_set():
                # Done event is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, batch_indices = r
            try:
                # samples = collate_fn([dataset[i] for i in batch_indices])
                samples = collate_fn(dataset[batch_indices])
            except Exception:
                # It is important that we don't store exc_info in a variable,
                # see NOTE [ Python Traceback Reference Cycle Problem ]
                data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
            else:
                data_queue.put((idx, samples))
                del samples
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass


torch.utils.data.dataloader._worker_loop = my_worker_loop

if __name__ == '__main__':
    # from Learner import TorchDataset, RandomIdSampler
    # from config import conf as gl_conf
    #
    # ds = TorchDataset(gl_conf.use_data_folder)
    # loader = torch.utils.data.DataLoader(
    #     ds, batch_size=conf.batch_size, num_workers=conf.num_workers,
    #     shuffle=False,
    #     sampler=RandomIdSampler(ds.imgidx,
    #                             ds.ids, ds.id2range),
    #     drop_last=True,
    #     pin_memory=True,
    # )
    # for data in loader:
    #     print(data.keys())

    cleanup()
    # anaylze_imp('emore.r50.dop/')
    pass

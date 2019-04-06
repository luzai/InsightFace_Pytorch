import sys

sys.path.insert(0, '/home/xinglu/prj/InsightFace_Pytorch')
import json
import lz
from lz import *
from torch.multiprocessing import Queue, Lock, Process
from sklearn.preprocessing import normalize
import argparse

mega_lst = '/data/share/megaface.ori/devkit/templatelists/megaface_features_list.json_100000_1'
fea_root = '/data/xinglu/prj/insightface/Evaluation/Megaface/'

t = json_load(mega_lst)
imgfns = t['path']

comb_from = ['feature_out.cl', 'feature_out.r152.ada.chkpnt.3.cl', 'feature_out.asia.emore.r50.ada']
# suffix = [ '_zju.artificial.idiot.bin']
dst_name = f'comb6'
dst = f'{fea_root}/{dst_name}/'

parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
parser.add_argument('--w1', type=str, default='.33')
parser.add_argument('--w2', type=str, default='.33')
args = parser.parse_args()
w1, w2 = args.w1, args.w2
w1 = float(w1)
w2 = float(w2)
w3 = 1 - w1 - w2


def consumer(queue, lock):
    while True:
        fn, fn2, fn3, dstfn = queue.get()
        f1 = load_mat(fn)
        f2 = load_mat(fn2)
        # f3 = load_mat(fn3)
        # f = w1 * f1 + w2 * f2 + w3 * f3
        assert f1[-1] == f2[-1]
        last = f1[-1]
        f = np.vstack((f1[:-1], f2[:-1]))
        f = normalize(f, axis=0)
        f = np.vstack((f, [last]))
        # if not osp.exists(osp.dirname(dstfn)):
        mkdir_p(osp.dirname(dstfn), delete=False, verbose=False)
        save_mat(dstfn, f)


queue = Queue(60)
lock = Lock()
consumers = []
for i in range(12):
    p = Process(target=consumer, args=(queue, lock))
    p.daemon = True
    consumers.append(p)
for c in consumers:
    c.start()
comb_from_ = comb_from[0]
assert osp.exists(f'{fea_root}/{comb_from_}')
for fn in glob.glob(f'{fea_root}/{comb_from_}/facescrub/**/*.bin', recursive=True):
    fn2 = fn.replace(comb_from[0], comb_from[1])
    assert osp.exists(fn2), fn2
    fn3 = None  # fn3 = fn.replace(comb_from[0], comb_from[2])
    dstfn = fn.replace(comb_from[0], dst_name)
    queue.put((fn, fn2, fn3, dstfn))
for ind, imgfn in enumerate(imgfns):
    if ind % 99 == 0:
        print(ind, len(imgfns))
    fn = f'{fea_root}/{comb_from[0]}/megaface/{imgfn}'
    fn2 = f'{fea_root}/{comb_from[1]}/megaface/{imgfn}'
    fn3 = f'{fea_root}/{comb_from[2]}/megaface/{imgfn}'
    fn = glob.glob(f'{fn}*.bin')[0]
    fn2 = glob.glob(f'{fn2}*.bin')[0]
    assert osp.exists(fn2), fn2
    fn3 = None  # fn3 = glob.glob(f'{fn3}*.bin')[0]
    dstfn = fn2.replace(comb_from[1], dst_name)
    # if not osp.exists((dstfn)):
    #     mkdir_p(osp.dirname(dstfn), delete=False)
    queue.put((fn, fn2, fn3, dstfn))

while not queue.empty():
    time.sleep(1)
    print('wait ...')

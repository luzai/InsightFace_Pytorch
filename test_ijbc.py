import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from lz import *

ijb_path = '/data2/share/ijbc/'
test1_path = ijb_path + '/IJB-C/protocols/test1/'
img_path = ijb_path + 'IJB/IJB-C/images/'
df_enroll = pd.read_csv(test1_path + '/enroll_templates.csv')
df_verif = pd.read_csv(test1_path + '/verif_templates.csv')
df_match = pd.read_csv(test1_path + '/match.csv')
dst = ijb_path + '/ijb.test1.proc/'

df1 = df_enroll[['TEMPLATE_ID', 'SUBJECT_ID']].groupby('TEMPLATE_ID').mean()
df2 = df_verif[['TEMPLATE_ID', 'SUBJECT_ID']].groupby('TEMPLATE_ID').mean()
df = pd.concat((df1, df2))
t2s = dict(zip(df.index, df.SUBJECT_ID))
all_tids = list(t2s.keys())

# IJBC_path = '/data1/share/IJB_release/'
# fn = (os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_face_tid_mid.txt'))
# df_tm = pd.read_csv(fn, sep=' ', header=None)
# fn = (os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_template_pair_label.txt'))
# df_pair = pd.read_csv(fn, sep=' ', header=None)
# fn = os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_name_5pts_score.txt')
# df_name = pd.read_csv(fn, sep=' ', header=None)

chkpnt_path = Path('work_space/arcsft.triadap.s64.0.1')
model_path = chkpnt_path / 'save'
conf = get_config(training=False, work_path=chkpnt_path)

from torch.utils.data.dataloader import default_collate

# mem = []
true_batch_size = 128 * 4 * conf.num_devs


def my_collate(batch):
    # global mem
    newb = []
    for b1 in batch:
        for b2 in b1:
            newb.append(b2)
    # mem.extend(newb)
    # newb = mem[:true_batch_size]
    # mem = mem[true_batch_size:]
    return {key: default_collate([d[key] for d in newb]) for key in newb[0]}


class DatasetIJBC(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = conf.test_transform

    def __getitem__(self, item):
        item = all_tids[item]
        df = None
        if item in np.asarray(df_enroll.TEMPLATE_ID):
            df = df_enroll
        if item in np.asarray(df_verif.TEMPLATE_ID):
            assert df is None
            df = df_verif
        assert df is not None, 'df None'
        tid = item
        sid = t2s[tid]
        imps = glob.glob(f'{dst}/{sid}/{tid}/*.png')
        res = []
        for imp in imps:
            img = cvb.read_img(imp)
            img = to_image(img)
            mirror = torchvision.transforms.functional.hflip(img)
            img = self.transform(img)
            mirror = self.transform(mirror)
            res.append({'img': img, 'sid': sid, 'tid': tid, 'finish': 0, 'imgp': imp})
            res.append({'img': mirror, 'sid': sid, 'tid': tid, 'finish': 0, 'imgp': imp})
        #     res.append(img)
        #     res.append(mirror)
        # res = torch.stack(res)
        # res = {'img': res, 'sid': sid, 'tid': tid, 'finish': 1}
        res[-1]['finish'] = 1
        return res

    def __len__(self):
        return len(all_tids)


def extract2db(dbname):
    import tqdm

    ds = DatasetIJBC()
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=12, shuffle=False, pin_memory=True,
                                         collate_fn=my_collate)

    learner = face_learner(conf, inference=True)
    assert conf.device.type != 'cpu'
    prefix = list(model_path.glob('model*_*.pth'))[0].name.replace('model_', '')
    learner.load_state(conf, prefix, True, True)
    learner.model.eval()
    logging.info('learner loaded')

    timer.since_last_check('start')
    db = Database(dbname)
    for ind, res in tqdm.tqdm(enumerate(loader), ):
        img = res['img']
        imp = res['imgp']
        # print(img.shape[0])
        sid = res['sid']
        tid = res['tid']
        # todo basthc size here msut 1
        sid = sid[0].item()
        tid = tid[0].item()
        finish = res['finish']
        # extract fea
        start = 0
        fea_l = []
        norm_l = []
        with torch.no_grad():
            while start < len(img):
                img_now = img[start:start + true_batch_size]
                # todo multi gpu: fix dataparallel
                fea, norm = learner.model(img_now, return_norm=True, normalize=False)
                start += true_batch_size
                fea_l.append(fea.cpu())
                norm_l.append(norm.cpu())
        fea = torch.cat(fea_l).numpy()
        norm = torch.cat(norm_l).numpy()
        ## save to db
        db[f'fea/{sid}/{tid}'] = fea
        db[f'norm/{sid}/{tid}'] = norm
        db[f'imp/{sid}/{tid}'] = msgpack_dumps(imp)
        # msgpack_loads(db[f'imp/{sid}/{tid}'][...].tolist())
        # if ind>10:break
    db.close()


from verifacation import *
# from torch.multiprocessing import Pool
from multiprocessing import Pool

if __name__ == '__main__':
    # load
    # agg
    dbname = work_path + 'ijbc.2.h5'
    extract2db(dbname)
    db = None


    def init_db():
        from lz import l2_normalize_np, Database, work_path
        global db
        dbname = work_path + 'ijbc.2.h5'
        db = Database(dbname, 'r')
        print(db, id(db), 'start')


    timer.since_last_check('start ')
    indt = 0
    df_matchl = df_match.to_records(index=False).tolist()


    def get_dist_label(enroll, verif=None):
        from lz import l2_normalize_np, Database, work_path
        if isinstance(enroll, tuple):
            enroll, verif = enroll
        tid = enroll
        sid_e = t2s[tid]
        try:
            fea_enroll = db[f'fea/{sid_e}/{tid}'].mean(axis=0, keepdims=True)
        except:
            print(f'fea/{sid_e}/{tid}')
            raise ValueError()
        # fea_enroll = np.clip(fea_enroll, 1e-6, 1e6)
        fea_enroll = l2_normalize_np(fea_enroll)
        tid = verif
        sid_v = t2s[tid]
        try:
            fea_verif = db[f'fea/{sid_v}/{tid}'].mean(axis=0, keepdims=True)
        except:
            print(f'fea/{sid_v}/{tid}')
            raise ValueError()
        # fea_verif = np.clip(fea_verif, 1e-6, 1e6)
        fea_verif = l2_normalize_np(fea_verif)
        diff = np.subtract(fea_enroll, fea_verif).astype(np.float64)
        dist = np.sum(np.square(diff), 1)
        # dist = 2 - 2 * (fea_enroll * fea_verif).sum()
        return float(dist), int(sid_e == sid_v)


    multi_pool = Pool(initializer=init_db, processes=38, )
    res = multi_pool.map(get_dist_label, df_matchl)
    res = np.asarray(res)
    dists, issames = res[:, 0], res[:, 1]
    issames = np.asarray(issames, dtype=int)

    # dists = []
    # issames = []
    # for ind, (enroll, verif) in enumerate(df_matchl):
    #     dist, label = get_dist_label(enroll, verif)
    #     dists.append(dist)
    #     issames.append(label)
    #     indt += 1
    #     if indt % 999 == 0:
    #         logging.info(f'{indt} / {df_match.shape[0]}')
    #     # if indt > 9999: break

    # get acc
    lz.msgpack_dump([np.asarray(dists, order='C'),
                     np.asarray(issames, order='C')
                     ], work_path + 't.pk')
    timer.since_last_check('end ')
    tpr, fpr, acc = calculate_roc_by_dist(dists, issames)
    tprat = []
    for fpr_thresh in [1e-5, 1e-4, 1e-3, 1e-2]:
        ind = np.where(fpr <= fpr_thresh)[0][-1]
        tprat.append(tpr[ind])
    print('acc is ', acc, 'tprat is ', tprat)
    plt.plot(fpr, tpr)
    plt.show()
    plt.semilogx(fpr, tpr)
    plt.show()
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        print(x_labels[fpr_iter], tpr[min_index])
    # db.close()
    from IPython import embed

    embed()
    timer.since_last_check('end')

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

chkpnt_path = Path('work_space/arcsft.bs2')
model_path = chkpnt_path / 'save'
conf = get_config(training=False, work_path=chkpnt_path)

from torch.utils.data.dataloader import default_collate

# mem = []
true_batch_size = 100 * 2 * conf.num_devs


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


def extract2db(dbname=work_path + 'ijbc.h5'):
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
                fea, norm = learner.model(img_now, need_norm=True)
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

if __name__ == '__main__':
    # load
    # agg
    dbname = work_path + 'ijbc.h5'
    # extract2db(dbname)
    db = Database(dbname, mode='r')
    # feaes = []
    # feavs = []
    dists = []
    issames = []
    timer.since_last_check('start ')
    for ind, row in df_match.iterrows():
        # print(ind, )
        enroll, verif = row
        tid = enroll
        sid_e = t2s[tid]
        fea_enroll = l2_normalize_np(db[f'fea/{sid_e}/{tid}'].mean(axis=0, keepdims=True))
        tid = verif
        sid_v = t2s[tid]
        fea_verif = l2_normalize_np(db[f'fea/{sid_v}/{tid}'].mean(axis=0, keepdims=True))
        # feaes.append(fea_enroll)
        # feavs.append(fea_verif)
        diff = np.subtract(fea_enroll, fea_verif)
        dist = np.sum(np.square(diff), 1)
        dists.append(dist)
        issames.append(int(sid_e == sid_v))
        # if ind > 100: break
    # get acc
    timer.since_last_check('end ')
    threshs = np.arange(0, 4, 0.01)
    issames = np.asarray(issames)
    # feaes = np.concatenate(feaes, axis=0)
    # feavs = np.concatenate(feavs, axis=0)
    dists = np.concatenate(dists, axis = 0 )
    tpr, fpr, acc, best_threshs = calculate_roc_by_dist(threshs, dists, issames, )
    print('acc is ', acc)
    db.close()
    timer.since_last_check('end')

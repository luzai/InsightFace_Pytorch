import sys

sys.path.insert(0, '/data1/xinglu/prj/InsightFace_Pytorch')
from lz import *
import lz
from torchvision import transforms as trans
import redis
import argparse
from mtcnn import get_reference_facial_points, warp_and_crop_face
import torch.utils.data, torchvision.transforms.functional
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn.metrics import auc, roc_curve
import h5py, lmdb, six
from PIL import Image

from config import conf

os.chdir(lz.root_path)
bs = conf.batch_size * 2
use_mxnet = False
DIM = conf.embedding_size  # 512
dump_mid_res = False

use_ijbx = 'IJBB'
IJBC_path = '/data1/share/IJB_release/' if 'amax' in hostname() else '/home/zl/zl_data/IJB_release/'
fn = (os.path.join(IJBC_path + f'{use_ijbx}/meta', f'{use_ijbx.lower()}_face_tid_mid.txt'))
df_tm = pd.read_csv(fn, sep=' ', header=None)
fn = (os.path.join(IJBC_path + f'{use_ijbx}/meta', f'{use_ijbx.lower()}_template_pair_label.txt'))
df_pair = pd.read_csv(fn, sep=' ', header=None)
fn = os.path.join(IJBC_path + f'{use_ijbx}/meta', f'{use_ijbx.lower()}_name_5pts_score.txt')
df_name = pd.read_csv(fn, sep=' ', header=None)

unique_tid = np.unique(df_pair.iloc[:, :2].values.flatten())
refrence = get_reference_facial_points(default_square=True)
img_list_path = IJBC_path + f'{use_ijbx}/meta/{use_ijbx.lower()}_name_5pts_score.txt'
img_path = IJBC_path + f'{use_ijbx}/loose_crop'
img_list = open(img_list_path)
files = img_list.readlines()
num_imgs = len(files)
img_feats = np.empty((df_tm.shape[0], DIM))


class DatasetIJBC2(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        try:
            self.env = lmdb.open(img_path + '/../imgs_lmdb', readonly=True,
                                 max_readers=1, lock=False,
                                 # readahead=False, meminit=False
                                 )
        except:
            self.env = None

    def __len__(self):
        return len(files)

    def get_raw(self, item):
        img_index = item
        each_line = files[img_index]
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        with open(img_name, 'rb') as f:
            bb = f.read()
        return bb

    def __getitem__(self, item):
        img_index = item
        each_line = files[img_index]
        name_lmk_score = each_line.strip().split(' ')
        faceness_score = float(name_lmk_score[-1])
        if self.env is None:
            img_name = os.path.join(img_path, name_lmk_score[0])
            img = cvb.read_img(img_name)
            img = cvb.bgr2rgb(img)  # this is RGB
        else:
            with self.env.begin(write=False) as txn:
                imgbuf = txn.get(str(item).encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            f = Image.open(buf)
            img2 = f.convert('RGB')
            img2 = np.asarray(img2)
            img = img2
        assert img is not None, img_name
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        warp_img = preprocess(img, landmark=lmk)
        # cvb.write_img(warp_img, f'/share/data/aligned/{name_lmk_score[0]}', )
        warp_img = to_image(warp_img)
        if not use_mxnet:
            img = self.test_transform(warp_img)
        else:
            img = np.array(np.transpose(warp_img, (2, 0, 1)))
            img = lz.to_torch(img).float()
        return img, faceness_score, item, name_lmk_score[0]


@torch.no_grad()
def test_ijbc3(conf, learner):
    if not use_mxnet:
        learner.model.eval()
    ds = DatasetIJBC2(flip=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs,
                                         num_workers=conf.num_workers,
                                         shuffle=False,
                                         pin_memory=False, )
    len(loader) * bs
    for ind, data in enumerate(loader):
        (img, faceness_score, items, names) = data
        if ind % 9 == 0:
            logging.info(f'ok {ind} {len(loader)}')
        if not use_mxnet:
            with torch.no_grad():
                img_feat = learner.model(img)
                img_featf = learner.model(img.flip((3,)))
                fea = (img_feat + img_featf) / 2.
                fea = fea.cpu().numpy()
        else:
            img = img.numpy()
            img_feat = learner.gets(img)
            img_featf = learner.gets(img[:, :, :, ::-1].copy())
            fea = (img_feat + img_featf) / 2.
        # fea = fea * faceness_score.numpy().reshape(-1, 1) # todo need?
        img_feats[ind * bs: min((ind + 1) * bs, ind * bs + fea.shape[0]), :] = fea
    print('last fea shape', fea.shape, np.linalg.norm(fea, axis=-1)[-10:], img_feats.shape)
    if dump_mid_res:
        import h5py
        f = h5py.File('/tmp/feas.ijbb.h5', 'w')
        f['feas'] = img_feats
        f.flush()
        f.close()

    templates, medias = df_tm.values[:, 1], df_tm.values[:, 2]
    p1, p2, label = df_pair.values[:, 0], df_pair.values[:, 1], df_pair.values[:, 2]
    unique_templates = np.unique(templates)
    # cache_tfn = lz.work_path + 'ijbc.tfeas.256.pk'
    cache_tfn = None
    if cache_tfn and osp.exists(cache_tfn):
        template_norm_feats = lz.msgpack_load(cache_tfn).copy()
        clst = msgpack_load(lz.work_path + 'ijbc.tclst.pk').copy()
        # clst = msgpack_load(work_path + 'ijbc.tgt.pk').copy()
        lbs = np.unique(clst)
        for lb in lbs:
            if lb == -1: continue
            mask = clst == lb
            if mask.sum() <= 4: continue
            nowf = template_norm_feats[mask, :]
            dist = cdist(nowf, nowf)
            dist[np.arange(dist.shape[0]), np.arange(dist.shape[0])] = dist.max() * 10
            wei = lz.softmax_th(-dist, dim=1, temperature=1)
            refinef = np.matmul(wei, nowf)
            refinef = refinef * 0.3 + nowf * 0.7
            refinef = normalize(refinef, axis=1)
            template_norm_feats[mask, :] = refinef
            if lb % 999 == 1:
                print('now refine ', lb, len(lbs), np.linalg.norm(nowf, axis=1), np.linalg.norm(refinef, axis=1))
    else:
        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
        for count_template, uqt in enumerate(unique_templates):
            (ind_t,) = np.where(templates == uqt)
            face_norm_feats = img_feats[ind_t]
            face_medias = medias[ind_t]
            unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
            media_norm_feats = []
            for u, ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                else:  # image features from the same video will be aggregated into one feature
                    media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
            media_norm_feats = np.array(media_norm_feats)
            # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
            template_feats[count_template] = np.sum(media_norm_feats, 0)
            if count_template % 2000 == 0:
                print('Finish Calculating {} template features.'.format(count_template))
        template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
        if cache_tfn:
            lz.msgpack_dump(template_norm_feats, cache_tfn)

    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)

    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]].reshape(-1, DIM)
        feat2 = template_norm_feats[template2id[p2[s]]].reshape(-1, DIM)
        # similarity_score = np.sum(feat1 * feat2, -1)
        similarity_score = - (
                np.linalg.norm(feat1, axis=-1) + np.linalg.norm(feat2, axis=-1) - 2 * np.sum(feat1 * feat2, -1))
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    if dump_mid_res:
        msgpack_dump([label, score], f'/tmp/score.ijbb.pk')
    print('score range', score.max(), score.min())
    # _ = plt.hist(score)
    fpr, tpr, _ = roc_curve(label, score)
    # plt.figure()
    # plt.semilogx(fpr, tpr, '.-')
    # plt.show()
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    res = []
    x_labels = [10 ** -6, 10 ** -4, 10 ** -3, ]
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(
                abs(fpr - x_labels[fpr_iter]), range(len(fpr))
            ))
        )
        print(x_labels[fpr_iter], tpr[min_index])
        res.append((x_labels[fpr_iter], tpr[min_index]))
    roc_auc = auc(fpr, tpr)
    print('roc aux', roc_auc)
    logging.info(f'perf {res}')
    if not use_mxnet:
        learner.model.train()
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelp',
                        # default='n2.irse.elu.casia.arcneg.15.0.48.0.02',
                        # 'irse.elu.casia.arc.ft',
                        # 'n2.irse.elu.casia.arcneg.15.0.4.0.1',
                        # 'irse.elu.casia.arc.mid.bl.ds',
                        default='r100.128.retina.clean.arc',
                        type=str)
    args = parser.parse_args()
    # lz.init_dev(lz.get_dev(2))
    # lz.init_dev((0, 1, ))
    if use_mxnet:
        from recognition.embedding import Embedding

        learner = Embedding(
            prefix=lz.home_path + 'prj/insightface/logs/r50-arcface-retina/model',
            epoch=16,
            ctx_id=0)
    else:
        from config import conf

        conf.need_log = False
        bs = conf.batch_size * 2
        conf.ipabn = False
        conf.cvt_ipabn = False
        conf.arch_ft = False
        conf.use_act = 'prelu'
        conf.net_depth = 100
        conf.net_mode = 'ir_se'
        conf.embedding_size = 512
        conf.input_size = 128
        conf.ds = False
        conf.use_bl = False
        conf.mid_type = ''  # 'gpool'
        from Learner import FaceInfer, face_learner

        gpuid = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        print(gpuid)
        learner = FaceInfer(conf, gpuid)
        learner.load_state(
            resume_path=f'work_space/{args.modelp}/save/',
            latest=True,
        )
        learner.model.eval()
    test_ijbc3(conf, learner)

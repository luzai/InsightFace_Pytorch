from lz import *
import lz
from torchvision import transforms as trans
import redis
os.chdir(lz.root_path)
use_redis = False
IJBC_path = '/data1/share/IJB_release/' if 'amax' in hostname() else '/home/zl/zl_data/IJB_release/'
ijbcp = IJBC_path + 'ijbc.info.h5'
try:
    df_tm, df_pair, df_name = df_load(ijbcp, 'tm'), df_load(ijbcp, 'pair'), df_load(ijbcp, 'name')
except:
    fn = (os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_face_tid_mid.txt'))
    df_tm = pd.read_csv(fn, sep=' ', header=None)
    fn = (os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_template_pair_label.txt'))
    df_pair = pd.read_csv(fn, sep=' ', header=None)
    fn = os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_name_5pts_score.txt')
    df_name = pd.read_csv(fn, sep=' ', header=None)
    df_dump(df_tm, ijbcp, 'tm')
    df_dump(df_pair, ijbcp, 'pair')
    df_dump(df_name, ijbcp, 'name')

use_mxnet = False
bs = 600
if use_mxnet:
    from recognition.embedding import Embedding
    
    learner = Embedding(prefix='/home/zl/prj/models/MS1MV2-ResNet100-Arcface/MS1MV2-ResNet100-Arcface', epoch=22,
                        ctx_id=7)
else:
    from config import conf
    
    conf.need_log = False
    conf.batch_size *= 4 * conf.num_devs
    bs = min(conf.batch_size, bs)
    conf.fp16 = False
    conf.ipabn = False
    conf.cvt_ipabn = False
    conf.net_depth = 152
    conf.use_chkpnt = False
    
    from Learner import FaceInfer
    
    learner = FaceInfer(conf, gpuid=range(conf.num_devs))
    learner.load_state(
        resume_path='work_space/emore.r152.ada.chkpnt.3/save/',
        latest=True,
    )
    # learner.load_model_only('work_space/backbone_ir50_ms1m_epoch120.pth')
    # learner.load_model_only('work_space/backbone_ir50_asia.pth')
    learner.model.eval()

unique_tid = np.unique(df_pair.iloc[:, :2].values.flatten())
from mtcnn import get_reference_facial_points, warp_and_crop_face
import torch.utils.data, torchvision.transforms.functional

refrence = get_reference_facial_points(default_square=True)
img_list_path = IJBC_path + './IJBC/meta/ijbc_name_5pts_score.txt'
img_path = '/share/data/loose_crop'
if not osp.exists(img_path):
    img_path = IJBC_path + './IJBC/loose_crop'

img_list = open(img_list_path)
files = img_list.readlines()
num_imgs = len(files)
img_feats = np.empty((df_tm.shape[0], 512))


class DatasetIJBC2(torch.utils.data.Dataset):
    def __init__(self, flip=False):
        self.flip = flip
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if use_redis:
            self.r = redis.Redis()
        else:
            self.r = None
    
    def __len__(self):
        return len(files)
    
    def __getitem__(self, item):
        img_index = item
        each_line = files[img_index]
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        if use_redis and f'ijbc/imgs/{name_lmk_score[0]}' in self.r:
            bb = self.r.get(f'ijbc/imgs/{name_lmk_score[0]}')
        else:
            with open(img_name, 'rb') as f:
                bb = f.read()
            if use_redis:
                self.r.set(f'ijbc/imgs/{name_lmk_score[0]}', bb)
        img = cvb.img_from_bytes(bb)  # also BGR
        #         img = cvb.read_img(img_name)
        img = cvb.bgr2rgb(img)  # this is RGB
        assert img is not None, img_name
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        warp_img = preprocess(img, landmark=lmk)
        # cvb.write_img(warp_img, f'/share/data/aligned/{name_lmk_score[0]}', )
        warp_img = to_image(warp_img)
        faceness_score = float(name_lmk_score[-1])
        if self.flip:
            import torchvision
            warp_img = torchvision.transforms.functional.hflip(warp_img)
        #         from IPython import embed; embed()
        if not use_mxnet:
            img = self.test_transform(warp_img)
        else:
            img = np.array(np.transpose(warp_img, (2, 0, 1)))
            img = lz.to_torch(img).float()
        
        return img, faceness_score, item, name_lmk_score[0]


ds = DatasetIJBC2(flip=False)

loader = torch.utils.data.DataLoader(ds, batch_size=bs,
                                     num_workers=12 if 'amax' in hostname() else 44,
                                     shuffle=False,
                                     pin_memory=True, )
# for ind, data in enumerate((loader)):
#     if ind % 9 == 0:
#         logging.info(f'ok {ind} {len(loader)}')
#     pass
# exit(0)
for ind, data in enumerate((loader)):
    
    (img, faceness_score, items, names) = data
    if ind % 9 == 0:
        logging.info(f'ok {ind} {len(loader)}')
    if not use_mxnet:
        with torch.no_grad():
            img_feat = learner.model(img)
            img_featf = learner.model(img.flip((3,)))
            fea = (img_feat + img_featf) / 2.
            fea = fea.cpu().numpy()
            fea = fea * faceness_score.numpy().reshape(-1, 1)
    else:
        img = img.numpy()
        img_feat = learner.gets(img)
        img_featf = learner.gets(img[:, :, :, ::-1].copy())
        fea = (img_feat + img_featf) / 2.
        fea = fea * faceness_score.numpy().reshape(-1, 1)
    img_feats[ind * bs: (ind + 1) * bs, :] = fea

templates, medias = df_tm.values[:, 1], df_tm.values[:, 2]
p1, p2, label = df_pair.values[:, 0], df_pair.values[:, 1], df_pair.values[:, 2]

unique_templates = np.unique(templates)
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

template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
for count_template, uqt in enumerate(unique_templates):
    template2id[uqt] = count_template

score = np.zeros((len(p1),))  # save cosine distance between pairs

total_pairs = np.array(range(len(p1)))
batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
total_sublists = len(sublists)
for c, s in enumerate(sublists):
    feat1 = template_norm_feats[template2id[p1[s]]]
    feat2 = template_norm_feats[template2id[p2[s]]]
    similarity_score = np.sum(feat1 * feat2, -1)
    score[s] = similarity_score.flatten()
    if c % 10 == 0:
        print('Finish {}/{} pairs.'.format(c, total_sublists))

from sklearn.metrics import roc_curve

msgpack_dump(score, 'work_space/t.pk')
print(score.max(), score.min())
# _ = plt.hist(score)
fpr, tpr, _ = roc_curve(label, score)

# plt.figure()
# plt.plot(fpr, tpr, '.-')
# plt.show()
#
# plt.figure()
# plt.semilogx(fpr, tpr, '.-')
# plt.show()

fpr = np.flipud(fpr)
tpr = np.flipud(tpr)  # select largest tpr at same fpr

x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
for fpr_iter in np.arange(len(x_labels)):
    _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
    print(x_labels[fpr_iter], tpr[min_index])
# plt.plot(fpr, tpr, '.-')
# plt.show()
# plt.semilogx(fpr, tpr, '.-')
# plt.show()
from sklearn.metrics import auc

roc_auc = auc(fpr, tpr)
print(roc_auc)
logging.info('finish ')

# from IPython import embed
# embed()

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
from pathlib import Path

# ijb_path = '/data2/share/ijbc/'
# test1_path = ijb_path + '/IJB-C/protocols/test1/'
# img_path = ijb_path + 'IJB/IJB-C/images/'
# df_enroll = pd.read_csv(test1_path + '/enroll_templates.csv')
# df_verif = pd.read_csv(test1_path + '/verif_templates.csv')
# df_match = pd.read_csv(test1_path + '/match.csv')
# dst = ijb_path + '/ijb.test1.proc/'
#
# df1 = df_enroll[['TEMPLATE_ID', 'SUBJECT_ID']].groupby('TEMPLATE_ID').mean()
# df2 = df_verif[['TEMPLATE_ID', 'SUBJECT_ID']].groupby('TEMPLATE_ID').mean()
# df = pd.concat((df1, df2))
# t2s = dict(zip(df.index, df.SUBJECT_ID))
# all_tids = list(t2s.keys())

IJBC_path = '/data1/share/IJB_release/'
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

conf = get_config()
learner = face_learner(conf, )
learner.load_state(conf, 'model.final.pth',
                   resume_path='work_space/glint.cont.2/models/',
                   latest=True,
                   model_only=True, )
learner.model.eval()
logging.info('learner loaded')

# use_topk = 999
# df_pair = df_pair.iloc[:use_topk, :]
unique_tid = np.unique(df_pair.iloc[:, :2].values.flatten())
from mtcnn import get_reference_facial_points, warp_and_crop_face
import torch.utils.data, torchvision.transforms.functional

refrence = get_reference_facial_points(default_square=True)
## way 1: speed up
img_list_path = IJBC_path + './IJBC/meta/ijbc_name_5pts_score.txt'
# img_path = IJBC_path + './IJBC/loose_crop'
img_path = '/share/data/loose_crop'
img_list = open(img_list_path)
files = img_list.readlines()
num_imgs = len(files)
# img_feats = np.ones((df_tm.shape[0], 512)) * np.nan
img_feats = np.empty((df_tm.shape[0], 512))


class DatasetIJBC2(torch.utils.data.Dataset):
    def __init__(self, flip=False):
        self.flip = flip
        self.lock = torch.multiprocessing.Lock()
    
    def __len__(self):
        return len(files)
    
    def __getitem__(self, item):
        img_index = item
        each_line = files[img_index]
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cvb.read_img(img_name)
        assert img is not None, img_name
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        warp_img = preprocess(img, landmark=lmk)
        warp_img = to_image(warp_img)
        faceness_score = float(name_lmk_score[-1])
        if self.flip:
            import torchvision
            warp_img = torchvision.transforms.functional.hflip(warp_img)
        img = conf.test_transform(warp_img)
        return img, faceness_score, item, name_lmk_score[0]


ds = DatasetIJBC2(flip=False)

bs = 513
loader = torch.utils.data.DataLoader(ds, batch_size=bs,
                                     num_workers=12,
                                     shuffle=False,
                                     pin_memory=True, )

for ind, data in enumerate((loader)):
    (img, faceness_score, items, names) = data
    if ind % 9 == 0:
        logging.info(f'ok {ind} {len(loader)}')
    with torch.no_grad():
        img_feat = learner.model(img)
        img_featf = learner.model(img.flip((3,)))
        fea = (img_feat + img_featf) / 2.
        fea = fea.cpu().numpy()
        fea = fea * faceness_score.numpy().reshape(-1, 1)
    img_feats[ind * bs: (ind + 1) * bs, :] = fea

## way 2 original
# for ind, row in df_name.iterrows():
#     if ind % 1000 == 0:
#         logging.info(f'extract {ind}/{df_name.shape[0]}')
#     tid = df_tm.iloc[ind, 1]
#     if not tid in unique_tid: continue
#     imgfn = row.iloc[0]
#     lmks = row.iloc[1:11]
#     lmks = np.asarray(lmks, np.float32).reshape((5, 2))
#     score = row.iloc[-1]
#     score = float(score)
#     imgfn = '/data1/share/IJB_release/IJBC/loose_crop/' + imgfn
#     img = cvb.read_img(imgfn)
#     warp_img_ori = warp_and_crop_face(img, lmks, refrence, crop_size=(112, 112))
#     warp_img = conf.test_transform(warp_img_ori).cuda()
#     flip_img = conf.test_transform(warp_img_ori[:, ::-1, :].copy()).cuda()
#     inp_img = torch.stack([warp_img, flip_img])
#     with torch.no_grad():
#         fea = learner.model(inp_img, normalize=False, return_norm=False)
#         fea = (fea[0, :] + fea[1, :]) / 2.
#         fea = fea.cpu().numpy().flatten()
#         # fea = sklearn.preprocessing.normalize(fea )
#     fea *= score
#     img_feats[ind, :] = fea

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

print(score.max(), score.min())
_ = plt.hist(score)
fpr, tpr, _ = roc_curve(label, score)

plt.figure()
plt.plot(fpr, tpr, '.-')
plt.show()

plt.figure()
plt.semilogx(fpr, tpr, '.-')
plt.show()

fpr = np.flipud(fpr)
tpr = np.flipud(tpr)  # select largest tpr at same fpr

x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
for fpr_iter in np.arange(len(x_labels)):
    _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
    print(x_labels[fpr_iter], tpr[min_index])
plt.plot(fpr, tpr, '.-')
plt.show()
plt.semilogx(fpr, tpr, '.-')
plt.show()
from sklearn.metrics import auc

roc_auc = auc(fpr, tpr)
print(roc_auc)
logging.info('finish ')

from IPython import embed

embed()

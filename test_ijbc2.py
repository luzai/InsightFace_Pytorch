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

chkpnt_path = Path('work_space/arcsft.triadap.s64.0.1/')
model_path = chkpnt_path / 'save'
conf = get_config(training=False, work_path=chkpnt_path)

from torch.utils.data.dataloader import default_collate

# mem = []
true_batch_size = 100 * 2 * conf.num_devs

import os
import numpy as np
import pickle as cPickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import timeit
import sklearn
import cv2
import sys
import glob

sys.path.append('./recognition')

# from menpo.visualize import print_progress
# from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
# from prettytable import PrettyTable
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def read_template_media_list(path):
    path_pk = path.replace('.txt', '.pk')
    try:
        templates, medias = msgpack_load(path_pk)
    except:
        ijb_meta = np.loadtxt(path, dtype=str)
        templates = ijb_meta[:, 1].astype(np.int)
        medias = ijb_meta[:, 2].astype(np.int)
        msgpack_dump([templates, medias], path_pk)
    return templates, medias


def read_template_pair_list(path):
    path_pk = path.replace('.txt', '.pk')
    try:
        t1, t2, label = msgpack_load(path_pk)
    except:
        pairs = np.loadtxt(path, dtype=str)
        t1 = pairs[:, 0].astype(np.int)
        t2 = pairs[:, 1].astype(np.int)
        label = pairs[:, 2].astype(np.int)
        msgpack_dump([t1, t2, label], path_pk)
    return t1, t2, label


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = cPickle.load(fid)
    return img_feats


class Embedding():
    def __init__(self):
        learner = face_learner(conf, inference=True)
        learner.load_state(conf, None, True, True)
        learner.model.eval()
        logging.info('learner loaded')
        self.learner = learner

    def get(self, rimg, landmark):
        warp_img = preprocess(rimg, landmark=landmark)
        warp_img = to_image(warp_img)
        img = conf.test_transform(warp_img)
        img = img.cuda().unsqueeze(0)
        with torch.no_grad():
            fea = self.learner.model(img)
        return fea.cpu().numpy().flatten()  # this norm =1 !

    def get_batch(self, rimgs, ):
        with torch.no_grad():
            fea = self.learner.model(rimgs)
        return fea.cpu().numpy()


def get_image_feature(img_path, img_list_path, model_path, gpu_id):
    img_list = open(img_list_path)
    files = img_list.readlines()
    dbimg = Database(img_path + '/imgs.h5')

    class DatasetIJBC2(torch.utils.data.Dataset):
        def __init__(self, flip=False):
            self.flip = flip

        def __len__(self):
            return len(files)

        def __getitem__(self, item):
            img_index = item
            each_line = files[img_index]
            name_lmk_score = each_line.strip().split(' ')
            img_name = os.path.join(img_path, name_lmk_score[0])
            try:
                warp_img = dbimg[img_name]
            except:
                img = cv2.imread(img_name)
                lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
                lmk = lmk.reshape((5, 2))
                warp_img = preprocess(img, landmark=lmk)
                dbimg[img_name] = warp_img

            warp_img = to_image(warp_img)
            faceness_score = float(name_lmk_score[-1])
            if self.flip:
                warp_img = torch.utils.functional.hflip(warp_img)
            img = conf.test_transform(warp_img)
            return img, faceness_score

    embedding = Embedding()
    db = Database(work_path + 'ijbc.fea.2.h5')
    ds = DatasetIJBC2()
    bs = 128 * 4 * 2
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=12, shuffle=False, pin_memory=True)
    for ind, (img, faceness_score) in enumerate(loader):
        if ind % 9 == 0:
            logging.info(f'ok {ind} {len(loader)}')
        with torch.no_grad():
            img_feat = embedding.get_batch(img)
        db[f'img_feats/{ind}'] = img_feat
        db[f'faceness_score/{ind}'] = faceness_score

    # for img_index, each_line in enumerate((files)):
    #     if img_index % 99 == 0:
    #         logging.info(f'ok {img_index} {len(files)}')
    #     name_lmk_score = each_line.strip().split(' ')
    #     img_name = os.path.join(img_path, name_lmk_score[0])
    #     img = cv2.imread(img_name)
    #     lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
    #     lmk = lmk.reshape((5, 2))
    #     img_feat = embedding.get(img, lmk)
    #     db[f'img_feats/{img_index}'] = img_feat
    #     faceness_score = float(name_lmk_score[-1])
    #     db[f'faceness_score/{img_index}'] = faceness_score
    #     img_feats.append(img_feat)
    #     faceness_scores.append(faceness_score)

    db.close()
    from IPython import embed
    embed()
    db = Database(work_path + 'ijbc.fea.2.h5', 'r')
    img_feats = []
    faceness_scores = []
    for ind in db.keys():
        iind = int(ind)
        fea = db[f'img_feats/{ind}']
        score = db[f'faceness_score/{ind}']
        img_feats.append(fea)
        faceness_scores.append(score)
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
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
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
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
        score[s] = similarity_score
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = cPickle.load(fid)
    return img_feats


## Step1: Load Meta Data
IJBC_path = '/data1/share/IJB_release/'
# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_face_tid_mid.txt'))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_template_pair_label.txt'))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
# img_feats = read_image_feature('./MS1MV2/IJBB_MS1MV2_r100_arcface.pkl')
img_path = IJBC_path + './IJBC/loose_crop'
img_list_path = IJBC_path + './IJBC/meta/ijbc_name_5pts_score.txt'
model_path = IJBC_path + './pretrained_models/MS1MV2-ResNet100-Arcface/model'
gpu_id = 1
img_feats, faceness_scores = get_image_feature(img_path, img_list_path, model_path, gpu_id)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))

## get template faeture

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

# use_norm_score = True  # if Ture, TestMode(N1)   # todo has he use norm?
# use_detector_score = True  # if Ture, TestMode(D1)
# use_flip_test = True  # if Ture, TestMode(F1) # todo has he flip?
use_norm_score = False  # if Ture, TestMode(N1)   # todo has he use norm?
use_detector_score = False  # if Ture, TestMode(D1)
use_flip_test = False  # if Ture, TestMode(F1) # todo has he flip?

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] / 2] + img_feats[:, img_feats.shape[1] / 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] / 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:, np.newaxis], 1, img_input_feats.shape[1])
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

## get template similarity
# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
score_save_name = work_path + 'ijbc.res.npy'
np.save(score_save_name, score)

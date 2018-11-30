'''
An example to show the interface.
'''
from skimage import io
from lz import *
import lz
import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from mtcnn import MTCNN
from Learner import face_learner
from model import l2_norm

# Note to load your model outside of `FaceVerification` function,
# otherwise, model will be loaded every comparison, which is too time-consuming.
# model = load_model()
mtcnn = MTCNN()
logging.info('mtcnn ok')
chkpnt_path = Path('work_space/arcsft.bs2')
model_path = chkpnt_path / 'save'
conf = get_config(training=False, work_path=chkpnt_path)
learner = face_learner(conf, inference=True)
assert conf.device.type != 'cpu'
prefix = list(model_path.glob('model*_*.pth'))[0].name.replace('model_', '')
learner.load_state(conf, prefix, True, True)
learner.model.eval()
logging.info('learner loaded')


def imgp2face(img_path1):
    img1 = io.imread(img_path1)  # this is rgb
    img1 = img1[..., ::-1]
    img1 = Image.fromarray(img1)
    bboxes, faces = mtcnn.align_multi(img1, conf.face_limit, 16)
    # todo handle multi bboxes
    face1 = faces[0]
    return face1


def face2fea(img):
    mirror = torchvision.transforms.functional.hflip(img)
    with torch.no_grad():
        fea = learner.model(conf.test_transform(img).cuda().unsqueeze(0))
        fea_mirror = learner.model(conf.test_transform(mirror).cuda().unsqueeze(0))
        fea = l2_norm(fea + fea_mirror).cpu().numpy().reshape(512)
    return fea


def FaceVerification(img_path1, img_path2, thresh=1.4):  # 1.7 1.5 1.4
    face1 = imgp2face(img_path1)
    face2 = imgp2face(img_path2)

    fea1 = face2fea(face1)
    fea2 = face2fea(face2)
    diff = np.subtract(fea1, fea2)
    dist = np.sum(np.square(diff), )  # dist is 0 - 2
    if dist > thresh:
        return 0
    else:
        return 1


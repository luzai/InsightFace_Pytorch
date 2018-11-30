import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import lz
from torchvision import transforms
from model import l2_norm

import pims, cvbase as cvb
from lz import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", action="store_true", default=True)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true",
                        default=False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true", default=True)
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true", default=True)
    args = parser.parse_args()

    conf = get_config(False)

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    # conf.work_path = Path('work_space/')
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')


    def extract_fea_from_img(img):
        img = img.copy()[..., ::-1].reshape(112, 112, 3)
        img = Image.fromarray(img)
        mirror = transforms.functional.hflip(img)
        with torch.no_grad():
            fea = learner.model(conf.test_transform(img).cuda().unsqueeze(0))
            fea_mirror = learner.model(conf.test_transform(mirror).cuda().unsqueeze(0))
            fea = l2_norm(fea + fea_mirror).cpu().numpy().reshape(512)

        return fea


    def extract_fea(res):
        res3 = {}
        for path, img in res.items():
            res3[path] = extract_fea_from_img(img)
        return res3


    # res, res2 = lz.msgpack_load(lz.work_path + '/yy.yy2.pk')
    # res3 = extract_fea(res)
    # res4 = extract_fea(res2)
    # lz.msgpack_dump([res3, res4], lz.work_path + 'yy.yy2.fea.pk')

    imgs = pims.ImageSequence(lz.work_path + 'face.yy2/gallery/*.png', process_func=cvb.rgb2bgr)
    res = {}
    for img, path in zip(imgs, imgs._filepaths):
        print(path)
        fea = extract_fea_from_img(img)
        res[path] = fea
    lz.msgpack_dump(res, lz.work_path + '/face.yy2/gallery/gallery.pk')

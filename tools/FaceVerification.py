'''
An example to show the interface.
'''
from lz import *
from skimage import io
import cv2, logging, torchvision, numpy as np
from PIL import Image
from pathlib import Path
import torch, lz
from mtcnn import MTCNN

# conf_str = 'th.ms1m.fan'
# conf_str = 'th.glint.fan'
# conf_str = 'mx.fan'
conf_str = 'th.ms1m.mtcnn'
# conf_str = 'th.glint.mtcnn'
# conf_str = 'mx.mtcnn'
if 'mtcnn' not in conf_str:
    import face_alignment
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      # face_detector='dlib',
                                      device='cuda:' + str(0),
                                      # device='cpu'
                                      )
else:
    mtcnn = MTCNN()
    logging.info('mtcnn ok')

if 'th' in conf_str:
    from config import conf
    
    conf.need_log = False
    conf.ipabn = False
    conf.cvt_ipabn = True
    conf.net_depth = 50
    from Learner import face_learner, FaceInfer
    from models.model import l2_norm
    
    learner = FaceInfer(conf, )
    learner.load_state(conf,
                       resume_path=root_path + 'work_space/asia.emore.r50.5/models'
                       # resume_path= root_path+'work_space/emore.r152.cont/models'
                       )
    learner.model.eval()
    logging.info('learner loaded')
else:
    
    from recognition.embedding import Embedding
    import os
    
    model_path = '/home/xinglu/prj/insightface/logs/model-r100-arcface-ms1m-refine-v2/model'
    assert os.path.exists(os.path.dirname(model_path)), os.path.dirname(model_path)
    embedding = Embedding(model_path, 0, 0)
    print('mx embedding loaded')


def imgp2face_fan(img_path1):
    img1 = io.imread(img_path1)  # this is rgb
    lmks = fa.get_landmarks_from_image(img1)
    if lmks is None or lmks == -1:
        warp_face = to_image(img1).resize((112, 112), Image.BILINEAR)
        print('no face', img_path1)
    else:
        if len(lmks) > 1:
            logging.warning(f'{img_path1} two face')
            ind = 0
        else:
            ind = 0
        kpoint = to_landmark5(lmks[ind])
        demo_img = img1.copy()
        for kp in kpoint:
            cv2.circle(demo_img, (kp[0], kp[1]), 1, (0, 255, 255,), 1)
        # plt_imshow(demo_img, );plt.show()
        warp_face = preprocess(img1, bbox=None, landmark=kpoint)  # this is rgb
        # plt_imshow(warp_face);plt.show()
        # cvb.write_img(warp_face[...,::-1], f'fan/{img_path1.split("/")[-1]}')
    return to_image(warp_face)


def imgp2face(img_path1):
    img1 = io.imread(img_path1)  # this is rgb
    img1 = img1[..., ::-1]  # this is bgr
    face = mtcnn.align_best(img1, min_face_size=16, imgfn=img_path1)  # bgr
    face = to_numpy(face)
    face = face[..., ::-1].copy()  # rgb
    face = to_img(face)
    return face


def face2fea(img):  # input img is bgr
    img = to_image(img)
    mirror = torchvision.transforms.functional.hflip(img)
    with torch.no_grad():
        fea = learner.model(conf.test_transform(img).cuda().unsqueeze(0))
        fea_mirror = learner.model(conf.test_transform(mirror).cuda().unsqueeze(0))
        fea = l2_norm(fea + fea_mirror).cpu().numpy().reshape(512)
    return fea


import sklearn


def feac2fea_mx(img):  # input img should be rgb!!
    fea = embedding.get(img, normalize=False)
    norm = np.sqrt((fea ** 2).sum())
    # print(norm)
    fea_n = sklearn.preprocessing.normalize(fea.reshape(1, -1)).flatten()
    return fea_n


def FaceVerification(img_path1, img_path2, thresh=1.5):  # 1.7 1.5 1.4
    import os
    assert os.path.exists(img_path1), img_path1
    if 'fan' not in conf_str:
        face1 = imgp2face(img_path1)
        face2 = imgp2face(img_path2)
    else:
        face1 = imgp2face_fan(img_path1)
        face2 = imgp2face_fan(img_path2)
    
    if 'mx' not in conf_str:
        fea1 = face2fea(face1)
        fea2 = face2fea(face2)
    else:
        fea1 = feac2fea_mx(np.asarray(face1))
        fea2 = feac2fea_mx(np.asarray(face2))
    
    diff = np.subtract(fea1, fea2)
    dist = np.sum(np.square(diff), )  # dist is 0 - 4
    # logging.info(f'dist: {dist} {img_path1} ')
    if dist > thresh:
        return 0, dist
    else:
        return 1, dist

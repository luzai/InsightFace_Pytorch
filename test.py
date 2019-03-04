# -*- coding: future_fstrings -*-
import lz, argparse, torchvision, struct, numpy as np, logging, cv2
import torch, glob, cvbase as cvb, os.path as osp
from torchvision import transforms as trans
from config import conf
from PIL import Image
from Learner import FaceInfer, l2_norm
from mtcnn import MTCNN
import itertools

cv_type_to_dtype = {
    5: np.dtype('float32'),
    6: np.dtype('float64')
}

dtype_to_cv_type = {v: k for k, v in cv_type_to_dtype.items()}


def write_mat(f, m):
    """Write mat m to file f"""
    if len(m.shape) == 1:
        rows = m.shape[0]
        cols = 1
    else:
        rows, cols = m.shape
    header = struct.pack('iiii', rows, cols, cols * 4, dtype_to_cv_type[m.dtype])
    f.write(header)
    f.write(m.data)


def save_mat(filename, m):
    """Saves mat m to the given filename"""
    return write_mat(open(filename, 'wb'), m)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='images_aligned_sample')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

root_folder_name = [name for name in args.data_dir.split('/') if name != '.'][0]
dst_folder = args.data_dir.replace(root_folder_name, root_folder_name + '_OPPOFeatures')
lz.mkdir_p(dst_folder, delete=False)

mtcnn = MTCNN()
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

conf.need_log = False
conf.fp16 = False
conf.ipabn = False
conf.cvt_ipabn = True
conf.net_depth = 50

learner = FaceInfer(conf, )
learner.load_state(
    # resume_path='work_space/emore.r152.cont/save/',
    resume_path='work_space/asia.emore.r50.5/save/',
    latest=True,
)
learner.model.eval()
ind = 0
for imgfn in itertools.chain(
        glob.glob(args.data_dir + '/**/*.jpg', recursive=True),
        glob.glob(args.data_dir + '/**/*.JPEG', recursive=True)):
    ind += 1
    if ind % 10 == 0:
        print(f'proc {ind}, {imgfn}')
    feafn = imgfn.replace(root_folder_name, root_folder_name + '_OPPOFeatures') + '_OPPO.bin'
    dst_folder = osp.dirname(feafn)
    lz.mkdir_p(dst_folder, delete=False)
    img = cvb.read_img(imgfn)  # bgr
    img1 = Image.fromarray(img)
    face = mtcnn.align_best(img1, limit=None, min_face_size=16, imgfn=imgfn)
    face = np.asarray(face)
    face = cvb.bgr2rgb(face)  # rgb
    face = Image.fromarray(face)
    mirror = torchvision.transforms.functional.hflip(face)
    with torch.no_grad():
        fea = learner.model(test_transform(face).cuda().unsqueeze(0))
        fea_mirror = learner.model(test_transform(mirror).cuda().unsqueeze(0))
        fea = l2_norm(fea + fea_mirror).cpu().numpy().reshape(512, 1)
    save_mat(feafn, fea)

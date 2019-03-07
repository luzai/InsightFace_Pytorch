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
parser.add_argument('--data_dir', type=str, default='/data/xinglu/prj/images_aligned_2018Autumn')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

assert osp.exists(args.data_dir), "The input dir not exist"
root_folder_name = args.data_dir.split('/')[-1]
src_folder = args.data_dir.replace(root_folder_name, root_folder_name + '_OPPOFaces')
# try:
assert osp.exists(src_folder), "Please run python crop_face_oppo.py --data_dir DATASET first"
# except:
#     lz.shell("")
dst_folder = args.data_dir.replace(root_folder_name, root_folder_name + '_OPPOFeatures')
lz.mkdir_p(dst_folder, delete=False)


class TestData(torch.utils.data.Dataset):
    def __init__(self):
        self.imgfn_iter = itertools.chain(
            glob.glob(src_folder + '/**/*.jpg', recursive=True),
            glob.glob(src_folder + '/**/*.JPEG', recursive=True))
        try:
            self.length = lz.msgpack_load(src_folder + '/nimgs.pk')
        except:
            logging.info(
                "After crop_face_oppo.py runned, *_OPPOFaces/nimgs.pk will be generetd, which logs the number of imgs. However no such file find now. We will guess the number of imgs. If any problem occurs, please rerun crop_face_oppo.py")
            self.length = int(10 * 10 ** 6)  # assume ttl test img less than 10M
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, item=None):
        try:
            imgfn = next(self.imgfn_iter)
            finish = 0
            img = cvb.read_img(imgfn)  # bgr
            img = cvb.bgr2rgb(img)
        except StopIteration:
            # logging.info(f'folder iter end')
            imgfn = ""
            finish = 1
            img = np.zeros((112, 112, 3), dtype=np.uint8)
        mirror = img[..., ::-1].copy()
        img = conf.test_transform(img)
        mirror = conf.test_transform(mirror)
        return {'imgfn': imgfn,
                "finish": finish,
                'img': img,
                'img_flip': mirror}


loader = torch.utils.data.DataLoader(TestData(), batch_size=args.batch_size,
                                     num_workers=24,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=False
                                     )

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
from sklearn.preprocessing import normalize

for ind, data in enumerate(loader):
    if (data['finish'] == 1).all().item():
        logging.info('finish')
        break
    if ind % 10 == 0:
        print(f'proc batch {ind}')
    img = data['img'].cuda()
    img_flip = data['img_flip'].cuda()
    with torch.no_grad():
        fea = learner.model(img)
        fea_mirror = learner.model(img_flip)
        fea += fea_mirror
        fea = fea.cpu().numpy()
    fea = normalize(fea, axis=1)
    for imgfn_, fea_ in zip(data['imgfn'], fea):
        feafn_ = imgfn_.replace(root_folder_name+'_OPPOFaces', root_folder_name + '_OPPOFeatures') + '_OPPO.bin'
        dst_folder = osp.dirname(feafn_)
        lz.mkdir_p(dst_folder, delete=False)
        save_mat(feafn_, fea_)

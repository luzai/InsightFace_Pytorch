import cv2
import argparse
from pathlib import Path
import torch, logging
from config import conf
conf.need_log = False
conf.batch_size *= 2
conf.fp16 = False
conf.ipabn = False
conf.cvt_ipabn = True
conf.upgrade_irse = False
conf.net_mode = 'ir'
conf.net_depth = 50
from Learner import l2_norm, FaceInfer, get_rec, unpack_auto
learner = FaceInfer(conf, )
# learner.load_state(
#         resume_path='work_space/asia.emore.r50.5/models/',
#         latest=True,
#     )
learner.load_model_only('work_space/backbone_ir50_ms1m_epoch120.pth')
# learner.load_model_only('work_space/backbone_ir50_asia.pth')
learner.model.eval()
logging.info('learner loaded')
from PIL import Image
from torchvision import transforms as trans
from lz import *
import mxnet as mx

"""
We use the same format as Megaface(http://megaface.cs.washington.edu) 
except that we merge all files into a single binary file.
"""
import struct
import numpy as np
import sys, os

cv_type_to_dtype = {
    5: np.dtype('float32')
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


ms1m_path = '/data2/share/'
ms1m_lmk_path = '/data1/share/testdata_lmk.txt'
lmks = pd.read_csv(ms1m_lmk_path, sep=' ', header=None).to_records(index=False).tolist()
logging.info(f'len lmks {len(lmks)}')
rec_test = get_rec('/data2/share/glint_test/train.idx')
use_rec = True


def img2db(name):
    
    class DatasetMS1M3(torch.utils.data.Dataset):
        def __init__(self):
            self.test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        def __getitem__(self, item):
            if not use_rec:
                data = lmks[item]
                imgfn = data[0]
                lmk = np.asarray(data[1:])
                img = cvb.read_img(f'{ms1m_path}/{imgfn}')  # bgr
                img = cvb.bgr2rgb(img)  # rgb
                warp_img = preprocess(img, landmark=lmk)
            else:
                rec_test.lock.acquire()
                s = rec_test.imgrec.read_idx(item + 1)
                rec_test.lock.release()
                header, img = unpack_auto(s, 'glint_test')
                img = mx.image.imdecode(img).asnumpy()  # rgb
                warp_img = np.array(img, dtype=np.uint8)
            # plt_imshow(img)
            # plt.show()
            # plt_imshow(warp_img)
            # plt.show()
            warp_img = Image.fromarray(warp_img)
            flip_img = torchvision.transforms.functional.hflip(warp_img)
            warp_img = self.test_transform(warp_img)
            flip_img = self.test_transform(flip_img)
            return {'img': warp_img,
                    'flip_img': flip_img, }
        
        def __len__(self):
            return len(lmks)
    
    ds = DatasetMS1M3()
    bs = 512
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=0, shuffle=False, pin_memory=True)
    db = Database(work_path + name + '.h5', )
    for ind, data in enumerate(loader):
        if ind % 9 == 0:
            logging.info(f'ind {ind}/{len(loader)}')
        warp_img = data['img']
        flip_img = data['flip_img']
        with torch.no_grad():
            from sklearn.preprocessing import normalize
            fea = learner.model(warp_img) + learner.model(flip_img)
            fea = fea.cpu().numpy()
            fea = normalize(fea)
        db[f'{ind}'] = fea
    
    db.close()


def db2np(name):
    db = Database(work_path + name + '.h5', 'r')
    res = np.empty((len(lmks), 512), dtype=np.float32)
    for ind in db.keys():
        bs = db[ind].shape[0]
        iind = int(ind)
        res[iind * bs: iind * bs + bs, :] = db[ind]
    db.close()
    
    save_mat(work_path + name + '.bin', res)
    msgpack_dump(res, work_path + name + '.pk', )


def read_mat(f):
    """
    Reads an OpenCV mat from the given file opened in binary mode
    """
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4 * 4))
    mat = np.fromstring(f.read(rows * stride), dtype=cv_type_to_dtype[type_])
    return mat.reshape(rows, cols)


def load_mat(filename):
    """
    Reads a OpenCV Mat from the given filename
    """
    return read_mat(open(filename, 'rb'))


def chknp():
    mat = load_mat(work_path + 'sfttri.bin')
    print(mat.shape, mat.dtype)


def rand2np():
    res = np.random.rand(1862120, 2).astype(np.float32)
    res = l2_normalize_np(res)
    save_mat(work_path + 'test3.bin', res)


if __name__ == '__main__':
    name = 'sfttri'
    # rm(name)
    img2db(name)
    db2np(name)

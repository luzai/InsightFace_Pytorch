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


ijb_path = '/data2/share/ijbc/'
ms1m_path = '/data1/share/'
ms1m_lmk_path = '/data1/share/testdata_lmk.txt'

# for ind, imgfn in enumerate(glob.iglob(f'{ms1m_path}/*/*/*.jpg') ) :
#     print(ind, imgfn)
#     if ind>5:break

# lmks = np.loadtxt('/data1/share/testdata_lmk.txt', dtype=object, delimiter=' ')
# msgpack_dump(lmks, work_path + 'lmk.pk')
# lmks = msgpack_load(work_path + 'lmk.pk')
lmks = pd.read_csv('/data1/share/testdata_lmk.txt', sep=' ', header=None).to_records(index=False).tolist()


# chkpnt_path = Path('work_space/arcsft.bs2')
def img2db():
    chkpnt_path = Path('work_space/arcsft.triadap.s64.0.1')
    model_path = chkpnt_path / 'save'
    conf = get_config(training=False, work_path=chkpnt_path)
    learner = face_learner(conf, inference=True)
    learner.load_state(conf, None, True, True)
    learner.model.eval()
    logging.info('learner loaded')
    
    from Learner import l2_norm
    from PIL import Image
    
    class DatasetMS1M3(torch.utils.data.Dataset):
        def __init__(self):
            pass
        
        def __getitem__(self, item):
            data = lmks[item]
            imgfn = data[0]
            lmk = np.asarray(data[1:])
            img = cvb.read_img(f'{ms1m_path}/{imgfn}')
            warp_img = preprocess(img, landmark=lmk)
            # plt_imshow(img)
            # plt.show()
            # plt_imshow(warp_img)
            # plt.show()
            warp_img = Image.fromarray(warp_img)
            flip_img = torchvision.transforms.functional.hflip(warp_img)
            warp_img = conf.test_transform(warp_img)
            flip_img = conf.test_transform(flip_img)
            return {'img': warp_img,
                    'flip_img': flip_img,
                    }
        
        def __len__(self):
            return len(lmks)
    
    ds = DatasetMS1M3()
    bs = 128 * 4 * 2
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=0, shuffle=False, pin_memory=True)
    db = Database(work_path + 'sfttri.h5', )
    for ind, data in enumerate(loader):
        if ind % 9 == 0:
            logging.info(f'ind {ind}')
        warp_img = data['img']
        flip_img = data['flip_img']
        with torch.no_grad():
            fea = l2_norm(learner.model(warp_img) + learner.model(flip_img)).cpu().numpy()
        db[f'{ind}'] = fea
    
    db.close()


def db2np():
    db = Database(work_path + 'sfttri.h5', 'r')
    res = np.empty((len(lmks), 512), dtype=np.float32)
    for ind in db.keys():
        bs = db[ind].shape[0]
        iind = int(ind)
        res[iind * bs: iind * bs + bs, :] = db[ind]
    db.close()
    
    save_mat(work_path + 'sfttri.bin', res)
    msgpack_dump(res, work_path + 'sfttri.pk', )


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
    img2db()
    db2np()

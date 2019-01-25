import sys, pandas as pd
from PIL import Image

sys.path.insert(0, '/home/xinglu/prj/InsightFace_Pytorch/')
ijb_path = '/data2/share/ijbc/'
test1_path = ijb_path + '/IJB-C/protocols/test1/'
img_path = ijb_path + 'IJB/IJB-C/images/'
df_enroll = pd.read_csv(test1_path + '/enroll_templates.csv')
df_verif = pd.read_csv(test1_path + '/verif_templates.csv')
df_match = pd.read_csv(test1_path + '/match.csv')
src = '/share/data/loose_crop/'
dst = '/share/data/loose_crop.2/'
from torch.multiprocessing import set_start_method

set_start_method('spawn')

from lz import *
from mtcnn import MTCNN

from config import conf

mtcnn = MTCNN()
mtcnn.share_memory()


def alignface(img1, ):
    img1 = Image.fromarray(img1)
    try:
        face1 = mtcnn.align_best(img1, limit=10, min_face_size=16, )
        face1 = np.asarray(face1)
        return face1, True
    
    except:
        logging.info(f'fail !! {img1}')
        face1 = to_image(img1).resize((112, 112), Image.BILINEAR)
        face1 = np.asarray(face1)
        return face1, False


if __name__ == '__main__':
    for imgfn in glob.glob(src + '/*.jpg'):
        img = cvb.read_img(imgfn)
        assert img is not None, imgfn
        face_ali, succ = alignface(img, )
        if not succ:
            # cvb.write_img(img, dst+'/')
            print('fail: ', imgfn)
import lz, argparse, torchvision, struct, numpy as np, logging, cv2
import torch, glob, cvbase as cvb, os.path as osp
from torchvision import transforms as trans
from config import conf
from PIL import Image
from Learner import FaceInfer, l2_norm
from mtcnn import MTCNN
import itertools
from lz import save_mat

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/xinglu/prj/images_aligned_2018Autumn')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--num_consumers', type=int, default=6)
parser.add_argument('--gpus', type=str, default="0")  # todo allow multiple gpu
args = parser.parse_args()

assert osp.exists(args.data_dir), "The input dir not exist"
root_folder_name = args.data_dir.split('/')[-1]
src_folder = args.data_dir.replace(root_folder_name, root_folder_name + '_OPPOFaces')
if not osp.exists(src_folder):
    logging.info('first crop face, an alternative way is run python crop_face_oppo.py --data_dir DATASET. ')
    from crop_face_oppo_fast import crop_face
    # from crop_face_oppo import crop_face
    
    crop_face(args)

dst_folder = args.data_dir.replace(root_folder_name, root_folder_name + '_OPPOFeatures')
lz.mkdir_p(dst_folder, delete=False)


class TestData(torch.utils.data.Dataset):
    def __init__(self, imgfn_iter):
        self.imgfn_iter = imgfn_iter
        try:
            self.imgfns = lz.msgpack_load(src_folder + '/all_imgs.pk')
        except:
            logging.info(
                "After crop_face_oppo.py runned, *_OPPOFaces/all_imgs.pk will be generetd, which logs img list. But, all_imgs.pk cannot be loaded, we are regenerating all img list now ...")
            self.imgfns = list(self.imgfn_iter)
        self.length = len(self.imgfns)
        # self.imgfn_iter is not thread safe
        # self.lock = torch.multiprocessing.Lock()
        # self.length = int(10 * 10 ** 6)  # assume ttl test img less than 10M
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, item):
        
        # self.lock.acquire()
        # imgfn = next(self.imgfn_iter)
        # self.lock.release()
        imgfn = self.imgfns[item]
        finish = 0
        img = cvb.read_img(imgfn)  # bgr
        img = cvb.bgr2rgb(img)
        
        mirror = img[..., ::-1].copy()
        img = conf.test_transform(img)
        mirror = conf.test_transform(mirror)
        return {'imgfn': imgfn,
                "finish": finish,
                'img': img,
                'img_flip': mirror}


imgfn_iter = itertools.chain(
    glob.glob(src_folder + '/**/*.jpg', recursive=True),
    glob.glob(src_folder + '/**/*.JPEG', recursive=True))
loader = torch.utils.data.DataLoader(TestData(imgfn_iter), batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=False
                                     )

conf.need_log = False
conf.fp16 = False
conf.ipabn = False
conf.cvt_ipabn = True
conf.net_depth = 50
conf.use_chkpnt = False
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
    # if ind % 10 == 0:
    print(f'proc batch {ind}')
    img = data['img'].cuda()
    img_flip = data['img_flip'].cuda()
    imgfn = data['imgfn']
    with torch.no_grad():
        fea = learner.model(img)
        fea_mirror = learner.model(img_flip)
        fea += fea_mirror
        fea = fea.cpu().numpy()
    fea = normalize(fea, axis=1)
    for imgfn_, fea_ in zip(imgfn, fea):
        feafn_ = imgfn_.replace(root_folder_name + '_OPPOFaces', root_folder_name + '_OPPOFeatures') + '_OPPO.bin'
        dst_folder = osp.dirname(feafn_)
        lz.mkdir_p(dst_folder, delete=False)
        save_mat(feafn_, fea_)

import lz, argparse, torchvision, struct, numpy as np, logging, cv2
import torch, glob, cvbase as cvb, os.path as osp
from torchvision import transforms as trans
from config import conf
from PIL import Image
from Learner import FaceInfer, l2_norm
from mtcnn import MTCNN
import itertools


def crop_face(args):
    assert osp.exists(args.data_dir), "The input dir not exist"
    root_folder_name = args.data_dir.split('/')[-1]
    dst_folder = args.data_dir.replace(root_folder_name, root_folder_name + '_OPPOFaces')
    lz.mkdir_p(dst_folder, delete=False)
    mtcnn = MTCNN()
    ind = 0
    all_img = []
    for imgfn in itertools.chain(
            glob.glob(args.data_dir + '/**/*.jpg', recursive=True),
            glob.glob(args.data_dir + '/**/*.JPEG', recursive=True)):
        ind += 1
        if ind % 10 == 0:
            print(f'proc {ind}, {imgfn}')
        dstimgfn = imgfn.replace(root_folder_name, root_folder_name + '_OPPOFaces')
        dst_folder = osp.dirname(dstimgfn)
        lz.mkdir_p(dst_folder, delete=False)
        img = cvb.read_img(imgfn)  # bgr
        img1 = Image.fromarray(img)
        face = mtcnn.align_best(img1, limit=None, min_face_size=16, imgfn=imgfn)
        face = np.asarray(face)  # bgr
        # face = cvb.bgr2rgb(face)  # rgb
        cvb.write_img(face, dstimgfn)
        all_img.append( dstimgfn)
    logging.info(f'finish crop all {ind} imgs')
    lz.msgpack_dump(all_img, dst_folder + '/' + 'all_imgs.pk')
    del mtcnn
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data1/xinglu/prj/images_aligned_2018Autumn')
    args = parser.parse_args()
    crop_face(args)

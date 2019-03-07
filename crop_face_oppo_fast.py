# -*- coding: future_fstrings -*-
from lz import *
import torch
import torchvision.transforms as transforms
from align_utils import mobilenet_v1
import numpy as np
import cv2, cvbase as cvb
import dlib
from align_utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from align_utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, \
    dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
import argparse, itertools, glob
import torch.backends.cudnn as cudnn, lz
from torchvision import transforms as trans
from torch.multiprocessing import Process, Queue, Lock

STD_SIZE = 120
default_args = dict(
    mode='gpu',
    show_flg=False,
    dump_res=False,
    dlib_bbox=True,
)

lz.init_dev(lz.get_dev(1))


class TestData(torch.utils.data.Dataset):
    def __init__(self, src_folder):
        self.imgfn_iter = itertools.chain(
            glob.glob(src_folder + '/**/*.jpg', recursive=True),
            glob.glob(src_folder + '/**/*.JPEG', recursive=True))
        logging.info('globbing the dir and obtaining the img list, may take some time ... ')
        self.imgfns = list(self.imgfn_iter)
        #  for performance measure
        # imgfns = self.imgfns.copy()
        # for i in range(10):
        #     self.imgfns.extend(imgfns.copy())
        
        # self.length = int(10 * 10 ** 6)  # assume ttl test img less than 10M
        self.length = len(self.imgfns)
        self.test_transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        self.face_detector = dlib.get_frontal_face_detector()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, item):
        # try:
        # imgfn = next(self.imgfn_iter)
        imgfn = self.imgfns[item]
        finish = 0
        img_ori = cvb.read_img(imgfn)  # bgr
        rects, scores, _ = self.face_detector.run(img_ori, 1, -1)
        if len(rects) == 1:
            rect = rects[0]
            # logging.info(f'{imgfn} {scores[0]}')
        elif len(rects) >= 2:
            chs_ind = np.argmax(scores)  # now chs max score face todo chs center bbox
            rect = rects[chs_ind]
            # logging.info(f'{imgfn} {scores[chs_ind]}')
        else:
            l, r, t, b = [0, 0, img_ori.shape[1], img_ori.shape[0]]
            rect = dlib.rectangle(l, r, t, b)
            logging.warning(f'{imgfn} no face')
        bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
        roi_box = parse_roi_box_from_bbox(bbox)
        img = crop_img(img_ori, roi_box)
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        # img = cvb.bgr2rgb(img) # rgb
        # except StopIteration:
        #     # logging.info(f'folder iter end')
        #     imgfn = ""
        #     finish = 1
        #     img = np.zeros((STD_SIZE, STD_SIZE, 3), dtype=np.uint8)
        #     roi_box = [0, 0, STD_SIZE, STD_SIZE]
        img = self.test_transform(img)
        roi_box = to_torch(np.asarray(roi_box, dtype=np.float32)).float()
        return {'imgfn': imgfn,
                "finish": finish,
                'img': img,
                'roi_box': roi_box
                }


def consumer(queue, lock):
    while True:
        imgfn, param, roi_box, dst_imgfn = queue.get()
        pts68 = [predict_68pts(param[i], roi_box[i]) for i in range(param.shape[0])]
        for img_fp, pts68_, dst in zip(imgfn, pts68, dst_imgfn):
            img_ori = cvb.read_img(img_fp)
            pts5 = to_landmark5(pts68_[:2, :].transpose())
            warped = preprocess(img_ori, landmark=pts5)
            # plt_imshow(warped, inp_mode='bgr');  plt.show()
            lz.mkdir_p(osp.dirname(dst), delete=False)
            cvb.write_img(warped, dst)


def crop_face(args):
    for k, v in default_args.items():
        setattr(args, k, v)
    assert osp.exists(args.data_dir), "The input dir not exist"
    root_folder_name = args.data_dir.split('/')[-1]
    src_folder = args.data_dir
    dst_folder = args.data_dir.replace(root_folder_name, root_folder_name + '_OPPOFaces')
    lz.mkdir_p(dst_folder, delete=False)
    ds = TestData(src_folder)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False,
                                         pin_memory=True,
                                         drop_last=False
                                         )
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'
    
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    
    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()
    
    # 2. load dlib model for face detection and landmark used for face cropping
    queue = Queue()
    lock = Lock()
    consumers = []
    for i in range(args.num_consumers):
        p = Process(target=consumer, args=(queue, lock))
        p.daemon = True
        consumers.append(p)
    for c in consumers:
        c.start()
    # 3. forward
    ttl_nimgs = 0
    ttl_imgs = []
    data_meter = lz.AverageMeter()
    model_meter = lz.AverageMeter()
    post_meter = lz.AverageMeter()
    lz.timer.since_last_check('start crop face')
    for ind, data in enumerate(loader):
        
        data_meter.update(lz.timer.since_last_check(verbose=False))
        if (data['finish'] == 1).all().item():
            logging.info('finish')
            break
        if ind % 10 == 0:
            logging.info(
                f'proc batch {ind}, data time: {data_meter.avg:.2f}, model: {model_meter.avg:.2f}, post: {post_meter.avg:.2f}')
        mask = data['finish'] == 0
        input = data['img'][mask]
        input_np = input.numpy()
        roi_box = data['roi_box'][mask].numpy()
        imgfn = np.asarray(data['imgfn'])[mask.numpy().astype(bool)]
        dst_imgfn = [img_fp.replace(root_folder_name, root_folder_name + '_OPPOFaces') for img_fp in imgfn]
        ttl_imgs.extend(dst_imgfn)
        ttl_nimgs += mask.sum().item()
        with torch.no_grad():
            if args.mode == 'gpu':
                input = input.cuda()
            param = model(input)
            param = param.squeeze().cpu().numpy().astype(np.float32)
        model_meter.update(lz.timer.since_last_check(verbose=False))
        queue.put((imgfn, param, roi_box, dst_imgfn))
        # pts68 = [predict_68pts(param[i], roi_box[i]) for i in range(param.shape[0])]
        # pts68_proc = [predict_68pts(param[i], [0, 0, STD_SIZE, STD_SIZE]) for i in range(param.shape[0])]
        # for img_fp, pts68_, pts68_proc_, img_, dst in zip(imgfn, pts68, pts68_proc, input_np, dst_imgfn):
        #     ## this may need opt to async read write
        #     img_ori = cvb.read_img(img_fp)
        #     pts5 = to_landmark5(pts68_[:2, :].transpose())
        #     warped = preprocess(img_ori, landmark=pts5)
        #     # plt_imshow(warped, inp_mode='bgr');  plt.show()
        #     lz.mkdir_p(osp.dirname(dst), delete=False)
        #     cvb.write_img(warped, dst)
        #
        #     ## this may cause black margin
        #     # pts5 = to_landmark5(pts68_proc_[:2, :].transpose())
        #     # warped = preprocess(to_img(img_), landmark=pts5)
        #     # # plt_imshow(warped, inp_mode='bgr'); plt.show()
        #     # dst = img_fp.replace(root_folder_name, root_folder_name + '_OPPOFaces')
        #     # cvb.write_img(warped, dst)
        #     if args.dump_res:
        #         img_ori = cvb.read_img(img_fp)
        #         pts_res = [pts68_]
        #         dst = img_fp.replace(root_folder_name, root_folder_name + '_kpts.demo')
        #         lz.mkdir_p(osp.dirname(dst), delete=False)
        #         draw_landmarks(img_ori, pts_res,
        #                        wfp=dst,
        #                        show_flg=args.show_flg)
        post_meter.update(lz.timer.since_last_check(verbose=False))
    lz.msgpack_dump(ttl_imgs, dst_folder + '/' + 'all_imgs.pk')
    del model, input
    torch.cuda.empty_cache()
    while not queue.empty():
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('--data_dir', type=str, default='/data1/xinglu/prj/images_aligned_2018Autumn')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_consumers', type=int, default=6)
    args = parser.parse_args()
    crop_face(args)

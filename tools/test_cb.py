import os
import numpy as np
from tqdm import tqdm
import re
from easydict import EasyDict as edict
import sys
import cv2
import argparse
import sklearn.preprocessing
from IPython import embed
import torch
import torchvision
from torchvision import transforms as trans

pca = False
output_name = 'fc1' if not pca else 'feature'
flip_test = False
preprocess_img = False


class TestDataset(object):
    def __init__(self, args):
        spisok = open(args.images_list).read().split('\n')[:-1]
        for i in range(len(spisok)):
            spisok[i] = args.prefix + spisok[i].split(' ')[0]
        for_test = np.array(spisok, dtype='str')
        self.for_test = for_test
        self.trans = trans.Compose([
            trans.ToTensor(), ])

    def __len__(self):
        return len(self.for_test)

    def __getitem__(self, item):
        imgfn = self.for_test[item]
        img = cv2.imread(imgfn)[..., ::-1]
        img = self.trans(img)
        return img


@torch.no_grad()
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    batch_size = args.batch_size
    print('#####', args.model, args.output_root)
    gpuid = 0
    import torch, torch.utils.data

    model_prefix, epoch = args.model.split(',')
    sys.path.insert(0, args.code)
    from config import conf
    conf.need_log = False
    from Learner import face_learner, FaceInfer
    learner = FaceInfer(conf, (gpuid,))
    learner.load_state(
        resume_path=model_prefix,
    )
    learner.model.eval()

    loader = torch.utils.data.DataLoader(
        TestDataset(args), batch_size=batch_size, num_workers=12,
        shuffle=False, pin_memory=True, drop_last=False
    )
    bin_filename = os.path.join(args.images_list.split('/')[-2], args.images_list.split('/')[-1].split('.')[0] + '.bin')
    if args.use_torch:
        model_name = model_prefix.strip('/').split('/')[-2]
    else:
        model_name = os.path.basename(model_prefix)
    if args.model_name is not None:
        model_name = args.model_name
    dump_path = os.path.join(args.output_root, model_name, bin_filename)
    print('###### features will be dumped to:%s' % dump_path)
    dirname = os.path.dirname(dump_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    dump = open(dump_path, 'wb')

    for batch in loader:
        import lz, torch
        dev = torch.device(f'cuda:{gpuid}')
        batch = batch - 127.5
        batch = batch / 127.5
        with torch.no_grad():
            embs = learner.model(lz.to_torch(batch).to(dev)).cpu().numpy()
            if flip_test:
                embs += learner.model(lz.to_torch(batch[..., ::-1].copy()).to(dev)).cpu().numpy()
        # from IPython import embed;  embed()    
        embs = sklearn.preprocessing.normalize(embs)

        for k in range(embs.shape[0]):
            dump.write(embs[k].astype(np.float32))
            dump.flush()
    dump.flush()
    dump.close()


def extract_test():
    import lz
    from config import conf
    conf.need_log=False
    data_prefix = '/home/zl/prj/data_old/'
    conf.use_torch = 1
    conf.code = lz.root_path
    conf.image_list = data_prefix + '/lists/MediaCapturedImagesStd_02_en_sim/jk_all_list.txt'
    conf.model = conf.load_from +',0'
    conf.prefix = data_prefix + '/zj_and_kj/'
    conf.gpu_num = 7
    conf.model_name = 'bak'
    conf.output_root = '/tmp/'
    main(conf)


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('--images_list', type=str, help='Path to list with images.')
        parser.add_argument('--output_root', type=str, help='Path to save embeddings.')
        parser.add_argument('--prefix', type=str, help='Prefix for paths to images.', default='')
        parser.add_argument('--batch_size', type=int, help='Batch size.', default=128)
        parser.add_argument('--gpu_num', type=int, help='Number of GPU to use.', default=0)
        parser.add_argument('--model', type=str, help='model_prefix,epoch')
        parser.add_argument('--use_torch', type=int)
        parser.add_argument('--code', type=str)
        parser.add_argument('--model_name', type=str, default=None)
        # python embeddings_test.py --prefix ../data/zj_and_jk/ --gpu_num $3 --model $MODEL','$EPOCH ../data/lists/MediaCapturedImagesStd_02_en_sim/jk_all_list.txt "$OUTPUT_ROOT" 
        parser.set_defaults(
            images_list='/home/zl/prj/data_old/lists/MediaCapturedImagesStd_02_en_sim/jk_all_list.txt',
            model='/home/zl/prj/work_space.old/emore.r100.bs.ft.tri.dop/save/,0',
            prefix='/home/zl/prj/data_old/zj_and_jk/',
            use_torch=1,
            batch_size=64,
            gpu_num=7,
            output_root='/home/zl/prj/output',
        )
        args = parser.parse_args()
        return args


    args = parse_arguments()
    # main(args)
    extract_test()
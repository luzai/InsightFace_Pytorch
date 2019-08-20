import os
import numpy as np
from tqdm import tqdm
import re
import mxnet as mx
from mxnet import ndarray as nd
from easydict import EasyDict as edict
import sys
import cv2
import argparse
import multiprocessing as mp
import time
from IPython import embed
import pdb

pca = False
output_name = 'fc1' if not pca else 'feature'
use_torch = False


def get_mod(gpuid):
    if not use_torch:
        import mxnet as mx
        ctx = mx.gpu(gpuid)
        batch_size = args.batch_size
        model_prefix, epoch = args.model.split(',')
        epoch = int(epoch)
        print('loading %s %d' % (model_prefix, epoch))
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
        sym = sym.get_internals()['fc1_output']
        mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 112, 112))])
        mod.set_params(arg_params, aux_params)
        return mod
    else:
        sys.path.insert(0, '/home/zl/prj/InsightFace_Pytorch/')
        from config import conf
        conf.ipabn = False
        conf.need_log = False
        from Learner import face_learner, FaceInfer
        learner = FaceInfer(conf, gpuid)
        learner.load_state(
            resume_path=('work_space/emore.r100.bs.ft.tri.dop/save/'),
        )
    return learner


def embedding_generator(gpu_id, q_in, q_out):
    gpuid = gpu_id
    # from IPython import embed; embed()
    mod = get_mod(gpu_id)
    # from torchvision import transforms as trans
    # test_transform = trans.Compose([
    #  trans.ToTensor(),
    #  trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    #  ])
    while True:
        data = q_in.get(block=True)
        if data is None:
            break
        imlist = []
        imnamelist = []
        for impath in data:
            im = cv2.imread(impath).astype(np.float32)
            if preprocess_img:
                im = im - 127.5
                im = im / 128.
            imlist.append(im[:, :, ::-1])  # RGB
            imnamelist.append(impath.replace(args.prefix, ''))
        batch = np.array(imlist).transpose((0, 3, 1, 2))  # RGB

        if not use_torch:
            data = mx.nd.array(batch)
            db = mx.io.DataBatch(data=(data,))
            # from IPython import embed; embed()
            mod.forward(db, is_train=False)
            embs = mod.get_outputs()[0].asnumpy()
            if flip_test:
                batch = batch[:, :, :, ::-1]
                data = mx.nd.array(batch)
                db = mx.io.DataBatch(data=(data,))
                mod.forward(db, is_train=False)
                embs += mod.get_outputs()[0].asnumpy()
        else:
            import lz, torch
            # batch = test_transform(batch)
            #           embed()
            batch = batch - 127.5
            batch = batch / 127.5
            dev = torch.device(f'cuda:{gpuid}')
            #             lz.plt_imshow(batch[0])
            #             lz.plt.savefig('/tmp/t.png')
            with torch.no_grad():
                embs = mod.model(lz.to_torch(batch).to(dev)).cpu().numpy()
                if flip_test:
                    embs += mod.model(lz.to_torch(batch[..., ::-1].copy()).to(dev)).cpu().numpy()

        #         for j in range(embs.shape[0]):
        #             embs[j] = embs[j]/np.sqrt(np.sum(embs[j]**2))
        import sklearn, sklearn.preprocessing
        embs = sklearn.preprocessing.normalize(embs)

        q_out.put((embs, imnamelist))


def embedding_consumer(emb_path, imname_path, q_out):
    idx = 0
    emb_outfile = open(emb_path, 'wb')
    imname_outfile = open(imname_path, 'w', encoding='utf-8')
    print('in consumer')
    start_time = time.time()
    while True:
        data = q_out.get(block=True)
        if data is None:
            break
        embs, imnames = data

        emb_outfile.write(embs.tobytes())
        for imname in imnames:
            imname_outfile.write(('%s\n' % imname))
        idx += 1
        if idx % 100 == 0:
            speed = 100 / (time.time() - start_time)
            print('process [%d/%d], speed: %f its/s, left:%f h ' % (
                idx, int(args.total_batch), speed, (args.total_batch - idx) / speed / 60 / 60))
            start_time = time.time()
    emb_outfile.close()
    imname_outfile.close()


def main():
    batch_size = args.batch_size
    print('#####', args.model, args.output_root)
    model_prefix, epoch = args.model.split(',')
    imagelist = open(args.images_list, 'rt', encoding='utf-8').read().split('\n')[:-1]

    for i in range(len(imagelist)):
        imagelist[i] = args.prefix + imagelist[i].strip().split(' ')[0]

    args.total_batch = np.ceil(len(imagelist) / float(batch_size))

    def read_batch():
        imlist = []
        for impath in imagelist:
            imlist.append(impath)
            if len(imlist) == batch_size:
                yield imlist
                imlist = []
        if len(imlist) != 0:
            yield imlist

    q_ins = []
    emb_generators = []
    q_out = mp.Queue()
    for i in range(num_process):
        gpu_id = gpus[i % len(gpus)]
        q_in = mp.Queue()
        q_ins.append(q_in)
        p = mp.Process(target=embedding_generator, args=(gpu_id, q_in, q_out))
        emb_generators.append(p)
        p.start()

    bin_filename = os.path.join(args.images_list.split('/')[-2], args.images_list.split('/')[-1].split('.')[0] + '.bin')
    dump_path = os.path.join(args.output_root, os.path.basename(model_prefix), bin_filename)
    print('###### features will be dumped to:%s' % dump_path)
    dirname = os.path.dirname(dump_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    imlist_path = dump_path.replace('bin', 'txt')
    conp = mp.Process(target=embedding_consumer, args=(dump_path, imlist_path, q_out))
    conp.start()

    for idx, batch in enumerate(read_batch()):
        imlist = batch
        q_ins[idx % len(q_ins)].put(imlist)

    for q_in in q_ins:
        q_in.put(None)
    for p in emb_generators:
        p.join()

    q_out.put(None)
    conp.join()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_list', type=str, help='Path to list with images.')
    parser.add_argument('--output_root', type=str, help='Path to save embeddings.')
    parser.add_argument('--prefix', type=str, help='Prefix for paths to images.', default='')
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=128)
    parser.add_argument('--gpu_num', type=int, help='Number of GPU to use.', default=0)
    parser.add_argument('--model', type=str, help='model_prefix,epoch')

    parser.set_defaults(
        images_list='../data/lists/dis/dis_list.txt',
        output_root='../output',
        model='/home/zl/prj/models/MS1MV2-ResNet100-Arcface/MS1MV2-ResNet100-Arcface,0',
        gpu_num=0,
        prefix='../data/',
        batch_size=256
    )

    return parser.parse_args()


# python embeddings_dis.py --prefix ../data/ --gpu_num $3 --model $MODEL','$EPOCH ../data/lists/dis/dis_list.txt "$OUTPUT_ROOT"

args = parse_arguments()
flip_test = False
preprocess_img = True if 'pca' in args.model else False
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
num_process = 8
if __name__ == '__main__':
    main()

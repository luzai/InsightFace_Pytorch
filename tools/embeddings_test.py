import tensorflow as tf
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
import sklearn.preprocessing
from IPython import embed

pca = False
output_name = 'fc1' if not pca else 'feature'


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    batch_size = args.batch_size

    print('#####', args.model, args.output_root)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    for_test = np.array(['1231', '213213'], dtype='str')
    test_ims = tf.placeholder(for_test.dtype, [None])

    def input_parser2(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, dct_method="INTEGER_ACCURATE")
        image = tf.cast(image_decoded, tf.float32)
        image = tf.transpose(image, perm=[2, 0, 1])
        if preprocess_img:
            image = image - 127.5
            image = image / 128.
        return image

    test_data = tf.data.Dataset.from_tensor_slices((test_ims))
    test_data = test_data.map(input_parser2, num_parallel_calls=48)
    test_data = test_data.prefetch(batch_size * 100)
    test_data = test_data.batch(batch_size)
    iterator2 = test_data.make_initializable_iterator()
    next_element2 = iterator2.get_next()
    sess.run(iterator2.initializer, feed_dict={test_ims: for_test})
    if not args.use_torch:
        gpuid = 0
        ctx = mx.gpu(gpuid)
        net = edict()
        net.ctx = ctx
        model_prefix, epoch = args.model.split(',')
        epoch = int(epoch)
        print('loading %s %d' % (model_prefix, epoch))
        net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(model_prefix, epoch)
        net.sym = net.sym.get_internals()['fc1_output']
        net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
        net.model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 112, 112))])
        net.model.set_params(net.arg_params, net.aux_params)
    else:
        model_prefix, epoch = args.model.split(',')
        sys.path.insert(0,  os.environ['HOME'] + '/prj/InsightFace_Pytorch/')
        from config import conf
        gpuid = 0
        conf.ipabn = False
        conf.need_log = False
        from Learner import face_learner, FaceInfer
        learner = FaceInfer(conf, gpuid)
        learner.load_state(
            resume_path=model_prefix,
        )

    #     data = mx.nd.array(np.random.normal(size=(batch_size,3,112,112)))
    #     db = mx.io.DataBatch(data=(data,))
    #     net.model.forward(db, is_train=False)

    spisok = open(args.images_list).read().split('\n')[:-1]
    for i in range(len(spisok)):
        spisok[i] = args.prefix + spisok[i].split(' ')[0]

    for_test = np.array(spisok, dtype='str')
    sess.run(iterator2.initializer, feed_dict={test_ims: for_test})
    bin_filename = os.path.join(args.images_list.split('/')[-2], args.images_list.split('/')[-1].split('.')[0] + '.bin')
    #     embed()
    if args.use_torch:
        model_name = model_prefix.strip('/').split('/')[-2]
    else:
        model_name = os.path.basename(model_prefix)
    dump_path = os.path.join(args.output_root, model_name, bin_filename)
    print('###### features will be dumped to:%s' % dump_path)
    dirname = os.path.dirname(dump_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    dump = open(dump_path, 'wb')

    for i in tqdm(range(int(np.ceil(len(spisok) / float(batch_size))))):
        batch = sess.run(next_element2)
        data = mx.nd.array(batch)

        # im = np.transpose(batch[0],(1,2,0))
        # cv2.imshow('x',im.astype(np.uint8))
        # cv2.waitKey(0)
        db = mx.io.DataBatch(data=(data,))

        if not args.use_torch:
            net.model.forward(db, is_train=False)
            embs = net.model.get_outputs()[0].asnumpy()

            if flip_test:
                batch = batch[:, :, :, ::-1]
                data = mx.nd.array(batch)
                db = mx.io.DataBatch(data=(data,))
                net.model.forward(db, is_train=False)
                embs += net.model.get_outputs()[0].asnumpy()
        else:
            import lz, torch
            dev = torch.device(f'cuda:{gpuid}')
            # batch = test_transform(batch)
            # from IPython import embed; embed()
            batch = batch - 127.5
            batch = batch / 127.5
            with torch.no_grad():
                embs = learner.model(lz.to_torch(batch).to(dev)).cpu().numpy()
                if flip_test:
                    embs += learner.model(lz.to_torch(batch[..., ::-1].copy()).to(dev)).cpu().numpy()

        embs = sklearn.preprocessing.normalize(embs)

        for k in range(embs.shape[0]):
            dump.write(embs[k].astype(np.float32))

    dump.close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_list', type=str, help='Path to list with images.')
    parser.add_argument('--output_root', type=str, help='Path to save embeddings.')
    parser.add_argument('--prefix', type=str, help='Prefix for paths to images.', default='')
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=128)
    parser.add_argument('--gpu_num', type=int, help='Number of GPU to use.', default=0)
    parser.add_argument('--model', type=str, help='model_prefix,epoch')
    parser.add_argument('--use_torch', type=int)
    # python embeddings_test.py --prefix ../data/zj_and_jk/ --gpu_num $3 --model $MODEL','$EPOCH ../data/lists/MediaCapturedImagesStd_02_en_sim/jk_all_list.txt "$OUTPUT_ROOT" 
    parser.set_defaults(
        images_list='../data/lists/MediaCapturedImagesStd_02_en_sim/jk_all_list.txt',
        output_root='../output',
        model='/home/zl/prj/InsightFace_Pytorch.3/work_space/emore.r100.bs.ft.tri.dop/save/,0',
        prefix='../data/zj_and_jk/',
        batch_size=256,
        use_torch=1,
        gpu_num=0,
    )
    args = parser.parse_args()
    return args


args = parse_arguments()
flip_test = False
preprocess_img = True if 'pca' in args.model else False
if __name__ == '__main__':
    main(args)

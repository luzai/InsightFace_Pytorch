
import argparse
import cv2
import numpy as np
import sys
import mxnet as mx
import datetime
from skimage import transform as trans
import sklearn
from sklearn import preprocessing


class Embedding:
    def __init__(self, prefix, epoch, ctx_id=0):
        print('loading', prefix, epoch)
        ctx = mx.gpu(ctx_id)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        # arg_shapes, out_shapes, aux_shapes= sym.infer_shape(data=(100,3,112,112))
        image_size = (112, 112)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        self.src = src
    
    def __call__(self, imgs):
        return self.gets(imgs)
    
    def gets(self, imgs):
        imgs = lz.to_numpy(imgs)
        input_blob = np.asarray(imgs)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        return self.model.get_outputs()[0].asnumpy()
    
    def get(self, rimg, landmark, normalize=False):
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        # plt_imshow(rimg)
        # plt.show()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        feat = self.model.get_outputs()[0].asnumpy()
        embedding = (feat[0, ...] + feat[1, ...]) / 2
        if normalize:
            embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding


import lz
from lz import *

if __name__ == '__main__':
    model_path = lz.root_path + 'Evaluation/IJB/pretrained_models/MS1MV2-ResNet100-Arcface/model'
    assert os.path.exists(os.path.dirname(model_path)), os.path.dirname(model_path)
    gpu_id = 2
    embedding = Embedding(model_path, 0, gpu_id)

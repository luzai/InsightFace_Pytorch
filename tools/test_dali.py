from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from lz import *

base = "/media/mem_data/faces_ms1m_112x112/"
idx_files = [base + "train.tc.idx"]
rec_files = [base + "train.rec"]


# Let us define a simple pipeline that takes images stored in recordIO format, decodes them and prepares them for ingestion in DL framework (crop, normalize and NHWC -> NCHW conversion).

class RecordIOPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(RecordIOPipeline, self).__init__(batch_size,
                                               num_threads,
                                               device_id)
        self.input = ops.MXNetReader(path=rec_files,
                                     index_path=idx_files,
                                     random_shuffle=True,
                                     shard_id=device_id,
                                     num_shards=num_gpus)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.5 * 255., 0.5 * 255., 0.5 * 255.],
                                            std=[0.5 * 255., 0.5 * 255., 0.5 * 255.]
                                            )
        self.coin = ops.CoinFlip(probability=0.5)
        self.iter = 0

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        rng = self.coin()
        images = self.cmnp(images, mirror=rng)
        return (images, labels)

    def iter_setup(self):
        pass


from nvidia.dali.plugin.pytorch import DALIGenericIterator

num_gpus = 4
batch_size = 200
pipes = [RecordIOPipeline(batch_size=batch_size, num_threads=2, device_id=device_id, num_gpus=num_gpus) for device_id in
         range(num_gpus)]
pipes[0].build()
dali_iter = DALIGenericIterator(pipes, ['imgs', 'labels'],
                                pipes[0].epoch_size("Reader"))


def new_iter(diter):
    for data in diter:
        labels = []
        imgs = []
        for d in data:
            l = d["labels"]
            if len(d["labels"].shape) == 2:
                l = l[:, 0]
            labels.append(l)
            imgs.append(d["imgs"].to(0))
        labels = torch.cat(labels)
        imgs = torch.cat(imgs)
        yield {"imgs": imgs, "labels": labels}


for i, data in enumerate(new_iter(dali_iter)):
    label = data["labels"]
    imgs = data["imgs"]
# embed()

# pipe_out = pipes[0].run()
# images, labels = pipe_out
# im1 = images.asCPU()
# im2 = im1.as_array()
# print(im2.shape)
# plt_imshow(im2[0])
# plt.show()

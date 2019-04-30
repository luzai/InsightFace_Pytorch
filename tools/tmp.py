import numpy as np
import matplotlib.pyplot as plt
from lz import *

init_dev(3)


def show_adabound():
    T = 22709 * 9
    # T = 200 * 391
    T = int(T)
    t = np.arange(1, T)
    gamma = 1e-3
    low = (1 - 1 / (gamma * t + 1))
    high = (1 + 1 / (gamma * t))
    plt.plot(t, low)
    plt.plot(t, high)
    plt.yscale('log')
    # plt.show()
    # for ind in [T//8, T*3//4, ]:
    for ind in [T * 2 // 9, T * 5 // 9, T * 7 // 9]:
        print(high[ind] - 1)


'''
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np

base = "/data1/share/faces_emore/"
idx_files = [base + "train.idx"]
rec_files = [base + "train.rec"]

# Let us define a simple pipeline that takes images stored in recordIO format, decodes them and prepares them for ingestion in DL framework (crop, normalize and NHWC -> NCHW conversion).

class RecordIOPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(RecordIOPipeline, self).__init__(batch_size,
                                         num_threads,
                                         device_id)
        self.input = ops.MXNetReader(path = rec_files, index_path = idx_files)
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.iter = 0

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        return (images, labels)

    def iter_setup(self):
        pass

batch_size = 16

pipe = RecordIOPipeline(batch_size=batch_size, num_threads=2, device_id = 0)
pipe.build()
pipe_out = pipe.run()
images, labels = pipe_out
im1 = images.asCPU()
im2 = im1.as_array()
im2.shape
plt_imshow(im2[0])
plt.show()
'''

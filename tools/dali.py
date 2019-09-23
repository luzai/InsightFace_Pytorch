from lz import *
from config import conf
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

base = "/media/mem_data/" + conf.dataset_name + "/"
if not osp.exists(base):
    base = "/data1/share/" + conf.dataset_name + "/"
idx_files = [base + "train.tc.idx"]
rec_files = [base + "train.rec"]


class PlainMxnetDs(object):
    def __init__(self):
        from mxnet import recordio
        self.imgrec = recordio.MXIndexedRecordIO(
            base + "train.idx", rec_files[0],
            'r')
        s = self.imgrec.read_idx(0)
        header, _ = recordio.unpack(s)
        assert header.flag > 0, 'ms1m or glint ...'
        logging.info(f'header0 label {header.label}')
        self.header0 = (int(header.label[0]), int(header.label[1]))
        self.id2range = {}
        self.idx2id = {}
        self.imgidx = []
        self.ids = []
        ids_shif = int(header.label[0])
        for identity in list(range(int(header.label[0]), int(header.label[1]))):
            s = self.imgrec.read_idx(identity)
            header, _ = recordio.unpack(s)
            a, b = int(header.label[0]), int(header.label[1])
            self.id2range[identity] = (a, b)
            self.ids.append(identity)
            self.imgidx += list(range(a, b))
        self.ids = np.asarray(self.ids)
        self.num_classes = len(self.ids)
        self.ids_map = {identity - ids_shif: id2 for identity, id2 in
                        zip(self.ids, range(self.num_classes))}  # now cutoff==0, this is identitical
        ids_map_tmp = {identity: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
        self.ids = np.asarray([ids_map_tmp[id_] for id_ in self.ids])
        self.id2range = {ids_map_tmp[id_]: range_ for id_, range_ in self.id2range.items()}
        for id_, range_ in self.id2range.items():
            for idx_ in range(range_[0], range_[1]):
                self.idx2id[idx_] = id_
        conf.num_clss = self.num_classes


plmxds = PlainMxnetDs()


# Let us define a simple pipeline that takes images stored in recordIO format, decodes them and prepares them for ingestion in DL framework (crop, normalize and NHWC -> NCHW conversion).

class RecordIOPipeline(Pipeline):
    def __init__(self, batch_size, device_id, num_gpus, num_threads=2):
        super(RecordIOPipeline, self).__init__(batch_size,
                                               num_threads,
                                               device_id,
                                               prefetch_queue_depth={"cpu_size": 6, "gpu_size": 2}
                                               )
        self.input = ops.MXNetReader(path=rec_files,
                                     index_path=idx_files,
                                     random_shuffle=True,
                                     shard_id=device_id,
                                     num_shards=num_gpus,
                                     initial_fill=len(plmxds.imgidx) // num_gpus,
                                     )
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


num_gpus = conf.num_devs
batch_size = conf.batch_size // conf.num_devs
pipes = [RecordIOPipeline(batch_size=batch_size, device_id=device_id, num_gpus=num_gpus,
                          ) for device_id in
         range(num_gpus)]
pipes[0].build()


class FDALIGenericIterator(DALIGenericIterator):
    def __len__(self):
        return (len(plmxds.imgidx) // conf.batch_size) + 1

    def force_reset(self):
        if self._stop_at_epoch:
            self._counter = 0
        else:
            self._counter = self._counter % self._size
        for p in self._pipes:
            p.reset()

    def __next__(self):
        data = super(FDALIGenericIterator, self).__next__()
        if isinstance(data, list):
            labels = []
            imgs = []
            for d in data:
                l = d["labels"]
                if len(d["labels"].shape) == 2:
                    l = l[:, 0]
                labels.append(l.long())
                imgs.append(d["imgs"].to(0))
            labels = torch.cat(labels)
            imgs = torch.cat(imgs)
            return {"imgs": imgs, "labels": labels, "labels_cpu": labels}
        else:
            return data


fdali_iter = FDALIGenericIterator(pipes, ['imgs', 'labels'],
                                  pipes[0].epoch_size("Reader"))


def get_loader_enum(loader):
    succ = False
    while not succ:
        loader_enum = (enumerate(loader))
        succ = True
    return loader_enum


if __name__ == '__main__':
    loader_enum = get_loader_enum(fdali_iter)

    while True:
        try:
            ind_data, data = next(loader_enum)
        except StopIteration as err:
            logging.info(f'one epoch finish err is {err}, {ind_data}')
            fdali_iter.reset()
            loader_enum = get_loader_enum(fdali_iter)
            ind_data, data = next(loader_enum)
        label = data["labels"]
        imgs = data["imgs"]
        print(ind_data, imgs.shape, label.shape, np.unique(label, return_counts=True)[1].mean())

    # plt_imshow(imgs[0].cpu() )
    plt_imshow_tensor(imgs[label == 12].cpu(), ncol=6)
    plt.show()

    # pipe_out = pipes[0].run()
    # images, labels = pipe_out
    # im1 = images.asCPU()
    # im2 = im1.as_array()
    # print(im2.shape)
    # plt_imshow(im2[0])
    # plt.show()

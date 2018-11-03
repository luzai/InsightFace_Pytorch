from easydict import EasyDict as edict
from pathlib import Path
import torch
import lz
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

lz.init_dev(lz.get_dev(n=1))
# lz.init_dev((0, 1))


def get_config(training=True):
    conf = edict()

    dbg = lz.dbg
    if dbg:
        # conf.num_steps_per_epoch = 38049
        conf.num_steps_per_epoch = 3
        # conf.no_eval = False
        conf.no_eval = True
    else:
        conf.num_steps_per_epoch = 38049
        # conf.num_steps_per_epoch = 3
        conf.no_eval = False
        # conf.no_eval = True
    conf.loss = 'softmax'  # softmax arcface
    conf.fgg = ''  # g gg ''
    conf.fgg_wei = 0  # 1
    conf.start_eval = False

    conf.data_path = Path('/data2/share/')
    conf.work_path = Path('work_space/dbg.bak/')
    # conf.work_path = Path('work_space/arcsft')
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path / 'save'
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'  # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf.test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path / 'faces_vgg_112x112'
    if dbg:
        conf.ms1m_folder = Path(lz.work_path) / 'faces_small'
    else:
        conf.ms1m_folder = conf.data_path / 'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path / 'faces_emore'
    conf.batch_size = 100  # irse net depth 50
    #   conf.batch_size = 200 # mobilefacenet
    # --------------------Training Config ------------------------
    if training:
        conf.log_path = conf.work_path / 'log'
        conf.save_path = conf.work_path / 'save'
        #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [12,15,18]
        conf.momentum = 0.9
        conf.pin_memory = True
        #         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()

        conf.facebank_path = conf.data_path / 'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10
        # when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30
        # the larger this value, the faster deduction, comes with tradeoff in small faces


    # --------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path / 'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10
        # when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf


gl_conf = get_config()

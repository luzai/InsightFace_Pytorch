from pathlib import Path
import lz
from lz import *
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

num_devs = 1
lz.init_dev(lz.get_dev(num_devs))


# lz.init_dev((1, 0, 2))
# lz.init_dev((3,))

def get_config(training=True, work_path=None):
    conf = edict()
    conf.num_devs = num_devs
    dbg = lz.dbg
    # if dbg:
    #     conf.num_steps_per_epoch = 38049
    #     conf.no_eval = False
    # else:
    # conf.num_steps_per_epoch = 38049
    conf.no_eval = False
    conf.loss = 'arcface'  # softmax arcface
    # conf.num_imgs = 3804846  # 85k id, 3.8M imgs
    conf.num_clss = 85164
    conf.rand_ratio = 9 / 27
    conf.fgg = ''  # g gg ''
    conf.fgg_wei = 0  # 1
    conf.tri_wei = .5
    conf.scale = 64.  # 30.
    conf.dop = np.ones(conf.num_clss, dtype=int) * -1
    conf.start_eval = False
    conf.instances = 4
    conf.data_path = Path('/data2/share/')
    conf.work_path = work_path or Path('work_space/arcsft.triadap.dop.adam')
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path / 'save'
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.4
    conf.net_mode = 'ir_se'  # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.device2 = torch.device("cuda:1")  # todo for at least two gpu, seems no need
    conf.start_epoch = 0  # 0
    conf.use_opt = 'adam'
    
    conf.test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path / 'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path / 'faces_ms1m_112x112'
    conf.glint_folder = conf.data_path / 'glint'
    conf.emore_folder = conf.data_path / 'faces_emore'

    conf.use_data_folder = conf.glint_folder
    conf.batch_size = 88 * num_devs if not dbg else 8 * num_devs  # xent: 96 92 tri: 112 108
    # conf.batch_size = 200 # mobilefacenet
    conf.num_recs = 2 if not dbg else 1  # todo too much worse speed ?
    # --------------------Training Config ------------------------
    if training:
        conf.log_path = conf.work_path / 'log'
        conf.save_path = conf.work_path / 'save'
        conf.weight_decay = 5e-4  # 5e-4 , 1e-6 for 1e-3, 0.3 for 3e-3
        conf.adam_betas1 = .9  # .85 to .95
        conf.adam_betas2 = .99
        conf.lr = 3e-3  # 3e-3  0.1   0.04,  # 0.028,  # 0.028 , 1e-2 # tri  0.00063,
        conf.epochs = 14
        conf.milestones = [2, 8, 11]
        # conf.epochs = 25
        # conf.milestones = [13, 19, 22]
        # conf.epochs = 48
        # conf.milestones = [12, 24, 36]
        # conf.epochs = 8
        # conf.milestones = [4, 6, 8]
        conf.momentum = 0.9
        conf.pin_memory = True
        conf.num_workers = 12 if not dbg else 0
        conf.ce_loss = CrossEntropyLoss()
        
        # conf.facebank_path = conf.data_path / 'facebank'
        # conf.threshold = 1.5
        # conf.face_limit = 10
        # # when inference, at maximum detect 10 faces in one image, my laptop is slow
        # conf.min_face_size = 30
        # # the larger this value, the faster deduction, comes with tradeoff in small faces
    
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

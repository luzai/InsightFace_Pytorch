from pathlib import Path
import lz
from lz import *
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

num_devs = 4
lz.init_dev(lz.get_dev(num_devs))


# lz.init_dev((1, 0, 2))
# lz.init_dev((3,))

def get_config(training=True, work_path=None):
    conf = edict()
    conf.num_devs = num_devs
    dbg = lz.dbg
    conf.no_eval = False
    conf.loss = 'arcface'  # softmax arcface
    
    conf.num_clss = None
    conf.dop = None
    conf.id2range_dop = None
    conf.data_path = Path('/data2/share/')
    conf.work_path = work_path or Path('work_space/glint.imp')
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path / 'save'
    conf.vgg_folder = conf.data_path / 'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path / 'faces_ms1m_112x112'
    conf.glint_folder = conf.data_path / 'glint'
    conf.emore_folder = conf.data_path / 'faces_emore'
    
    # conf.use_data_folder = conf.glint_folder
    conf.use_data_folder = conf.ms1m_folder
    conf.cutoff = 10 if conf.use_data_folder ==  conf.ms1m_folder  else 15
    
    conf.rand_ratio = 9 / 27
    conf.fgg = ''  # g gg ''
    conf.fgg_wei = 0  # 1
    conf.tri_wei = .1
    conf.scale = 64.  # 30.
    conf.start_eval = False
    conf.instances = 4
    
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    
    conf.drop_ratio = 0.4
    conf.net_mode = 'nasnetamobile'  # 'nasnetamobile' 'seresnext101' 'mobilefacenet'  # 'ir_se'  # or 'ir'
    conf.net_depth = 50
    
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # conf.device2 = torch.device("cuda:1")
    
    conf.test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    conf.batch_size = 68 * num_devs if not dbg else 8 * num_devs  # xent: 96 92 tri: 112 108
    conf.finetune = False
    if conf.finetune:
        conf.batch_size *= 6
    conf.batch_size = conf.batch_size // conf.instances * conf.instances
    conf.num_recs = 2 if not dbg else 1
    # --------------------Training Config ------------------------
    if training:
        conf.log_path = conf.work_path / 'log'
        conf.save_path = conf.work_path / 'save'
        conf.weight_decay = 5e-4 # 5e-4 , 1e-6 for 1e-3, 0.3 for 3e-3
        conf.start_epoch = 0  # 0
        conf.use_opt = 'sgd'
        conf.adam_betas1 = .9  # .85 to .95
        conf.adam_betas2 = .99
        conf.lr = 1e-1  # 3e-3  0.1   0.04,  # 0.028,  # 0.028 , 1e-2 # tri  0.00063,
        conf.lr_gamma = 0.1
        conf.epochs = 100
        conf.milestones = range(9, 100, 5)
        # conf.epochs = 25
        # conf.milestones = [13, 19, 22]
        # conf.epochs = 8
        # conf.milestones = [4, 6, 8]
        conf.momentum = 0.9
        conf.pin_memory = True
        conf.num_workers = 12 if not dbg else 1
        conf.ce_loss = CrossEntropyLoss()
        
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

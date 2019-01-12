from pathlib import Path
import lz
from lz import *
from torch.nn import CrossEntropyLoss
# todo label smooth
from torchvision import transforms as trans

num_devs = 1
# lz.init_dev((3,2,1,))
# lz.init_dev(lz.get_dev(num_devs))
lz.init_dev((3,))

conf = edict()
conf.num_devs = num_devs
dbg = lz.dbg
conf.no_eval = False
conf.loss = 'arcface'  # softmax arcface

conf.num_clss = None
conf.dop = None  # top_imp
conf.id2range_dop = None  # sub_imp
conf.explored=None

conf.data_path = Path('/data2/share/')
conf.work_path = Path('work_space/emore.r50.dop')
conf.model_path = conf.work_path / 'models'
conf.log_path = conf.work_path / 'log'
conf.save_path = conf.work_path / 'save'
conf.vgg_folder = conf.data_path / 'faces_vgg_112x112'
conf.ms1m_folder = conf.data_path / 'faces_ms1m_112x112'
conf.glint_folder = conf.data_path / 'glint'
conf.emore_folder = conf.data_path / 'faces_emore'

conf.use_data_folder = conf.emore_folder  # conf.emore_folder  # conf.glint_folder #  conf.ms1m_folder

if conf.use_data_folder == conf.ms1m_folder:
    conf.cutoff = 10
elif conf.use_data_folder == conf.glint_folder:
    conf.cutoff = 15
elif conf.use_data_folder == conf.emore_folder:
    conf.cutoff = 0

# conf.cutoff = 0
conf.mining = 'dop'  # 'dop' 'imp' rand.img(slow) rand.id # todo imp.grad imp.loss
conf.mining_init = -1  # for imp 1.6
conf.eps_greed = .3  # todo
conf.rand_ratio = 9 / 27

conf.fgg = ''  # g gg ''
conf.fgg_wei = 0  # 1
conf.tri_wei = 0.5
conf.scale = 64.  # 30.
conf.start_eval = False
conf.instances = 4

conf.input_size = [112, 112]
conf.embedding_size = 512

conf.drop_ratio = 0.4
conf.net_mode = 'ir_se'  # 'seresnext101' 'mobilefacenet'  'ir_se'  'ir'
conf.net_depth = 50

conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# conf.device2 = torch.device("cuda:1")

conf.test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

conf.batch_size = 145 * num_devs if not dbg else 8 * num_devs  # 89 xent: 96 92 tri: 112 108
conf.use_chkpnt=True
conf.backbone_with_head=True
conf.num_recs = 1
# --------------------Training Config ------------------------
conf.log_path = conf.work_path / 'log'
conf.save_path = conf.work_path / 'save'
conf.weight_decay = 5e-4  # 5e-4 , 1e-6 for 1e-3, 0.3 for 3e-3
conf.start_epoch = 0  # 0
conf.use_opt = 'sgd'
conf.adam_betas1 = .9  # .85 to .95
conf.adam_betas2 = .999  # 0.999 0.99
conf.lr = 1e-1  # 3e-3  0.1 4e-2 5e-4  # tri 6e-4
conf.lr_gamma = 0.1
# conf.epochs = 25
# conf.milestones = [14, 19, 22]
conf.epochs = 15
conf.milestones = [5, 9, 11]
conf.momentum = 0.9
conf.pin_memory = True
conf.num_workers = 12 if not dbg else 1
conf.ce_loss = CrossEntropyLoss()
conf.finetune = False
training = True  # False means test
if not training or conf.finetune:
    conf.batch_size *= 2
if not training:
    conf.need_log = False
else:
    conf.need_log = True
conf.batch_size = conf.batch_size // conf.instances * conf.instances
conf.head_init = ''  # work_space/glint.15.fc7.pk


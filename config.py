from pathlib import Path
import lz
from lz import *
from torch.nn import CrossEntropyLoss
from vat import VATLoss
# todo label smooth
# todo batch read redis

from torchvision import transforms as trans

dist = False
num_devs = 1
# lz.init_dev(0)
lz.init_dev(lz.get_dev(num_devs))

if dist:
    num_devs = 1

conf = edict()
conf.num_devs = num_devs
conf.no_eval = False
conf.loss = 'arcface'  # softmax arcface

conf.local_rank = None
conf.num_clss = None
conf.dop = None  # top_imp
conf.id2range_dop = None  # sub_imp
conf.explored = None

conf.data_path = Path('/data2/share/') if "amax" in hostname() else Path('/home/zl/zl_data/')
conf.work_path = Path('work_space/asia.emore.r50.test')
conf.model_path = conf.work_path / 'models'
conf.log_path = conf.work_path / 'log'
conf.save_path = conf.work_path / 'save'
vgg_folder = conf.data_path / 'faces_vgg_112x112'
ms1m_folder = conf.data_path / 'faces_ms1m_112x112'
glint_folder = conf.data_path / 'glint'
emore_folder = conf.data_path / 'faces_emore'
asia_emore = conf.data_path / 'asia_emore'
glint_test = conf.data_path / 'glint_test'
alpha_f64 = conf.data_path / 'alpha_f64'
alpha_jk = conf.data_path / 'alpha_jk'

conf.use_data_folder = asia_emore  # emore_folder  # conf.glint_folder #  conf.ms1m_folder #alpha_f64
conf.dataset_name = str(conf.use_data_folder).split('/')[-1]

if conf.use_data_folder == ms1m_folder:
    conf.cutoff = 10
elif conf.use_data_folder == glint_folder:
    conf.cutoff = 15
elif conf.use_data_folder == emore_folder:
    conf.cutoff = 0
elif conf.use_data_folder == asia_emore:
    conf.cutoff = 10
else:
    conf.cutoff = 0
conf.mining = 'rand.id'  # 'dop' 'imp' rand.img(slow) rand.id # todo imp.grad imp.loss
conf.mining_init = 1  # imp 1.6; rand.id 1; dop -1
# conf.eps_greed = .3  # todo
conf.rand_ratio = 9 / 27

conf.margin = 0.4
conf.fgg = ''  # g gg ''
conf.fgg_wei = 0  # 1
conf.tri_wei = 0
conf.scale = 64.
conf.start_eval = False
conf.instances = 4

conf.input_size = [112, 112]
conf.embedding_size = 512

conf.drop_ratio = 0.4
conf.net_mode = 'ir_se'  # csmobilefacenet mobilefacenet ir_se resnext densenet widerresnet
conf.net_depth = 50  # 100 121 169 201 264

# conf.test_transform = trans.Compose([
#     trans.ToTensor(),
#     trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
conf.use_redis = False
conf.use_chkpnt = False
conf.chs_first = True
conf.prof = False
conf.fast_load = True
conf.fp16 = True
conf.ipabn = True
conf.cvt_ipabn = False

conf.kd = False
conf.sftlbl_from_file = False
conf.alpha = .95
conf.temperature = 6

conf.online_imp = False
conf.use_test = True  # todo 'ijbc' 'glint'
conf.train_ratio = .7

conf.batch_size = 145 * num_devs
conf.ftbs_mult = 2
conf.board_loss_every = 10  # 100
conf.other_every = None if not conf.prof else 51
conf.num_recs = 1
# --------------------Training Config ------------------------
conf.log_path = conf.work_path / 'log'
conf.save_path = conf.work_path / 'save'
conf.weight_decay = 5e-4  # 5e-4 , 1e-6 for 1e-3, 0.3 for 3e-3
conf.start_epoch = 0
conf.start_step = 0
conf.use_opt = 'sgd'
conf.adam_betas1 = .9  # .85 to .95
conf.adam_betas2 = .999  # 0.999 0.99
conf.lr = 1e-3  # 3e-3  0.1 4e-2 5e-4  # tri 6e-4
conf.lr_gamma = 0.1
# conf.epochs = 25
# conf.milestones = [14, 19, 22]
conf.epochs = 3
conf.milestones = []
# conf.epochs = 9
# conf.milestones = [2, 5, 7]
conf.momentum = 0.9
conf.pin_memory = True
conf.num_workers = 24 if "amax" in hostname() else 66  # 4


# todo may use kl_div to speed up
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    
    def __init__(self, epsilon=0.1, ):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets1 = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        # targets2 = torch.cuda.FloatTensor(inputs.size()).fill_(0).scatter_(1, targets.unsqueeze(1).detach(), 1)
        targets3 = (1 - self.epsilon) * targets1 + \
                   self.epsilon / inputs.shape[1]
        loss = (-targets3 * log_probs).mean(0).sum()
        return loss


conf.ce_loss = CrossEntropyLoss()
# conf.ce_loss = CrossEntropyLabelSmooth()
if conf.use_test:
    conf.vat_loss_func = VATLoss(xi=1e-6, eps=8, ip=1)

training = True  # False means test
if not training:
    conf.batch_size *= 2
if not training:
    conf.need_log = False
else:
    conf.need_log = True
conf.batch_size = conf.batch_size // conf.instances * conf.instances
conf.head_init = ''  # work_space/glint.15.fc7.pk

# -*- coding: future_fstrings -*-
from pathlib import Path
import lz
from lz import *
from torch.nn import CrossEntropyLoss
from tools.vat import VATLoss
from torchvision import transforms as trans

# todo label smooth
# print = lambda x: logging.info(f'do not prt {x}')
dist = False
num_devs = 2
if dist:
    num_devs = 1
else:
    # lz.init_dev(lz.get_dev(num_devs, ok=(2, 3)))
    lz.init_dev(lz.get_dev(num_devs))
    # lz.init_dev((2, 3))

conf = edict()
conf.num_workers = ndevs * 6
conf.num_devs = num_devs
conf.no_eval = False
conf.start_eval = False
conf.loss = 'arcface'  # adamarcface adamrg adacos softmax arcface arcfaceneg cosface

conf.writer = None
conf.local_rank = None
conf.num_clss = None
conf.dop = None  # top_imp
conf.id2range_dop = None  # sub_imp
conf.explored = None

conf.data_path = Path('/data2/share/') if "amax" in hostname() else Path('/home/zl/zl_data/')
conf.work_path = Path('work_space/mbfc.casia.2.sgd')
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
casia_folder = conf.data_path / 'casia'  # the cleaned one todo may need the other for exploring the noise
retina_folder = conf.data_path / 'ms1m-retinaface-t1'
dingyi_folder = conf.data_path / 'faces_casia'

conf.use_data_folder = dingyi_folder
conf.dataset_name = str(conf.use_data_folder).split('/')[-1]
conf.clean_ids = None  # np.asarray(msgpack_load(root_path + 'train.configs/noise.20.pk', allow_np=False))

if conf.use_data_folder == ms1m_folder:
    conf.cutoff = 0
elif conf.use_data_folder == glint_folder:
    conf.cutoff = 15
elif conf.use_data_folder == emore_folder:
    conf.cutoff = 0
elif conf.use_data_folder == asia_emore:
    conf.cutoff = 10
else:
    conf.cutoff = 0
conf.mining = 'rand.id'  # todo balance opt # 'dop' 'imp' rand.img(slow) rand.id # todo imp.grad imp.loss
conf.mining_init = 1  # imp 1.6; rand.id 1; dop -1
conf.rand_ratio = 9 / 27

conf.margin = .5  # todo do not forget if use adacos!
conf.margin2 = .2
conf.topk = 15
conf.fgg = ''  # g gg ''
conf.fgg_wei = 0  # 1
conf.tri_wei = 0
conf.scale = 48  # 48 64
conf.instances = 4

conf.phi = 1.9
conf.input_rg_255 = False
conf.input_size = 112  # 128 224 112
conf.embedding_size = 512
conf.drop_ratio = .4
conf.conv2dmask_drop_ratio = .2
conf.lambda_runtime_reg = 5
conf.net_mode = 'mobilefacenet'  # effnet mbfc sglpth hrnet mbv3 mobilefacenet ir_se resnext densenet widerresnet
conf.decs = None
conf.net_depth = 18  # 100 121 169 201 264 50 20
conf.mb_mode = 'face.large'
conf.mb_mult = 1.285
# conf.mb_mode = 'face.small'
# conf.mb_mult = 2.005 # 1.37
conf.mbfc_wm = 1  # 1.2 ** conf.phi
conf.mbfc_dm = 2  # 1.56 ** conf.phi
conf.mbfc_se = True
conf.lpf = False
conf.eff_name = 'efficientnet-b0'

conf.test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
conf.use_loader = 'torch'  # todo mxent lack valds path and testing speed
conf.flip = True

conf.upgrade_irse = True
conf.upgrade_bnneck = False  # todo may pretrain by imgnet
conf.use_redis = False
conf.use_chkpnt = False
conf.chs_first = True
conf.prof = False
conf.fast_load = False
conf.ipabn = False
conf.cvt_ipabn = False
conf.kd = False
conf.sftlbl_from_file = True
conf.alpha = .95
conf.temperature = 24
conf.teacher_head_dev = 0  # num_devs - 1  # -1 #
conf.teacher_head_in_dloader = False  # todo bug when True

conf.online_imp = False
conf.use_test = False  # 'ijbc' 'glint' False 'cfp_fp'
conf.model1_dev = list(range(num_devs))
conf.model2_dev = list(range(num_devs))
conf.tau = 0.05
conf.mutual_learning = 0

conf.fp16 = True
conf.opt_level = "O1"
conf.batch_size = 200 * num_devs
conf.ftbs_mult = 2
conf.board_loss_every = 15
conf.log_interval = 105
conf.need_tb = True
conf.other_every = None  # 11
conf.num_recs = 1
conf.acc_grad = 4 // num_devs
# --------------------Training Config ------------------------
conf.weight_decay = 5e-4  # 5e-4 , 1e-6 for 1e-3, 0.3 for 3e-3
conf.use_opt = 'sgd'  # adabound adam radam
conf.adam_betas1 = .9  # .85 to .95
conf.adam_betas2 = .999  # 0.999 0.99
conf.final_lr = 1e-1
conf.lr = 1e-1  # 3e-3  #
conf.lr_gamma = 0.1
conf.start_epoch = 0
conf.start_step = 0
# conf.epochs = 37
# conf.milestones = (np.array([23, 32])).astype(int)
conf.epochs = 38
conf.milestones = (np.array([9, 13])).astype(int)
conf.warmup = 1  # conf.epochs/25 # 1 0
conf.epoch_less_iter = 1
conf.momentum = 0.9
conf.pin_memory = True
conf.fill_cache = .7
conf.val_ijbx = False
conf.spec_norm = True
conf.use_of = False

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
            inputs: prediction matrix (before softmax) with shape (bs, num_classes)
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


class CrossEntropySigSoft(nn.Module):

    def forward(self, inputs, targets):
        log_probs = logsigsoftmax(inputs)
        bs = inputs.shape[0]
        idx_ = torch.arange(0, bs, dtype=torch.long)
        log_probs2 = log_probs[idx_, targets]
        loss = - log_probs2.mean()
        return loss


def logsigsoftmax(logits):
    """
    Computes sigsoftmax from the paper - https://arxiv.org/pdf/1805.10829.pdf
    """
    max_values = torch.max(logits, 1, keepdim=True)[0]
    exp_logits_sigmoided = torch.exp(logits - max_values) * torch.sigmoid(logits)
    sum_exp_logits_sigmoided = exp_logits_sigmoided.sum(1, keepdim=True)
    log_probs = logits - max_values + F.logsigmoid(logits) - torch.log(sum_exp_logits_sigmoided)
    return log_probs


class CrossEntropySigSoft2(nn.Module):

    def forward(self, inputs, targets):
        inputs2 = inputs + F.logsigmoid(inputs / 12)  # todo
        loss = F.cross_entropy(inputs2, targets)
        return loss


conf.ce_loss = CrossEntropyLoss()  # CrossEntropyLabelSmooth()  # CrossEntropySigSoft2()  #
if conf.use_test:
    conf.vat_loss_func = VATLoss(xi=1e-6, eps=8, ip=1)
conf.need_log = True
# conf.batch_size = conf.batch_size // conf.instances * conf.instances
conf.head_init = ''  # work_space/glint.15.fc7.pk

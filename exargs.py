import  argparse
from config import conf
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--local_rank', default=None, type=int, )
parser.add_argument('--mbfc_wm', default=conf.mbfc_wm, type=float, )
parser.add_argument('--mbfc_dm', default=conf.mbfc_dm, type=float, )
parser.add_argument('--work_path', default=None, type=str, )
parser.add_argument('--epochs', default=conf.epochs, type=int, )
parser.add_argument('--scale', default=conf.scale, type=int)
parser.add_argument('--prof', default=conf.prof, type=bool)
parser.add_argument('--batch_size', default=conf.batch_size, type=int)
parser.add_argument('--acc_grad', default=conf.acc_grad, type=int)
parser.add_argument('--loss', default=conf.loss, type=str)
parser.add_argument('--cutoff', default=conf.cutoff, type=int)
parser.add_argument('--margin', default=conf.margin, type=float)
parser.add_argument('--margin2', default=conf.margin2, type=float)
parser.add_argument('--net_mode', default=conf.net_mode, type=str)
parser.add_argument('--input_size', default=conf.input_size, type=int)
parser.add_argument('--tau', default=conf.tau, type=float)
parser.add_argument('--mutual_learning', default=conf.mutual_learning, type=float)
parser.add_argument('--train_mode', default='mual', type=str)
parser.add_argument('--kd', default=conf.kd, type=bool)
parser.add_argument('--start_epoch', default=conf.start_epoch, type=int)
parser.add_argument('--start_step', default=conf.start_step, type=int)


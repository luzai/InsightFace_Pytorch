import argparse
from config import conf

parser = argparse.ArgumentParser(description='PyTorch Training')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2str(v):
    if isinstance(v, str):
        if v == 'None':
            return ''
        else:
            return v
    else:
        raise ValueError()

def str2tuple(v):
    if isinstance(v, (tuple,list)):
        return v
    elif isinstance(v, str):
        v=[int(k) for k in v.split('_')]
        return v
    else:
        raise argparse.ArgumentTypeError()

for k, v in conf.__dict__.items():
    if isinstance(v, bool):
        parser.add_argument(f"--{k}", type=str2bool, nargs='?',
                            const=True, default=v)
    elif isinstance(v, (float, int, )):  # note bool is int
        parser.add_argument(f'--{k}', default=v, type=type(v))
    elif isinstance(v, (str,)):
        parser.add_argument(f'--{k}', default=v, type=str2str, nargs='?', const=True, )
    elif isinstance(v, (tuple, list)):
        parser.add_argument(f'--{k}', default=v, type=str2tuple, nargs='?', const=True, )

parser.add_argument('--local_rank', default=None, type=int, )
parser.add_argument('--work_path', default=None, type=str, )
parser.add_argument('--data_url', default=None, type=str)
parser.add_argument('--init_method', default=None, type=str)
parser.add_argument('--train_url', default=None, type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

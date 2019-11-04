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


for k, v in conf.__dict__.items():
    if isinstance(v, bool):
        parser.add_argument(f"--{k}", type=str2bool, nargs='?',
                            const=True, default=v)
    elif isinstance(v, (float, int, str,)):
        parser.add_argument(f'--{k}', default=v, type=type(v))


parser.add_argument('--local_rank', default=None, type=int, )
parser.add_argument('--work_path', default=None, type=str, )

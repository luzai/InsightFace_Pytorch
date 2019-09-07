import argparse
from config import conf

parser = argparse.ArgumentParser(description='PyTorch Training')

for k,v in conf.__dict__.items():
    if isinstance(v, (bool, float, int ,str, )):
        parser.add_argument(f'--{k}', default=v, type = type(v))

parser.add_argument('--local_rank', default=None, type=int, )
parser.add_argument('--work_path', default=None, type=str, )



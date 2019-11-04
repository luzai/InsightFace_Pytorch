#!/usr/bin/env python3
import torch
from lz import *
from config import conf
from pathlib import Path

torch.backends.cudnn.benchmark = True


def log_conf(conf):
    conf2 = {k: v for k, v in conf.items() if not isinstance(v, (dict, np.ndarray))}
    logging.info(f'training conf is {conf2}')


from exargs import parser

if __name__ == '__main__':
    args = parser.parse_args()
    logging.info(f'args is {args}')
    if args.work_path:
        conf.work_path = Path(args.work_path)
        conf.model_path = conf.work_path / 'models'
        conf.log_path = conf.work_path / 'log'
        conf.save_path = conf.work_path / 'save'
    else:
        args.work_path = conf.work_path
    conf.update(args.__dict__)
    if conf.local_rank is not None:
        torch.cuda.set_device(conf.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method="env://")
        if torch.distributed.get_rank() != 0:
            set_stream_logger(logging.WARNING)
    
    from Learner import *

    # decs = msgpack_load('decs.pk')
    # conf.decs = None#decs
    # conf.net_mode = 'sglpth'
    learner = face_learner(conf, )
    ttl_params = (sum(p.numel() for p in learner.model.parameters()) / 1000000.0)
    from thop import profile

    flops, params = profile(learner.model.module,
                            input_size=(1, 3, conf.input_size, conf.input_size),
                            only_ops=(nn.Conv2d, nn.Linear),
                            device='cuda:0',
                            )
    flops /= 10 ** 9
    params /= 10 ** 6
    print('Total params: %.2fM' % ttl_params,
          'flops is',  flops,)

    # print(learner.model.module(...))

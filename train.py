#!/usr/bin/env python3
# -*- coding: future_fstrings -*-

from lz import *
import lz
from config import conf

conf.net_depth = 152
conf.fp16 = False
conf.ipabn = False
conf.cvt_ipabn = False
# conf.batch_size = 4
# conf.need_log = False
conf.tri_wei = .1
import argparse
from pathlib import Path

if not conf.online_imp:
    torch.backends.cudnn.benchmark = True


def log_conf(conf):
    conf2 = {k: v for k, v in conf.items() if not isinstance(v, (dict, np.ndarray))}
    logging.info(f'training conf is {conf2}')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--local_rank', default=None, type=int)
if __name__ == '__main__':
    args = parser.parse_args()
    conf.local_rank = args.local_rank
    if conf.local_rank is not None:
        torch.cuda.set_device(conf.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method="env://")
    
    from Learner import face_learner
    
    learner = face_learner(conf, )
    
    learner.load_state(
        # resume_path=Path('work_space/asia.emore.r50.test.ijbc.2/models/'),
        resume_path=Path('work_space/emore.r152.ada.chkpnt.2/models/'),
        load_optimizer=False,
        load_head=True,
        load_imp=False,
        latest=True,
    )
    # learner.calc_img_feas(out='work_space/emore.r152.fea.h5')
    # exit(0)
    # learner.init_lr()
    # conf.tri_wei = 0
    # log_conf(conf)
    # learner.train(conf, 1, mode='finetune', name='ft')
    
    # learner.init_lr()
    # conf.tri_wei = 0
    # log_conf(conf)
    # learner.train(conf, 1, name='xent')
    
    learner.init_lr()
    log_conf(conf)
    learner.train(conf, conf.epochs)
    # learner.train_simple(conf, conf.epochs)
    # learner.train_use_test(conf, conf.epochs)
    
    # learner.validate(conf,)
    # def calc_importance():
    #     steps = learner.list_steps(conf.model_path)
    #     for step in steps[::-1]:
    #         # step = steps[3]
    #         print('step', step, steps)
    #         learner.load_state_by_step(
    #             resume_path=conf.model_path,
    #             step=step,
    #             load_head=True,
    #         )
    #         learner.calc_importance(f'{conf.work_path}/{step}.pk')
    #
    #
    # calc_importance()
    
    # log_lrs, losses = learner.find_lr(conf,
    #                                   # final_value=100,
    #                                   num=200,
    #                                   bloding_scale=1000)
    # best_lr = 10 ** (log_lrs[np.argmin(losses)])
    # print(best_lr)
    # conf.lr = best_lr
    # learner.push2redis()

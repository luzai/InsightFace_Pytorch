#!/usr/bin/env python3

from lz import *
from config import conf
from Learner import face_learner
import argparse
from pathlib import Path
import lz

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
    
    learner = face_learner(conf, )
    
    learner.load_state(
        resume_path=Path('work_space/asia.emore.r50.5/save/'),
        load_optimizer=False,
        load_head=True,
        load_imp=False,
        latest=True,
    )
    
    # learner.init_lr()
    # conf.tri_wei = 0
    # log_conf(conf)
    # learner.train(conf, 1, mode='finetune', name='ft')
    
    # learner.init_lr()
    # conf.tri_wei = 0
    # log_conf(conf)
    # learner.train(conf, 1, name='xent')
    
    learner.init_lr()
    conf.tri_wei = 0
    log_conf(conf)
    # learner.train(conf, conf.epochs)
    learner.train_simple(conf, conf.epochs)
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
    
    # learner.calc_feature(out='work_space/ms1m.rv1.fc7.pk')
    
    # log_lrs, losses = learner.find_lr(conf,
    #                                   # final_value=100,
    #                                   num=200,
    #                                   bloding_scale=1000)
    # best_lr = 10 ** (log_lrs[np.argmin(losses)])
    # print(best_lr)
    # conf.lr = best_lr
    # learner.push2redis()

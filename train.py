#!/usr/bin/env python3

from lz import *
from config import conf
from Learner import face_learner
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--local_rank', default=None, type=int)
if __name__ == '__main__':
    args = parser.parse_args()
    conf.local_rank=args.local_rank
    if conf.local_rank is not None:
        torch.cuda.set_device(conf.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                               init_method="env://")
    learner = face_learner(conf, )
    
    ## for resume or evaluate
    learner.load_state(conf,
                       resume_path=Path('work_space/emore.r50.dop.nohead.notri.chkpnt/models'),
                       model_only=False,
                       load_optimizer=True,
                       latest=True,
                       load_imp=True,
                       )
    
    # log_lrs, losses = learner.find_lr(conf,
    #                                   # final_value=100,
    #                                   num=200,
    #                                   bloding_scale=1000)
    # best_lr = 10 ** (log_lrs[np.argmin(losses)])
    # print(best_lr)
    # conf.lr = best_lr
    # learner.push2redis()
    learner.init_lr()  # todo what if ...
    learner.train(conf, conf.epochs)
    
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

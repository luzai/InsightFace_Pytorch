#!/usr/bin/env python3
# -*- coding: future_fstrings -*-

from lz import *
from config import conf

# conf.need_log = False
# conf.net_mode = 'ir'
# conf.upgrade_irse = False
import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True


def log_conf(conf):
    conf2 = {k: v for k, v in conf.items() if not isinstance(v, (dict, np.ndarray))}
    logging.info(f'training conf is {conf2}')


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--local_rank', default=None, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    conf.local_rank = args.local_rank
    if conf.local_rank is not None:
        torch.cuda.set_device(conf.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method="env://")
        if torch.distributed.get_rank() != 0:
            set_stream_logger(logging.WARNING)
    from Learner import *

    learner = face_cotching(conf, )
    # learner = face_cotching_head(conf, )
    ress = {}
    for p in [
        # 'ms1m.mb.arc.warmup.ghm',
        # 'ms1m.mb.arc.warmup.ghm.2',
        # 'ms1m.mb.arc.2',
        # 'ms1m.mb.arc.cl.2'
        # 'ms1m.mb.arc.cutoff',
        # 'ms1m.mb.arc.cutoff.bl',
        # 'ms1m.mb.arc.all',
        # 'ms1m.mb.arc.neg',
        # 'ms1m.mb.arcneg.2.2.5',
        # 'ms1m.mb.neg.top10',
        # 'ms1m.mb.neg',
        # 'ms1m.mb.neg.2',
        # 'ms1m.mb.sft',
        # 'ms1m.mb.sft.long',
        # 'emore.r50.dop',
        # 'emore.r152.ada.chkpnt',
        # 'emore.r152.ada.chkpnt.2',
        # 'emore.r152.ada.chkpnt.3',
        # 'casia.r50.arc.bl',
        # 'casia.r50.sft',
        # 'casia.r50.arc.use_test',
        # 'casia.coth.cont',
        # 'casia.r20.cotch',
    ]:
        learner.load_state(
            # fixed_str='2019-04-06-20_accuracy:0.707857142857143_step:2268_None.pth',
            resume_path=Path(f'work_space/{p}/save/'),
            load_optimizer=False,
            load_head=False, # todo note !!!
            load_imp=False,
            latest=False,
            load_model2=True,
        )
        # res = learner.validate_ori(conf)
        # ress[p] = res
        # logging.warning(f'{p} res: {res}')
    print(ress)
    # learner.calc_img_feas(out='work_space/casia.r50.arc.h5')
    # exit(0)

    # learner.init_lr()
    # conf.tri_wei = 0
    # log_conf(conf)
    # learner.train(conf, 1, name='xent')

    learner.init_lr()
    log_conf(conf)
    # learner.warmup(conf, conf.warmup)
    # learner.train(conf, conf.epochs)
    # learner.train_dist(conf, conf.epochs)
    # learner.train_simple(conf, conf.epochs)
    # learner.train_cotching(conf, conf.epochs)
    learner.train_cotching_accbs(conf, conf.epochs)
    # learner.train_ghm(conf, conf.epochs)
    # learner.train_with_wei(conf, conf.epochs)
    # learner.train_use_test(conf, conf.epochs)
    res = learner.validate_ori(conf)

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

    # log_lrs, losses = learner.find_lr(conf,
    #                                   # final_value=100,
    #                                   num=200,
    #                                   bloding_scale=1000)
    # best_lr = 10 ** (log_lrs[np.argmin(losses)])
    # print(best_lr)
    # conf.lr = best_lr
    # learner.push2redis()
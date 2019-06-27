#!/usr/bin/env python3
# -*- coding: future_fstrings -*-
import torch
from lz import *
from config import conf
import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True


def log_conf(conf):
    conf2 = {k: v for k, v in conf.items() if not isinstance(v, (dict, np.ndarray))}
    logging.info(f'training conf is {conf2}')


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--local_rank', default=None, type=int, )
parser.add_argument('--mbfc_wm', default=conf.mbfc_wm, type=float, )
parser.add_argument('--mbfc_dm', default=conf.mbfc_dm, type=float, )
parser.add_argument('--work_path', default=None, type=str, )
parser.add_argument('--epochs', default=conf.epochs, type=int, )

if __name__ == '__main__':
    args = parser.parse_args()
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

    # exit()
    learner = face_learner(conf, )
    # learner = face_cotching(conf, )
    ress = {}
    for p in [
        # 'emore.r50.dop',
        # 'emore.r152.ada.chkpnt',
        # 'emore.r152.ada.chkpnt.2',
        # 'emore.r152.ada.chkpnt.3',
        # 'retina.r50',
        # 'hrnet.retina.arc.3',
        # 'mbv3.retina.arc',
        # 'mbfc.lrg.retina.arc.s48',
        # 'effnet.casia.arc',
    ]:
        learner.load_state(
            # fixed_str='2019-04-06-20_accuracy:0.707857142857143_step:2268_None.pth',
            resume_path=Path(f'work_space/{p}/models/'),
            load_optimizer=True,
            load_head=True,  # todo note!
            load_imp=False,
            latest=True,
        )
        # res = learner.validate_ori(conf)
        # ress[p] = res
        # logging.warning(f'{p} res: {res}')
    print(ress)

    # ttl_params = (sum(p.numel() for p in learner.model.parameters()) / 1000000.0)
    # from thop import profile
    #
    # flops, params = profile(learner.model.module,
    #                         input_size=(1, 3, conf.input_size, conf.input_size),
    #                         only_ops=(nn.Conv2d, nn.Linear),
    #                         device='cuda:0',
    #                         )
    # flops /= 10 ** 9
    # params /= 10 ** 6
    # print('Total params: %.2fM' % ttl_params, flops,'\n',
    #       conf.input_size, conf.mbfc_wm, conf.mbfc_dm)
    # exit()

    # from tools.test_ijbc3 import test_ijbc3
    # res = test_ijbc3(conf, learner)

    # learner.calc_img_feas(out='work_space/retina.hrnet.h5')

    # log_lrs, losses = learner.find_lr(
    #                                   num=999,
    #                                   bloding_scale=1000)
    # losses[np.isnan(losses)] = 999
    # best_lr = 10 ** (log_lrs[np.argmin(losses)])
    # print('best lr is ', best_lr)
    # conf.lr = best_lr
    # exit(0)

    # learner.init_lr()
    # conf.tri_wei = 0
    # log_conf(conf)
    # learner.train(conf, 1, name='xent')

    learner.init_lr()
    log_conf(conf)
    if conf.warmup:
        learner.warmup(conf, conf.warmup)
    # learner.train(conf, conf.epochs)
    # learner.train_dist(conf, conf.epochs)
    learner.train_simple(conf, conf.epochs)
    # learner.train_cotching(conf, conf.epochs)
    # learner.train_cotching_accbs(conf, conf.epochs)
    # learner.train_ghm(conf, conf.epochs)
    # learner.train_with_wei(conf, conf.epochs)
    # learner.train_use_test(conf, conf.epochs)
    # res = learner.validate_ori(conf)

    from tools.test_ijbc3 import test_ijbc3

    res = test_ijbc3(conf, learner)

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

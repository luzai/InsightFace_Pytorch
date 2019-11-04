#!/usr/bin/env python3

print('------------------------')
import os, time

try:
    import install
    from lz import *

    try:
        import moxing, moxing.pytorch as mox

        moxing.file.shift('os', 'mox')
    except:
        pass
    import torch
    from config import conf
    from pathlib import Path
    from exargs import parser

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    try:
        import mox_patch_0603_v4
        import moxing.pytorch as mox
    except:
        logging.warning('not in the cloud')
        conf.cloud = False
except Exception as f:
    print(f)
    time.sleep(600)


def main():
    args = parser.parse_args()
    print('args.data_url', args.data_url)
    if conf.cloud:
        mox.file.copy_parallel(args.data_url, '/cache/face_train/')
        args.data_url = '/cache/face_train/'
        conf.use_data_folder = args.data_url

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
    # if osp.exists(conf.save_path):
    #     logging.info('ok')
    #     exit(1)
    # simplify_conf(conf)
    # exit(0)
    from Learner import face_learner
    # decs = msgpack_load('decs.pk')
    # conf.decs = decs
    learner = face_learner(conf, )
    # fstrs = learner.list_fixed_strs('work_space/sglpth.casia/models')
    # stps = learner.list_steps('work_space/sglpth.casia/models')
    # fstr = fstrs[np.argmax(stps)]
    # stt_dct = torch.load('work_space/sglpth.casia/models/model_' + fstr)
    # learner.model.module.load_state_dict_sglpth(stt_dct)
    # print(fstrs, stps, fstr, )

    if conf.get('load_from'):
        # p= 'r100.128.retina.clean.arc',
        # 'hrnet.retina.arc.3',
        # 'mbv3.retina.arc',
        # 'mbfc.lrg.retina.arc.s48',
        # 'effnet.casia.arc',
        # 'mbfc.retina.cl.distill.cont2',
        # 'mbfc2',
        # 'r18.l2sft',
        # 'r18.adamrg',
        # 'mbfc.se.elu.ms1m.radam.1',
        # 'mbfc.se.elu.specnrm.allbutdw.ms1m.adam.1',
        # 'mbfc.se.prelu.specnrm.ms1m.cesigsft.1',
        # 'irse.elu.ms1m',
        # 'irse.elu.casia.arc.2048',
        p = Path(conf.load_from)
        print('try to load from ', p, )
        learner.load_state(
            resume_path=p,
            load_optimizer=False,
            load_head=conf.head_load,  # todo note!
            load_imp=False,
            latest=True, strict=False,
        )
    # simplify_conf(conf)
    learner.cloud_sync_log()
    # res = learner.validate_ori(conf, valds_names=('cfp_fp', ))
    # exit(0)
    # learner.calc_img_feas(out='work_space/mbfc.crash.h5')
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
    simplify_conf(conf)
    if conf.head_init:
        learner.head_initialize()
    if conf.warmup:
        learner.warmup(conf, conf.warmup)
    learner.train_simple(conf, conf.epochs)

    # learner.train_dist(conf, conf.epochs)
    if conf.net_mode == 'sglpth':
        decs = learner.model.module.get_decisions()
        msgpack_dump(decs, 'decs.pk')

    # learner.train_cotching(conf, conf.epochs)
    # learner.train_cotching_accbs(conf, conf.epochs)
    # learner.train_ghm(conf, conf.epochs)
    # learner.train_with_wei(conf, conf.epochs)
    # learner.train_use_test(conf, conf.epochs)

    # res = learner.validate_ori(conf, )
    if not conf.cloud:
        from tools.test_ijbc3 import test_ijbc3
        res = test_ijbc3(conf, learner)
        tpr6, tpr4, tpr3 = res[0][1], res[1][1], res[2][1]
        learner.writer.add_scalar('ijbb/6', tpr6, learner.step)
        learner.writer.add_scalar('ijbb/4', tpr4, learner.step)
        learner.writer.add_scalar('ijbb/3', tpr3, learner.step)
    learner.writer.close()

    if conf.never_stop:
        img = torch.randn((conf.batch_size // 2, 3, conf.input_size, conf.input_size)).cuda()
        learner.model.eval()
        logging.info('never stop')
        while True:
            _ = learner.model(img)


if __name__ == '__main__':
    # try:
    main()
# except Exception as f:
#     print(f)
#     time.sleep(600)

from lz import *
from config import get_config, gl_conf
from Learner import face_learner
import argparse
from pathlib import Path

if __name__ == '__main__':
    # todo move args to config
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se',
                        type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]", default='emore',
                        type=str)
    parser.set_defaults(
        net='ir_se',
        net_depth='50',
        data_mode="ms1m",
    )
    # todo shoter epoch performance
    # todo find best lr and test it in several epoch
    args = parser.parse_args()

    conf = get_config()

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth

    conf.data_mode = args.data_mode
    learner = face_learner(conf, )
    ## for resume or evaluate
    learner.load_state(conf,
                       resume_path= Path('work_space/arcsft.triadap.s64.0.1/save'),
                       from_save_folder=False,
                       model_only=False,
                       )
    # log_lrs, losses = learner.find_lr(conf,
    #                                   # final_value=100,
    #                                   num=200)
    # best_lr = 10 ** (log_lrs[np.argmin(losses)])
    # print(best_lr)
    # conf.lr = best_lr
    learner.init_lr()
    learner.train(conf, conf.epochs)

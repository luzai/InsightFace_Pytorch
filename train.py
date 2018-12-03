from lz import *
from config import get_config, gl_conf
from Learner import face_learner
import argparse

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    # todo move args to config
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se',
                        type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]", default='emore',
                        type=str)
    parser.set_defaults(
        epochs=8,
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
    # learner.load_state(conf,
    #                    '2018-11-26-09-37_accuracy:0.8048571428571428_step:30730_None.pth',
    #                    from_save_folder=False,
    #                    model_only=False)
    # todo make it load from model of any folder
    log_lrs, losses = learner.find_lr(conf,
                                      final_value=10,
                                      num=1500)
    best_lr = 10 ** (log_lrs[np.argmin(losses)])
    # conf.lr = best_lr
    # learner.train(conf, args.epochs)

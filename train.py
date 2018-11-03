import lz
from config import get_config
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
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]", default='emore',
                        type=str)
    parser.set_defaults(
        epochs=8, # 4 epoch for test2 performance
        net='ir_se',
        net_depth='50',
        lr=0.028,  # 0.028 , 1e-2
        batch_size=100 if not lz.dbg else 8,
        num_workers=18 if not lz.dbg else 0,
        data_mode="ms1m",
    )
    # todo make dbg useful again
    # todo find best lr and test it in several epoch
    # todo test the effect of batch size in several epoch
    args = parser.parse_args()

    conf = get_config()

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth

    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    learner = face_learner(conf)
    # learner.load_state(conf,'2018-10-24-12-02_accuracy:0.876_step:304456_final.pth', True, True )
    # print(learner.find_lr(conf, num=1500))
    # todo start epoch for resume
    learner.train(conf, args.epochs)

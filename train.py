from lz import *
from config import get_config, gl_conf
from Learner import face_learner
import argparse
from pathlib import Path

if __name__ == '__main__':
    conf = get_config()
    learner = face_learner(conf, )
    
    ## for resume or evaluate
    # learner.load_state(conf,
    #                    # resume_path=Path('work_space/arcsft.triadap.dop.long/save'),
    #                    # model_only=True,
    #                    # load_optimizer=False,
    #                    # resume_path=Path('work_space/glint/models'),
    #                    resume_path=Path('work_space/glint.bs/models'),
    #                    model_only=False,
    #                    load_optimizer=True,
    #                    latest=True,
    #                    )
    
    # learner.save()
    # log_lrs, losses = learner.find_lr(conf,
    #                                   # final_value=100,
    #                                   num=200,
    #                                   bloding_scale=1000)
    # best_lr = 10 ** (log_lrs[np.argmin(losses)])
    # print(best_lr)
    # conf.lr = best_lr
    
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
    
    # learner.calc_feature()

import os, sys

sys.path.insert(0, '/home/xinglu/prj/InsightFace_Pytorch/')


def alignface(img1, mtcnn):
    assert mtcnn is not None
    img1 = Image.fromarray(img1)
    face1 = mtcnn.align_best(img1, conf.face_limit, 16)
    face1 = np.asarray(face1)
    return face1


def do_align_by_list(inps):
    import logging
    ind, tid, sid, fn, x, y, w, h, mtcnn = inps
    logging.info(f'{ind} start')
    x, y, w, h = list(map(int, [x, y, w, h]))
    imgp = img_path + fn
    assert osp.exists(imgp), imgp
    img = cvb.read_img(imgp)
    face = img[y:y + h, x:x + w, :]
    face_ali = alignface(face, mtcnn)
    dst_dir = f'{dst}/{sid}/{tid}'
    dst_fn = f'{dst}/{sid}/{tid}/{ind}.png'
    _ = mkdir_p(dst_dir, delete=False)
    _ = cvb.write_img(face_ali, dst_fn)


if __name__ == '__main__':
    from torch.multiprocessing import Pool, Process, set_start_method

    set_start_method('spawn')

    from skimage import io
    from lz import *
    import lz
    import cv2
    from PIL import Image
    import argparse
    from pathlib import Path
    from config import get_config
    from Learner import face_learner
    from utils import load_facebank, draw_box_name, prepare_facebank
    from mtcnn import MTCNN
    from Learner import face_learner
    from model import l2_norm

    ijb_path = '/data2/share/ijbc/'
    test1_path = ijb_path + '/IJB-C/protocols/test1/'
    img_path = ijb_path + 'IJB/IJB-C/images/'
    df_enroll = pd.read_csv(test1_path + '/enroll_templates.csv')
    df_verif = pd.read_csv(test1_path + '/verif_templates.csv')
    df_match = pd.read_csv(test1_path + '/match.csv')
    dst = ijb_path + '/ijb.test1.proc/'
    conf = get_config(training=False, )

    mtcnn = MTCNN()
    mtcnn.share_memory()


    def do_align_one(ind, val, ):
        tid = val['TEMPLATE_ID']
        sid = val['SUBJECT_ID']
        fn = val['FILENAME']
        dst_dir = f'{dst}/{sid}/{tid}'
        dst_fn = f'{dst}/{sid}/{tid}/{ind}.png'
        #     if osp.exists(dst_fn): continue
        x, y, w, h = val.iloc[-4:]
        x, y, w, h = list(map(int, [x, y, w, h]))
        imgp = img_path + fn
        assert osp.exists(imgp), imgp
        img = cvb.read_img(imgp)
        assert img is not None, 'impg'
        face = img[y:y + h, x:x + w, :]
        face_ali = alignface(face, mtcnn)
        _ = mkdir_p(dst_dir, delete=False)
        _ = cvb.write_img(face_ali, dst_fn)


    def do_align_slow():
        meter = lz.AverageMeter()
        lz.timer.since_last_check('start')
        # for ind, val in df_enroll.iterrows():
        #     do_align_one(ind, val)
        #     meter.update(lz.timer.since_last_check(verbose=False))
        #     if ind % 100 == 0:
        #         print(ind, meter.avg)
        # if ind > 199: break

        lz.timer.since_last_check('start')
        for ind, val in df_verif.iterrows():
            do_align_one(ind, val)
            meter.update(lz.timer.since_last_check(verbose=False))
            if ind % 100 == 0:
                print(ind, meter.avg)
            # if ind > 199: break


    do_align_slow()

    # multi_pool = Pool(processes=12, )
    # inps = []
    # for ind, val in df_enroll.iterrows():
    #     tid = val['TEMPLATE_ID']
    #     sid = val['SUBJECT_ID']
    #     fn = val['FILENAME']
    #     x, y, w, h = val.iloc[-4:]
    #     x, y, w, h = list(map(int, [x, y, w, h]))
    #     inps.append([ind, tid, sid, fn, x, y, w, h, mtcnn])
    #     if ind > 9999: break
    # # _ = multi_pool.map(do_align_by_list, inps)
    # _ = multi_pool.map_async(do_align_by_list, inps)
    # multi_pool.close()
    # multi_pool.join()

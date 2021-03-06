import sys, pandas as pd
from PIL import Image

ijb_path = '/data2/share/ijbc/'
test1_path = ijb_path + '/IJB-C/protocols/test1/'
img_path = ijb_path + 'IJB/IJB-C/images/'
df_enroll = pd.read_csv(test1_path + '/enroll_templates.csv')
df_verif = pd.read_csv(test1_path + '/verif_templates.csv')
df_match = pd.read_csv(test1_path + '/match.csv')
dst = ijb_path + '/ijb.test1.proc/'


def alignface(img1, mtcnn, img):
    assert mtcnn is not None
    img1 = Image.fromarray(img1)
    face1 = mtcnn.align_best(img1, limit=10, min_face_size=16, ori_img=img)
    face1 = np.asarray(face1)
    return face1


def do_align_by_list(inps):
    from lz import mkdir_p, cvb
    ind, tid, sid, fn, x, y, w, h, mtcnn = inps
    dst_dir = f'{dst}/{sid}/{tid}'
    dst_fn = f'{dst}/{sid}/{tid}/{ind}.png'
    if osp.exists(dst_fn): return
    
    # logging.info(f'{ind} start')
    x, y, w, h = list(map(int, [x, y, w, h]))
    imgp = img_path + fn
    assert osp.exists(imgp), imgp
    img = cvb.read_img(imgp)
    face = img[y:y + h, x:x + w, :]
    face_ali = alignface(face, mtcnn)
    
    _ = mkdir_p(dst_dir, delete=False)
    _ = cvb.write_img(face_ali, dst_fn)


if __name__ == '__main__':
    from torch.multiprocessing import set_start_method
    
    set_start_method('spawn')
    
    from lz import *
    import lz
    from mtcnn import MTCNN

    from config import get_config
    
    conf = get_config(training=False, )
    
    mtcnn = MTCNN()
    mtcnn.share_memory()
    
    
    def do_align_one(ind, val, ):
        tid = val['TEMPLATE_ID']
        sid = val['SUBJECT_ID']
        fn = val['FILENAME']
        dst_dir = f'{dst}/{sid}/{tid}'
        dst_fn = f'{dst}/{sid}/{tid}/{ind}.png'
        # if osp.exists(dst_fn): return
        x, y, w, h = val.iloc[-4:]
        x, y, w, h = list(map(int, [x, y, w, h]))
        imgp = img_path + fn
        assert osp.exists(imgp), imgp
        img = cvb.read_img(imgp)
        assert img is not None, 'impg'
        face = img[y:y + h, x:x + w, :]
        face_ali = alignface(face, mtcnn, img)
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
        # for ind, val in df_verif.iterrows():
        for ind, val in df_enroll.iterrows():
            # if ind < 1445: continue
            do_align_one(ind, val)
            meter.update(lz.timer.since_last_check(verbose=False))
            if ind % 100 == 0:
                print(ind, meter.avg)
            # if ind > 199: break
    
    
    do_align_slow()
    
    # multi_pool = Pool(processes=3, )
    # inps = []
    # # for ind, val in df_enroll.iterrows():
    # for ind, val in df_verif.iterrows():
    #     tid = val['TEMPLATE_ID']
    #     sid = val['SUBJECT_ID']
    #     fn = val['FILENAME']
    #     x, y, w, h = val.iloc[-4:]
    #     x, y, w, h = list(map(int, [x, y, w, h]))
    #     inps.append([ind, tid, sid, fn, x, y, w, h, mtcnn])
    # # inps = inps[::-1]
    # # random.shuffle(inps)
    # _ = multi_pool.map(do_align_by_list, inps)
    # # _ = multi_pool.map_async(do_align_by_list, inps)
    # multi_pool.close()
    # multi_pool.join()

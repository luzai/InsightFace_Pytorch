import sys

sys.path.insert(0, '/data1/xinglu/prj/InsightFace_Pytorch')
from lz import *
import lz
from torchvision import transforms as trans
import redis
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--modelp', default='mbv3.small',
                    type=str)
args = parser.parse_args()
os.chdir(lz.root_path)
lz.init_dev()
use_mxnet = False
bs = 512


def evaluate_ori(model, path, name, nrof_folds=10, tta=True):
    from utils import ccrop_batch, hflip_batch
    from models.model import l2_norm
    from verifacation import evaluate
    idx = 0
    from data.data_pipe import get_val_pair
    carray, issame = get_val_pair(path, name)
    carray = carray[:, ::-1, :, :]  # BGR 2 RGB!
    if use_mxnet:
        carray *= 0.5
        carray += 0.5
        carray *= 255.
    embeddings = np.zeros([len(carray), 512])
    if not use_mxnet:
        with torch.no_grad():
            while idx + bs <= len(carray):
                batch = torch.tensor(carray[idx:idx + bs])
                if tta:
                    # batch = ccrop_batch(batch)
                    fliped = hflip_batch(batch)
                    emb_batch = model(batch.cuda()) + model(fliped.cuda())
                    emb_batch = emb_batch.cpu()
                    embeddings[idx:idx + bs] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + bs] = model(batch.cuda()).cpu()
                idx += bs
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    # batch = ccrop_batch(batch)
                    fliped = hflip_batch(batch)
                    emb_batch = model(batch.cuda()) + model(fliped.cuda())
                    emb_batch = emb_batch.cpu()
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = model(batch.cuda()).cpu()
    else:
        from sklearn.preprocessing import normalize
        while idx + bs <= len(carray):
            batch = torch.tensor(carray[idx:idx + bs])
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch) + model(fliped)
                embeddings[idx:idx + bs] = normalize(emb_batch)
            else:
                embeddings[idx:idx + bs] = model(batch)
            idx += bs
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch) + model(fliped)
                embeddings[idx:] = normalize(emb_batch)
            else:
                embeddings[idx:] = model(batch)
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    roc_curve_tensor = None
    # buf = gen_plot(fpr, tpr)
    # roc_curve = Image.open(buf)
    # roc_curve_tensor = trans.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


if use_mxnet:
    from recognition.embedding import Embedding

    learner = Embedding(
        prefix='/home/xinglu/prj/insightface/logs/MS1MV2-ResNet100-Arcface/model',
        epoch=0,
        ctx_id=0)
else:
    from config import conf

    conf.need_log = False
    conf.batch_size *= 4 * conf.num_devs
    # bs = min(conf.batch_size, bs)
    conf.fp16 = False
    conf.ipabn = False
    conf.cvt_ipabn = False
    conf.fill_cache = False
    # conf.net_depth = 152
    # conf.net_mode = 'mobilefacenet'
    conf.use_chkpnt = False
    from Learner import FaceInfer, face_learner

    # learner = FaceInfer(conf, gpuid=range(conf.num_devs))
    # learner.load_state(
    #     resume_path=f'work_space/{args.modelp}/models/',
    #     latest=False,
    # )
    # learner.model.eval()

    learner = face_learner()
    learner.load_state(
        resume_path=f'work_space/{args.modelp}/models/', latest=False,
        load_optimizer=False, load_imp=False, load_head=False,
    )
    learner.model.eval()

from pathlib import Path

res = {}
for ds in ['agedb_30', 'lfw', 'cfp_fp', 'cfp_ff', 'calfw', 'cplfw', 'vgg2_fp', ]:
    if use_mxnet:
        m_ = learner
    else:
        m_ = learner.model
    accuracy, best_threshold, roc_curve_tensor = evaluate_ori(m_,
                                                              Path('/data2/share/faces_emore'),
                                                              ds)
    logging.info(f'validation accuracy on {ds} is {accuracy} ')
    res[ds] = accuracy

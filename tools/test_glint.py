import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms as trans
from lz import *
import mxnet as mx

use_mxnet = True
if not use_mxnet:
    from config import conf
    
    conf.need_log = False
    conf.batch_size *= 4
    conf.fp16 = False
    conf.ipabn = False
    conf.cvt_ipabn = True
    # conf.upgrade_irse = False
    # conf.net_mode = 'ir'
    conf.net_depth = 50
    from Learner import l2_norm, FaceInfer, get_rec, unpack_auto
    
    learner = FaceInfer(conf, )
    learner.load_state(
        resume_path='work_space/asia.emore.r50.ada/models/',
        latest=True,
    )
    # learner.load_model_only('work_space/backbone_ir50_ms1m_epoch120.pth')
    # learner.load_model_only('work_space/backbone_ir50_asia.pth')
    learner.model.eval()
else:
    from config import conf
    from recognition.embedding import Embedding
    from Learner import l2_norm, FaceInfer, get_rec, unpack_auto
    
    learner = Embedding(
        prefix='/home/xinglu/prj/insightface/logs/MS1MV2-ResNet100-Arcface/model', epoch=0, ctx_id=0)
logging.info('learner loaded')

ms1m_path = '/data2/share/'
ms1m_lmk_path = '/data1/share/testdata_lmk.txt'
lmks = pd.read_csv(ms1m_lmk_path, sep=' ', header=None).to_records(index=False).tolist()
logging.info(f'len lmks {len(lmks)}')
rec_test = get_rec('/data2/share/glint_test/train.idx')
use_rec = True
from sklearn.preprocessing import normalize


class DatasetMS1M3(torch.utils.data.Dataset):
    def __init__(self, use_rec=use_rec):
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.use_rec = use_rec
    
    def __getitem__(self, item):
        if not self.use_rec:
            data = lmks[item]
            imgfn = data[0]
            lmk = np.asarray(data[1:])
            img = cvb.read_img(f'{ms1m_path}/{imgfn}')  # bgr
            img = cvb.bgr2rgb(img)  # rgb
            warp_img = preprocess(img, landmark=lmk)
        else:
            rec_test.lock.acquire()
            s = rec_test.imgrec.read_idx(item + 1)
            rec_test.lock.release()
            header, img = unpack_auto(s, 'glint_test')
            img = mx.image.imdecode(img).asnumpy()  # rgb
            warp_img = np.array(img, dtype=np.uint8)
        # plt_imshow(img) ; plt.show()
        # plt_imshow(warp_img); plt.show()
        warp_img = Image.fromarray(warp_img)
        flip_img = torchvision.transforms.functional.hflip(warp_img)
        if use_mxnet:
            warp_img = np.array(np.transpose(warp_img, (2, 0, 1)))
            warp_img = to_torch(warp_img).float()
            flip_img = np.array(np.transpose(flip_img, (2, 0, 1)))
            flip_img = to_torch(flip_img).float()
        else:
            warp_img = self.test_transform(warp_img)
            flip_img = self.test_transform(flip_img)
        return {'img': warp_img,
                'flip_img': flip_img, }
    
    def __len__(self):
        return len(lmks)


def img2db(name):
    ds = DatasetMS1M3()
    bs = 128
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=12, shuffle=False, pin_memory=True)
    
    # ds2 = DatasetMS1M3(use_rec=False)
    # loader2 = torch.utils.data.DataLoader(ds2, batch_size=bs, num_workers=0, shuffle=False, pin_memory=True)
    #
    # for ind, (d1, d2) in enumerate(zip(loader, loader2)):
    #     im1 = d1["img"]
    #     im1*=127
    #     im1+=127
    #     im1 = im1.long()
    #     im2 = d2["img"]
    #     im2*=127
    #     im2+=127
    #     im2=im2.long()
    #     diff = (im1-im2).numpy()
    #     assert np.allclose(im1.numpy(), im2.numpy())
    #     data=d2
    #     if ind > 0:
    #         break
    
    db = Database(work_path + name + '.h5', )
    for ind, data in enumerate(loader):
        if ind % 9 == 0:
            logging.info(f'ind {ind}/{len(loader)}')
        warp_img = data['img'].cuda()
        flip_img = data['flip_img'].cuda()
        if not use_mxnet:
            with torch.no_grad():
                fea = learner.model(warp_img) + learner.model(flip_img)
                fea = fea.cpu().numpy()
                fea = normalize(fea)
        else:
            warp_img = data['img'].cpu().numpy()
            flip_img = data['flip_img'].cpu().numpy()
            img_feat = learner.gets(warp_img)
            img_feat += learner.gets(flip_img)
            fea = img_feat
            fea = normalize(fea)
        
        db[f'{ind}'] = fea
    
    db.close()


def db2np(name):
    db = Database(work_path + name + '.h5', 'r')
    res = np.empty((len(lmks), 512), dtype=np.float32)
    for ind in db.keys():
        bs = db[ind].shape[0]
        iind = int(ind)
        res[iind * bs: iind * bs + bs, :] = db[ind]
    db.close()
    
    save_mat(work_path + name + '.bin', res)
    msgpack_dump(res, work_path + name + '.pk', )


if __name__ == '__main__':
    name = 'mxnet'
    img2db(name)
    db2np(name)

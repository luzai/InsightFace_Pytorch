from lz import *
import lz
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

import lmdb

inputp = '/data/share/iccv19.lwface/iQIYI-VID-FACE/'
env = lmdb.open(inputp + '/imgs_lmdb', readonly=True, )
lines = open(os.path.join(inputp, 'filelist.txt'), 'r').readlines()
lines = [line.strip().split(' ')[0] for line in lines]
df = pd.read_csv(inputp + '/filelist.txt', header=None, sep=' ')

allids = lz.msgpack_load(inputp + '../allids.pk')
nimgs = lz.msgpack_load(inputp + '../nimgs.pk')
allimgs = lz.msgpack_load(inputp + '../allimgs.pk')
vdonm2imgs = lz.msgpack_load(inputp + '../vdonm2imgs.pk')
lines = np.asarray(lines)
allids = np.asarray(allids)
nimgs = np.asarray(nimgs)
allimgs = np.asarray(allimgs)

import six
from PIL import Image


def get_iccv_vdo(item):
    item = item.replace(inputp, '').strip('/')
    item = '/' + item
    with env.begin(write=False) as txn:
        imgbuf = txn.get(str(item).encode())
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    f = Image.open(buf)
    img = f.convert('RGB')
    img = np.asarray(img)
    return img


def get_iccv_vdos_by_ind(inds):
    vnms = allimgs[inds]
    imgs = [get_iccv_vdo(vnm) for vnm in vnms]
    return np.asarray(imgs)


def get_iccv_vdo_by_vnm(item, limits=None, shuffle=False):
    imgs = vdonm2imgs[item]
    imgs = list(imgs)
    if shuffle:
        random.shuffle(imgs)
    if limits is not None:
        imgs = imgs[:limits]
    imgs = [get_iccv_vdo(img) for img in imgs]
    return imgs


def cvt_vdonm2inds(vdonm):
    ind = np.where(lines == vdonm)[0][0]
    inds_start = nimgs[:ind].sum()
    inds_end = nimgs[:ind + 1].sum()
    return np.arange(inds_start, inds_end)


# imgs = get_iccv_vdo_by_vnm(lines[0])
# plt_imshow_tensor(imgs)
# inds = cvt_vdonm2inds(vdonm)

import h5py

feas = h5py.File('/home/xinglu/work/r100.unrm.allimg.h5', 'r')['feas'][...]
endind = np.where(feas.sum(axis=-1) == 0)[0][0]
feas = feas[:endind, :]

feas = feas.astype('float16')

len(allimgs)
from sklearn.preprocessing import normalize

feas_nrmed = normalize(feas / 10, )

norms = np.linalg.norm(feas, axis=-1)


def markmethod(seen):
    mapto = {}
    for s, e in seen:
        if s in mapto:
            mapto[e] = mapto[s]
        elif e in mapto:
            mapto[s] = mapto[e]
        else:
            mapto[s] = mapto[e] = len(mapto)
    mapback = collections.defaultdict(list)
    for uid, oid in mapto.items():
        mapback[oid].append(uid)
    #     print(len(seen), len(mapto), len(mapback) )
    return (list(mapback.values()))


nimg_thresh = 4
ratio = 10
nrm_times = '3nrm'


def reduce(feas_nrmedt):
    distmat = cdist(feas_nrmedt, feas_nrmedt)
    rows, cols = np.unravel_index(np.where(distmat.flatten() < 0.1)[0], distmat.shape)
    seen = set(zip(rows, cols))
    mapback = markmethod(seen)
    return mapback


def denoise(norms_rdc, feas_nrmed_rdc):
    alolinds = []
    q1 = np.quantile(norms_rdc, .25)
    q3 = np.quantile(norms_rdc, .75)
    iqr = q3 - q1
    olinds_nrm_sml = np.where(norms_rdc < q1 - ratio * iqr)[0].tolist()
    olinds_nrm_lrg = []  # np.where(norms_rdc>q3+ratio*iqr)[0].tolist()
    alolinds.extend(olinds_nrm_sml + olinds_nrm_lrg)

    fcent = feas_nrmed_rdc.mean(axis=0, keepdims=True)
    distmat = cdist(fcent, feas_nrmed_rdc).flatten()
    q1 = np.quantile(distmat, .25)
    q3 = np.quantile(distmat, .75)
    iqr = q3 - q1
    olinds_dst_sml = []  # np.where(distmat<q1-ratio*iqr)[0].tolist()
    olinds_dst_lrg = np.where(distmat > q3 + ratio * iqr)[0].tolist()
    alolinds.extend(olinds_dst_lrg + olinds_dst_sml)

    alolinds = np.unique(alolinds)
    # keepinds = np.setdiff1d(np.arange(len(feas_nrmed_rdc)),
    #                         alolinds)
    return alolinds


nol = 0
final_feas = np.empty((len(nimgs), 512), dtype='float32')
for ii in range(len(nimgs)):
    if ii % 9999 == 0:
        print('proc', ii, len(nimgs), )
    vdonm = lines[ii]
    #     imgst = get_iccv_vdo_by_vnm(vdonm)
    #     imgst=np.asarray(imgst)
    inds = cvt_vdonm2inds(vdonm)
    norms_t = norms[inds]
    feas_nrmed_t = feas_nrmed[inds, :]
    feas_t = feas[inds, :]
    # plt_imshow_tensor(imgst[:40])
    if len(inds) > nimg_thresh:
        alolinds = denoise(norms_t, feas_nrmed_t)
        nol += alolinds.shape[0]
        keepinds = np.setdiff1d(np.arange(len(feas_t)),
                                alolinds)
        feas_t = feas_t[keepinds, :]
        feas_nrmed_t = feas_nrmed_t[keepinds, :]
        norms_t = norms_t[keepinds]

        mapback = reduce(feas_nrmed_t)
        norms_rdc = []
        feas_rdc = []
        feas_nrmed_rdc = []
        for inds_ in mapback:
            norms_rdc.append(norms_t[inds_].mean(axis=0))
            feas_rdc.append(feas_t[inds_, ...].mean(axis=0))
            # feas_nrmed_rdc.append(feas_nrmed_t[inds_, ...].mean(axis=0))
            feas_nrmed_rdc.append(
                normalize(feas_nrmed_t[inds_, ...].mean(axis=0, keepdims=True)).flatten()
            )

        norms_t = np.asarray(norms_rdc)
        feas_t = np.asarray(feas_rdc)
        feas_nrmed_t = np.asarray(feas_nrmed_rdc)

    #         if len(olinds_nrm_sml)>0:
    #             break

    # feat = feas_t.mean(axis=0, keepdims=True)
    feat = feas_nrmed_t.mean(axis=0, keepdims=True)
    feat = normalize(feat)
    final_feas[ii, :] = feat.flatten()
#     break 
lz.save_mat('/home/xinglu/work/r100.vdo.t6.bin', final_feas)

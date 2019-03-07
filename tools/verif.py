#!/usr/bin/env python
from lz import *
import numpy as np
import lz
from config import conf
from tools.FaceVerification import FaceVerification as verif

os.chdir(lz.root_path)
# To see the verification example, change the paths below to your own,
# and change the path in `same_pairs.txt` and `diff_pairs.txt` to your own.
path_pairs = "images_aligned_2018Autumn/pairs1_nolabel.txt"


def main():
    same_pairs = np.loadtxt(path_pairs, dtype="str", delimiter="  ")
    results = []
    dists = []
    cnt_s = 0
    for sp in same_pairs:
        cnt_s += 1
        # pred, dist = verif(f'images_aligned_2018Autumn/{sp[0]}', f'images_aligned_2018Autumn/{sp[1]}')
        fea1 = lz.root_path + f'../images_aligned_2018Autumn_OPPOFeatures/{sp[0]}_OPPO.bin'
        fea2 = lz.root_path + f'../images_aligned_2018Autumn_OPPOFeatures/{sp[1]}_OPPO.bin'
        fea1 = load_mat(fea1)
        fea2 = load_mat(fea2)
        dist = ((fea1 - fea2) ** 2).sum()
        pred = dist < 1.5
        results.append(pred)
        dists.append(dist)
        print(cnt_s, sp[0], sp[1], pred, sep="  ")
    
    results = np.array(results)
    dists = np.array(dists)
    thresh = np.median(dists)
    thresh = 1.5
    print('chs thresh', thresh)
    results[dists > thresh] = 0
    results[dists <= thresh] = 1
    pred = results
    gt = open('images_aligned_2018Autumn/pairs1.txt').readlines()
    gt = [g.strip('\n') for g in gt]
    gts = [g.split('  ') for g in gt]
    gt = [g[-1][-1] for g in gts]
    pred = np.array(pred, dtype=int)
    gt = np.array(gt, int)
    print((gt == pred).sum() / gt.shape[0])
    
    np.savetxt('21831128-11831037+pairs1.txt', np.array(results), fmt='%d')
    plt.plot(np.sort(dists))
    plt.show()
    plt.hist(dists, bins=50)
    plt.show()


if __name__ == "__main__":
    main()

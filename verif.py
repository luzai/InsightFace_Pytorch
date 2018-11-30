#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import sys
import os
from lz import *

GPU_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_id)

# if use pycaffe
# caffe_root = "/home/wanghuan/Caffe/Caffe_default/python" # change this to your own pycaffe path
# sys.path.insert(0, caffe_root)
# import caffe
# caffe.set_mode_gpu()
# caffe.set_device(GPU_id)

from FaceVerification import FaceVerification as verif

# To see the verification example, change the paths below to your own,
# and change the path in `same_pairs.txt` and `diff_pairs.txt` to your own.
path_same_pairs = "images_aligned_sample/same_pairs.txt"
path_diff_pairs = "images_aligned_sample/diff_pairs.txt"


def main():
    same_pairs = np.loadtxt(path_same_pairs, dtype="str", delimiter="  ")
    diff_pairs = np.loadtxt(path_diff_pairs, dtype="str", delimiter="  ")
    result_same = []
    result_diff = []
    num_same = len(same_pairs)
    num_diff = len(diff_pairs)

    cnt_s = 0
    for sp in same_pairs:
        try:
            pred = verif(sp[0], sp[1])
            result_same.append(pred)
            cnt_s += 1
            print("same", cnt_s, sp[0], sp[1], pred, sep="  ")
        except:
            logging.error("Sth wrong, continue loop ..")
            continue

    cnt_d = 0
    for dp in diff_pairs:
        try:
            pred = (verif(dp[0], dp[1]))
            result_diff.append((verif(dp[0], dp[1])))
            cnt_d += 1
            print("diff", cnt_d, dp[0], dp[1], pred, sep="  ")
        except:
            logging.error("Sth wrong, continue loop ..")
            continue

    num_right_same = cnt_s - np.logical_xor(result_same, [1] * cnt_s).sum()
    num_right_diff = cnt_d - np.logical_xor(result_diff, [0] * cnt_d).sum()
    print("{0} pairs got right in {1} same pairs, {2} pairs got right in {3} diff pairs".format(
        num_right_same, cnt_s, num_right_diff,
        cnt_d)
    )
    print("verification accuracy: "
          "{:.4f}".format(float(num_right_same + num_right_diff) / (cnt_s + cnt_d)))


if __name__ == "__main__":
    main()

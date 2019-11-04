import sys
import os
import lmdb

from tools.test_ijbc3 import *
ds = DatasetIJBC2(flip=False)
print(len(ds), )

max_map_size = int(len(ds) * 112 ** 2 * 2 * 3 * 16)  # be careful with this
max_map_size = min(max_map_size, int(1e12))
env = lmdb.open('/data/share/IJB_release/IJBB/imgs_lmdb', map_size=max_map_size)
for item in range(len(ds)):
    if item % 999 == 0:
        print(item)
    with env.begin(write=True) as txn:
        imgb = ds.get_raw(item)
        txn.put(str(item).encode(), imgb)
# data_size = 1862120
# inpp = '/data/share/iccv19.lwface/iccv19-challenge-data/'
# i = 0
# max_map_size = int(data_size * 112 ** 2 * 2 * 3 * 16)  # be careful with this
# max_map_size = min(max_map_size, int(1e12))
# env = lmdb.open(inpp+'imgs_lmdb', map_size=max_map_size)
# for line in open(os.path.join(inpp, 'filelist.txt'), 'r'):
#     if i % 1000 == 0:
#         print("processing ", i, data_size, 1. * i / data_size)
#     i += 1
#     line = line.strip()
#     image_path = os.path.join(inpp, line)
#     with env.begin(write=True) as txn:
#         with open(image_path, 'rb') as f:
#             imgb = f.read()
#         txn.put(str(line).encode(), imgb)

# inpp = '/data/share/iccv19.lwface/iQIYI-VID-FACE/'
# i = 0
# filelist = os.path.join(inpp, 'filelist.txt')
# lines = open(filelist, 'r').readlines()
# al_imgs = []
# import glob
#
# for line in lines:
#     if i % 1000 == 0:
#         print("processing ", i)
#     i += 1
#     videoname = line.strip().split()[0]
#     images = glob.glob("%s/%s/*.jpg" % (inpp, videoname))
#     al_imgs.extend(images)
# data_size = len(al_imgs)
# print(len(lines), data_size)
# i = 0
# max_map_size = int(data_size * 112 ** 2 * 2 * 3 * 16)  # be careful with this
# max_map_size = min(max_map_size, int(1e12))
# env = lmdb.open(inpp + 'imgs_lmdb', map_size=max_map_size)
# for line in al_imgs:
#     if i % 1000 == 0:
#         print("processing ", i, data_size, 1. * i / data_size)
#     i += 1
#     item = line.replace(inpp, '')
#     image_path = line
#     assert os.path.exists(line)
#     with env.begin(write=True) as txn:
#         with open(image_path, 'rb') as f:
#             imgb = f.read()
#         txn.put(str(item).encode(), imgb)

env.close()

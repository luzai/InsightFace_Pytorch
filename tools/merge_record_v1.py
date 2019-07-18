# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

#curr_path = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback
#from builtins import range
from easydict import EasyDict as edict
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_preprocess
import face_image

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


# from lst_file return img_info(num_img)+range_head(1)+range_info(num_id)
def read_list(path_in):
    with open(path_in) as fin:
        identities = []
        last = [-1, -1]
        _idx = 1
        while True:
        # read lst line
            line = fin.readline()
            if not line:
                break
            item = edict()
            item.flag = 0
            # from line get aligned,image_path,label(bbox,landmark)
            item.image_path, label, item.bbox, item.landmark, item.aligned = face_preprocess.parse_lst_line(line)
            # cannot process this image, discard
            if not item.aligned and item.landmark is None:
              #print('ignore line', line)
              continue
            item.id = _idx
            item.label = [label, item.aligned]
            yield item        
            if label!=last[0]:
              if last[1]>=0:
                # add idx range
                identities.append( (last[1], _idx) )
              # record last label & idx
              last[0] = label
              last[1] = _idx
            _idx+=1
        identities.append( (last[1], _idx) )
        
        # item = edict()
        # item.flag = 2
        # item.id = 0
        # item.label = [float(_idx), float(_idx+len(identities))]
        # yield item
        
        _idx = 8652799
        for identity in identities:
          item = edict()
          item.flag = 2
          item.id = _idx
          _idx+=1
          item.label = [float(identity[0]), float(identity[1])]
          yield item



def image_encode(args, i, item, q_out):
    # idx
    oitem = [item.id]
    
    if item.flag==0:
      fullpath = item.image_path
      # create header
      header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
      
      if item.aligned:
        with open(fullpath, 'rb') as fin:
            img = fin.read()
        # pack
        s = mx.recordio.pack(header, img)
        q_out.put((i, s, oitem))
      else:
        img = cv2.imread(fullpath, args.color)
        assert item.landmark is not None
        img = face_preprocess.preprocess(img, bbox = item.bbox, landmark=item.landmark, image_size='%d,%d'%(args.image_h, args.image_w))
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, oitem))
    else:
      # info item.flag=2 item.label=range item.id=0/num_img+1~num_img+1+num_id
      header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
      #print('write', item.flag, item.id, item.label)
      s = mx.recordio.pack(header, b'')
      q_out.put((i, s, oitem))

# get data from q_in
def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)

def write_worker(q_out, working_dir):
    pre_time = time.time()
    count = 0
    # basename->get file name only, w/o dir
    fname_rec = 'asia_emore.rec'
    fname_idx = fname_rec[0:-4] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        # write in order
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                # write to rcd: item[0]=idx s=pack
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print(f'time: {cur_time - pre_time},  count: {100.0*count/8832519.0:.2f}%')
                pre_time = cur_time
            count += 1
    record.close()

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output lst and rec files.')
    #parser.add_argument('root', help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', type=bool, default=False,
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', type=bool, default=False,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    cgroup.add_argument('--shuffle', type=bool, default=True, help='If this is set as True, \
        im2rec will randomize the image order in <prefix>.lst')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=12,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--pack-label', type=bool, default=False,
        help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    #args.root = os.path.abspath(args.root)
    return args

if __name__ == '__main__':
    args = parse_args()
    args.image_h = 112
    args.image_w = 112
    
    # ------------------write asia data ---------------------------
    '''
    working_dir = '/data/zhangdy/FR/lst_rec/'
    fname = '/data/zhangdy/FR/lst_rec/glint_cn.lst'
    image_list = read_list(fname)
    
    # write_record
    # if args.num_thread > 1 and multiprocessing is not None:
    # queue.get() will block by default
    q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
    q_out = multiprocessing.Queue(1024)
    # 
    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                    for i in range(args.num_thread)]
    for p in read_process:
        p.start()
    write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
    write_process.start()

    write_cnt = 0
    for i, item in enumerate(image_list):
        # i:idx (from 0 on) item (info, i.e. path, label, etc)
        q_in[i % len(q_in)].put((i, item))
        write_cnt += 1

    # stop signal
    for q in q_in:
        q.put(None)
    for p in read_process:
        p.join()
    '''
    working_dir = '/data/zhangdy/FR/lst_rec/'
    q_out = multiprocessing.Queue(1024)
    write_process = multiprocessing.Process(target=write_worker, args=(q_out, working_dir))
    write_process.start()

    write_cnt = 0

    path_rec1 = '/data/share/asia/glint_cn.rec'
    path_idx1 = path_rec1[0:-4] + '.idx'
    imgrec1 = mx.recordio.MXIndexedRecordIO(path_idx1, path_rec1, 'r')
    s01 = imgrec1.read_idx(0)
    header01, _ = mx.recordio.unpack(s01)
    start1, end1 = header01.label
    num_img_a = start1-1
    num_id_a = end1-start1

    path_rec2 = '/data/share/faces_emore/train.rec'
    path_idx2 = path_rec2[0:-4] + '.idx'
    imgrec2 = mx.recordio.MXIndexedRecordIO(path_idx2, path_rec2, 'r')
    s02 = imgrec2.read_idx(0)
    header02, _ = mx.recordio.unpack(s02)
    start2, end2 = header02.label
    num_img_e = start2-1
    num_id_e = end2-start2
    

    for i in range(1, int(start1)):
        s = imgrec1.read_idx(i)
        header, img = mx.recordio.unpack(s)
        
        # header.label[0] += 93979.0
        # header.id = header.id-1+2830146
        header_new = mx.recordio.IRHeader(header.flag, [float(header.label[0]), 0.0], int(header.id), 0)

        s_new = mx.recordio.pack(header_new, img)
        oitem = [header_new.id]
        q_out.put((write_cnt, s_new, oitem))
        write_cnt += 1

    ran_new_start_a = num_img_a+num_img_e+1
    for i in range(int(start1), int(end1)):
        s = imgrec1.read_idx(i)
        header, _ = mx.recordio.unpack(s)
        
        # header.label[0] = header.label[0]+2830145.0
        # header.label[1] = header.label[1]+2830145.0
        # header.id = header.id-start+8746778
        
        header_new = mx.recordio.IRHeader(header.flag, [float(header.label[0]), float(header.label[1])], int(header.id-start1+ran_new_start_a), 0)

        s_new = mx.recordio.pack(header_new, b'')
        oitem = [header_new.id]
        q_out.put((write_cnt, s_new, oitem))
        write_cnt += 1


    # ------------------write emore data ---------------------------
    
    for i in range(1, int(start2)):
        s = imgrec2.read_idx(i)
        header, img = mx.recordio.unpack(s)
        
        # header.label[0] += 93979.0
        # header.id = header.id-1+2830146
        header_new = mx.recordio.IRHeader(header.flag, [float(header.label+num_id_a), 0.0], int(header.id+num_img_a), 0)

        s_new = mx.recordio.pack(header_new, img)
        oitem = [header_new.id]
        q_out.put((write_cnt, s_new, oitem))
        write_cnt += 1

    ran_new_start_e = ran_new_start_a+num_id_a
    for i in range(int(start2), int(end2)):
        s = imgrec2.read_idx(i)
        header, _ = mx.recordio.unpack(s)
        
        # header.label[0] = header.label[0]+2830145.0
        # header.label[1] = header.label[1]+2830145.0
        # header.id = header.id-start+8746778
        
        header_new = mx.recordio.IRHeader(header.flag, [float(header.label[0]+num_img_a), float(header.label[1]+num_img_a)], int(header.id-start2+ran_new_start_e), 0)

        s_new = mx.recordio.pack(header_new, b'')
        oitem = [header_new.id]
        q_out.put((write_cnt, s_new, oitem))
        write_cnt += 1

    # header1 = header0
    # header1.label[0] = 8652799.0
    # header1.label[1] = 8832520.0
    # header1.id = 0
    # ------------------write header0 ---------------------------
    header_new = mx.recordio.IRHeader(2, [ran_new_start_a, ran_new_start_e+num_id_e], 0, 0)

    s_new = mx.recordio.pack(header_new, b'')
    oitem = [header_new.id]
    q_out.put((write_cnt, s_new, oitem))

    q_out.put(None)
    write_process.join()

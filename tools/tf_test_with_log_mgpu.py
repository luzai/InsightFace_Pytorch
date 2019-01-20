import os, sys
from ctypes import *
import numpy as np
import re
import functools
from scipy import spatial
import re
import multiprocessing as mp
import sklearn.preprocessing
import cPickle as pickle
import argparse
import tensorflow as tf
import mxnet as mx
import pdb

def generate_test_pair(jk_list, zj_list):
    file_paths = [jk_list, zj_list]
    jk_dict = {}
    zj_dict = {}
    jk_zj_dict_list = [jk_dict, zj_dict]
    for path, x_dict in zip(file_paths, jk_zj_dict_list):
        with open(path,'r') as fr:
            for line in fr:
                label = line.strip().split(' ')[1]
                tmp = x_dict.get(label,[])
                tmp.append(line.strip())
                x_dict[label] = tmp
    jk2zj_pairs = []
    zj2jk_pairs = []
    for key in jk_dict:
        jk_file_list = jk_dict[key]
        zj_file_list = zj_dict[key]
        # for jk_file in jk_file_list:
        #     for zj_file in zj_file_list:
        #         jk2zj_pairs.append([jk_file, zj_file])
        #         zj2jk_pairs.append([zj_file, jk_file])
        for zj_file in zj_file_list:
            jk_list_tmp = []
            for jk_file in jk_file_list:
                jk_list_tmp.append(jk_file)
                jk2zj_pairs.append([jk_file, [zj_file]])
            zj2jk_pairs.append([zj_file, jk_list_tmp])
    return jk2zj_pairs, zj2jk_pairs

def get_test_set_dict(test_cache_root, jk_list, zj_list):
    ret_dict = {}
    with open(zj_list) as label_fr:
        with open(os.path.join(test_cache_root,'zj_list.bin'),'rb') as feature_fr:
            for line in label_fr:
                key = line.strip()
                feature = np.fromstring(feature_fr.read(feature_len*4), dtype=np.float32)
                ret_dict[key] = feature
    with open(jk_list) as label_fr:
        with open(os.path.join(test_cache_root,'jk_all_list.bin'),'rb') as feature_fr:
            for line in label_fr:
                key = line.strip()
                feature = np.fromstring(feature_fr.read(feature_len*4), dtype=np.float32)
                ret_dict[key] = feature
    return ret_dict

def tf_build_graph(sess):
    ret_list = []
    dis_features_list = np.array_split(dis_features, gpu_used)
    idx_start = [0]+[f.shape[0] for f in dis_features_list]
    idx_start = [sum(idx_start[:i]) for i in range(1, len(idx_start))]
    feed_dict = {}
    for device_id in range(gpu_used):
        with tf.device('/gpu:%s' % device_id):
            dis_feature = tf.placeholder(tf.float32, shape=dis_features_list[device_id].shape)
            disv_feature = tf.Variable(dis_feature)
            feed_dict[dis_feature] = dis_features_list[device_id]
            query_feature = tf.placeholder(tf.float32, shape=(None, feature_len))
            similarity = tf.matmul(query_feature, tf.transpose(disv_feature))
            similarity = tf.squeeze(similarity)
            print similarity.get_shape()
            query_results = tf.nn.top_k(similarity, k=100)
            ret_list.append((query_results, query_feature))
    sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
    return ret_list, idx_start

def tf_build_graph_merge_topk(gpu_nums):
    # return topk, ids of the merge result
    sim_feed_keys, ids_feed_keys = [],[]
    sim_tensor_list, id_tensor_list = [], []
    with tf.device('/cpu'):
        for i in range(gpu_nums):
            v = tf.placeholder(tf.float32)
            sim_feed_keys.append(v)
            sim_tensor_list.append(v)
            
            v = tf.placeholder(tf.int32)
            ids_feed_keys.append(v)
            v = v+idx_start[i]
            id_tensor_list.append(v)
        axis = -1
        total = tf.concat(sim_tensor_list, axis=axis)
        total_ids = tf.concat(id_tensor_list, axis=axis)
        topk = tf.nn.top_k(total, k=100)
        return topk, total_ids, sim_feed_keys, ids_feed_keys
    
def tf_query(sess, search_feature, ret_pairs):
    '''
    :param sess:
    :param search_feature:
    :param ret_pairs:
    :param gpu_used:
    :param merge_topk_items: tuple (topk result, total_ids result, sim_feed_keys, id_feed_keys)
    :return:
    '''
    ids_list, similarities_list = [], []

    for device_id in range(gpu_used):
        with tf.device('/gpu:%s' % device_id):
            query_results, query_feature = ret_pairs[device_id]

            if len(search_feature.shape) < 2:
                search_feature = search_feature[np.newaxis, :]

            similarities, ids = sess.run(query_results, feed_dict={query_feature: search_feature})
            ids_list.append(ids)
            similarities_list.append(similarities)
   
    with tf.device('/gpu:0'):
        topk, total_ids, sim_feed_keys, id_feed_keys = topk_items
        assert len(ids_list) == len(id_feed_keys)
        assert len(similarities_list) == len(id_feed_keys)
        feed_dict = {}
        for ids, sims, id_feed_key, sim_feed_key in zip(ids_list, similarities_list, id_feed_keys, sim_feed_keys):
            feed_dict[id_feed_key] = ids
            feed_dict[sim_feed_key] = sims

        topk_result, total_ids_result = sess.run([topk, total_ids], feed_dict=feed_dict)
        similarities, ids = topk_result
        # pdb.set_trace()
        # TODO(HZF) ugly codes
        if len(total_ids_result.shape) > 1:
            ids_ret = []
            for t_ids, id_idx in zip(total_ids_result, ids):
                ids_ret.append(t_ids[id_idx])
            ids = np.array(ids_ret, dtype=np.int32)
        else:
            ids = total_ids_result[ids]
    return ids, similarities
    

def result_generator(pairs, test_set_dict, sess, query_results, label_list, q):

    def write_result(ids, similarities, pair):
        
        ret_str = ''
        ret_str += 'GroupId=%s;'%pair[0].split(' ')[1]
        ret_str += 'eType=%d;Src=%s,%s'%(idx, pair[0].split(' ')[0], pair[1].split(' ')[0])
        
        if pair[0] not in test_set_dict or pair[1] not in test_set_dict:
            ret_str+=',0;'
            for i in [1,5,10,50,100]:
                ret_str += 'top%d=0,'%i
            q.append(ret_str)
            return
        search_feature = test_set_dict[pair[0]]
        # print search_feature.shape
        searched_feature = test_set_dict[pair[1]]
        # print searched_feature.shape
        
        similarity = np.dot(search_feature, searched_feature)
        ret_str += ',%.3g;'%(similarity)
        flag = False
        for i in [1,5,10,50,100]:
            tk = 0
            if flag or similarity > similarities[i-1]:
                flag = True
                tk = 1
            ret_str += 'top%d=%d,'%(i, tk)
        if label_list is not None:
            ret_str = ret_str[:-1]+';'+'dis_top='
            for i in [1,2,3]:
                ret_str += '%.3g,%s,'%(similarities[i-1], label_list[ids[i-1]].split(' ')[0])
        q.append(ret_str)
            
    idx = 0
    count = 0
    #zj2jk
    for opair in pairs:
        search_feature = test_set_dict[opair[0]]
        ids, similarities = tf_query(sess, search_feature, query_results)
        
        for item in opair[1]:
            pair = [opair[0], item]
            write_result(ids, similarities, pair)
            count += 1
            if count % 1000 == 0:
                print 'process: %d'%count
    idx = 2
    count = 0
    #jk2zj
    for opair in pairs:
        search_features = []
        for item in opair[1]:
            search_features.append(test_set_dict[item])
        search_features = np.array(search_features)
        # print search_features.shape
        # pdb.set_trace()
        idss, similaritiess = tf_query(sess, search_features, query_results)
        # print idss.shape, similaritiess.shape
        assert len(idss) == len(opair[1]) and len(idss)==len(similaritiess)
        for ids, similarities, item in zip(idss, similaritiess, opair[1]):
            pair = [item, opair[0]]
            write_result(ids, similarities, pair)
            count += 1
            if count % 1000 == 0:
                print 'process: %d'%count
    
    
def comsumer(q, fw):
    for result in q:
        fw.write('%s\n'%result[:-1])
            
def get_final_result(test_cache_root, jk_list, zj_list, dist_list_path, result_file_path):
    jk2zj_pairs, zj2jk_pairs = generate_test_pair(jk_list, zj_list)
    dist_list = []
    with open(dist_list_path,'r') as fr:
        for line in fr:
            dist_list.append(line.strip().split(' ')[0])
    test_set_dict = get_test_set_dict(test_cache_root, jk_list, zj_list)
    result_pattern = 'GroupId=%s;eType=%d;Src=%s,%s;top%d=%d,top%d=%d,top%d=%d,top%d=%d,top%d=%d'
    fw = open(result_file_path,'w')
    q = []
    result_generator(zj2jk_pairs, test_set_dict, sess, query_results_pairs, dist_list, q)
    print len(q)
    comsumer(q,fw)
    fw.close()

            
feature_len = 512
query_process_num = 1
gpu_used = 2
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='model names')
    return parser.parse_args()

args = parse_arguments()
# This names of the files and tests.
media = 'MediaCapturedImagesStd_02_en_sim'
hy = 'hy'
facereg = 'facereg1N_Select30_sim'

total_test = 'testsets_mtcnn_align112x112'

dist_list = 'dis_list.txt' 

small_pix = '0509_Small_Pix_Select_clear'
big_angle = 'Big_Angle_all'
media_shade = 'MediaCapturedImagesStd_02_shade'

zj = '/zj_list.txt'
jk = '/jk_all_list.txt'


model_name = args.model_name
if 'pca' in model_name or '256' in model_name or 'beta' in model_name:
    feature_len=256
# Paths to the images lists, distractors dumps, images from lists dumps.
lists = '../data/lists/'
test_lists = '/mnt/109_ssd/ssd1/hzf_data/testsets_mtcnn_align112x112/'
dist = '../output/'+model_name+'/dis/'
zjjk = '../output/'+model_name+'/'
spzjjk = '../sp_output/'+model_name+'/'

dis_feature_len = 0
for im_name in os.listdir(dist):
    if 'bin' in im_name:
        st = os.stat(os.path.join(dist, im_name))
        dis_feature_len += st.st_size/feature_len/4
dis_features = np.zeros((dis_feature_len, feature_len))
idx = 0
print 'loading dis set: %d'%dis_feature_len
for im_name in os.listdir(dist):
    if 'bin' in im_name:
        with open(os.path.join(dist, im_name),'rb') as fr:
            while True:
                feature = fr.read(feature_len*4)
                if len(feature) == 0:
                    break
                feature = np.fromstring(feature, dtype=np.float32)
                dis_features[idx,:] = feature
                idx+=1
assert idx == dis_feature_len
print 'total dis feature length:%d'%idx

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
query_results_pairs, idx_start = tf_build_graph(sess)
topk_items = tf_build_graph_merge_topk(gpu_used)
#get_final_result(distractor_index, zjjk+total_test, test_lists+jk, test_lists+zj, dist+dist_list, zjjk+total_test+'.log')
get_final_result(zjjk+facereg, lists+facereg+jk, lists+facereg+zj, dist+dist_list, zjjk+facereg+'.log')
get_final_result(zjjk+media, lists+media+jk, lists+media+zj, dist+dist_list, zjjk+media+'.log')

#split test @deprecated
#get_final_result(distractor_index, zjjk+hy, lists+hy+jk, lists+hy+zj, lists+dist_list, zjjk+hy+'.log')

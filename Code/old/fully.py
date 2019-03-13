# -*- coding:utf-8 -*-
'''
maligancy

'''
import tensorflow as tf
from dataprepare import get_train_and_test_filenames, get_batch_withlabels, K_Cross_Split
import random
import time
import datetime
import os
import tqdm
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)


class model(object):

    def __init__(self,learning_rate,keep_prob,batch_size,epoch):
        print(" network begin...")
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.epoch = epoch

        self.cubic_shape = [[10, 20, 20], [6, 20, 20], [26, 40, 40]]
    
    def archi_1(self, input,keep_prob):
        # return out_fc2
        with tf.name_scope("Body"):
            # inputsize = 42, 3 channles
            # conv1 42 x 42 -> 38 x 38
            w_conv1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev = 0.01), dtype = tf.float32, name = 'w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape = [32]), dtype = tf.float32, name = 'b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(input, w_conv1, strides = [1,1,1,1], padding='VALID'), b_conv1))

            # conv2 38 x 38 -> 34 x 34
            w_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev = 0.01), dtype = tf.float32, name = 'w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape = [64]), dtype = tf.float32, name = 'b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv1, w_conv2, strides = [1,1,1,1], padding = 'VALID'), b_conv2))

            # conv3 34 x 34 -> 30 x 30
            w_conv3 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev = 0.01), dtype = tf.float32, name = 'w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape = [128]), dtype = tf.float32, name = 'b_conv3')
            out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv2, w_conv3, strides = [1,1,1,1], padding = 'VALID'), b_conv3))

            # conv4 30 x 30 -> 28 x 28
            w_conv4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev = 0.01), dtype = tf.float32, name = 'w_conv4')
            b_conv4 = tf.Variable(tf.constant(0.01, shape = [256]), dtype = tf.float32, name = 'b_conv4')
            out_conv4 = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv3, w_conv4, strides = [1,1,1,1], padding = 'VALID'), b_conv4))

            # conv5 28 x 28 -> 26 x 26
            w_conv5 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev = 0.01), dtype = tf.float32, name = 'w_conv5')
            b_conv5 = tf.Variable(tf.constant(0.01, shape = [256]), dtype = tf.float32, name = 'b_conv5')
            out_conv5 = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv4, w_conv5, strides = [1,1,1,1], padding = 'VALID'), b_conv5))

            '''
            code above is for the para sharing
            code after is for different attributes
            '''

            attributes_res = []

            # subtlety = [] 
            # 26 26 - 24 24
            sub_w_conv = tf.Variable(tf.random_normal([3, 3, 256, 128], stddev = 0.01), dtype = tf.float32, name = 'sub_w_conv')
            sub_b_conv = tf.Variable(tf.constant(0.01, shape = [128], dtype = tf.float32, name = 'sub_b_conv'))
            sub_out_conv = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv5, sub_w_conv, strides = [1, 1, 1, 1], padding = 'VALID'), sub_b_conv))

            # conv1 24 x 24 -> 22 x 22
            sub_w_conv1 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev = 0.01), dtype = tf.float32, name = 'sub_w_conv1')
            sub_b_conv1 = tf.Variable(tf.constant(0.01, shape = [64], dtype = tf.float32, name = 'sub_b_conv1'))
            sub_out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(sub_out_conv, sub_w_conv1, strides = [1, 1, 1, 1], padding = 'VALID'), sub_b_conv1))

            # pooling1 22 x 22 -> 11 x 11
            sub_pooling1 = tf.nn.max_pool(sub_out_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

            # conv2 11 x 11 -> 9 x 9
            sub_w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev = 0.01), dtype = tf.float32, name = 'sub_w_conv2')
            sub_b_conv2 = tf.Variable(tf.constant(0.01, shape = [32], dtype = tf.float32, name = 'sub_b_conv2'))
            sub_out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(sub_pooling1, sub_w_conv2, strides = [1, 1, 1, 1], padding = 'VALID'), sub_b_conv2))

            # conv3 9 x 9-> 7 x 7
            # sub_w_conv3 = tf.Variable(tf.random_normal([3, 3, 32, 3], stddev = 0.01), dtype = tf.float32, name = 'sub_w_conv3')
            # sub_b_conv3 = tf.Variable(tf.constant(0.01, shape = [3], dtype = tf.float32, name = 'sub_b_conv3'))
            # sub_out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(sub_out_conv2, sub_w_conv3, strides = [1, 1, 1, 1], padding = 'VALID'), sub_b_conv3))

            # sub_pooling2 = tf.nn.avg_pool(sub_out_conv3, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID')

            # sub_netout = tf.reduce_mean(sub_pooling2, axis=2, keep_dims=False)
            # sub_netout = tf.reduce_mean(sub_netout, axis=1, keep_dims=False)

            # attributes_res.append(sub_netout)

            sub_out_conv2 = tf.reshape(sub_out_conv2,[-1, 9 * 9 * 32 ])
            sub_w_fc1 = tf.Variable(tf.random_normal([9 * 9 * 32, 3],stddev=0.01),name='sub_w_fc1')
            sub_out_fc1 = tf.nn.relu(tf.add(tf.matmul(sub_out_conv2, sub_w_fc1), tf.constant(0.01,shape=[3])))
            sub_out_fc1 = tf.nn.dropout(sub_out_fc1,keep_prob)

            attributes_res.append(sub_out_fc1)

            # sphertlety = [] 
            
            spher_w_conv = tf.Variable(tf.random_normal([3, 3, 256, 128], stddev = 0.01), dtype = tf.float32, name = 'spher_w_conv')
            spher_b_conv = tf.Variable(tf.constant(0.01, shape = [128], dtype = tf.float32, name = 'spher_b_conv'))
            spher_out_conv = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv5, spher_w_conv, strides = [1, 1, 1, 1], padding = 'VALID'), spher_b_conv))

            # conv1 24 x 24 -> 22 x 22
            spher_w_conv1 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev = 0.01), dtype = tf.float32, name = 'spher_w_conv1')
            spher_b_conv1 = tf.Variable(tf.constant(0.01, shape = [64], dtype = tf.float32, name = 'spher_b_conv1'))
            spher_out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(spher_out_conv, spher_w_conv1, strides = [1, 1, 1, 1], padding = 'VALID'), spher_b_conv1))

            # pooling1 22 x 22 -> 11 x 11
            spher_pooling1 = tf.nn.max_pool(spher_out_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

            # conv2 11 x 11 -> 9 x 9
            spher_w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev = 0.01), dtype = tf.float32, name = 'spher_w_conv2')
            spher_b_conv2 = tf.Variable(tf.constant(0.01, shape = [32], dtype = tf.float32, name = 'spher_b_conv2'))
            spher_out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(spher_pooling1, spher_w_conv2, strides = [1, 1, 1, 1], padding = 'VALID'), spher_b_conv2))

            # conv3 9 x 9-> 7 x 7
            # spher_w_conv3 = tf.Variable(tf.random_normal([3, 3, 32, 3], stddev = 0.01), dtype = tf.float32, name = 'spher_w_conv3')
            # spher_b_conv3 = tf.Variable(tf.constant(0.01, shape = [3], dtype = tf.float32, name = 'spher_b_conv3'))
            # spher_out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(spher_out_conv2, spher_w_conv3, strides = [1, 1, 1, 1], padding = 'VALID'), spher_b_conv3))

            # spher_pooling2 = tf.nn.avg_pool(spher_out_conv3, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID')

            # spher_netout = tf.reduce_mean(spher_pooling2, axis=2, keep_dims=False)
            # spher_netout = tf.reduce_mean(spher_netout, axis=1, keep_dims=False)
            # attributes_res.append(spher_netout)

            spher_out_conv2 = tf.reshape(spher_out_conv2,[-1, 9 * 9 * 32 ])
            spher_w_fc1 = tf.Variable(tf.random_normal([9 * 9 * 32, 3],stddev=0.01),name='spher_w_fc1')
            spher_out_fc1 = tf.nn.relu(tf.add(tf.matmul(spher_out_conv2, spher_w_fc1), tf.constant(0.01,shape=[3])))
            spher_out_fc1 = tf.nn.dropout(spher_out_fc1,keep_prob)

            attributes_res.append(spher_out_fc1)

            # margin = [] 

            marg_w_conv = tf.Variable(tf.random_normal([3, 3, 256, 128], stddev = 0.01), dtype = tf.float32, name = 'marg_w_conv')
            marg_b_conv = tf.Variable(tf.constant(0.01, shape = [128], dtype = tf.float32, name = 'marg_b_conv'))
            marg_out_conv = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv5, marg_w_conv, strides = [1, 1, 1, 1], padding = 'VALID'), marg_b_conv))

            # conv1 24 x 24 -> 22 x 22
            marg_w_conv1 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev = 0.01), dtype = tf.float32, name = 'marg_w_conv1')
            marg_b_conv1 = tf.Variable(tf.constant(0.01, shape = [64], dtype = tf.float32, name = 'marg_b_conv1'))
            marg_out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(marg_out_conv, marg_w_conv1, strides = [1, 1, 1, 1], padding = 'VALID'), marg_b_conv1))

            # pooling1 22 x 22 -> 11 x 11
            marg_pooling1 = tf.nn.max_pool(marg_out_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

            # conv2 11 x 11 -> 9 x 9
            marg_w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev = 0.01), dtype = tf.float32, name = 'marg_w_conv2')
            marg_b_conv2 = tf.Variable(tf.constant(0.01, shape = [32], dtype = tf.float32, name = 'marg_b_conv2'))
            marg_out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(marg_pooling1, marg_w_conv2, strides = [1, 1, 1, 1], padding = 'VALID'), marg_b_conv2))

            # conv3 9 x 9-> 7 x 7
            # marg_w_conv3 = tf.Variable(tf.random_normal([3, 3, 32, 3], stddev = 0.01), dtype = tf.float32, name = 'marg_w_conv3')
            # marg_b_conv3 = tf.Variable(tf.constant(0.01, shape = [3], dtype = tf.float32, name = 'marg_b_conv3'))
            # marg_out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(marg_out_conv2, marg_w_conv3, strides = [1, 1, 1, 1], padding = 'VALID'), marg_b_conv3))

            # marg_pooling2 = tf.nn.avg_pool(marg_out_conv3, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID')

            # marg_netout = tf.reduce_mean(marg_pooling2, axis=2, keep_dims=False)
            # marg_netout = tf.reduce_mean(marg_netout, axis=1, keep_dims=False)

            # attributes_res.append(marg_netout)

            marg_out_conv2 = tf.reshape(marg_out_conv2,[-1, 9 * 9 * 32 ])
            marg_w_fc1 = tf.Variable(tf.random_normal([9 * 9 * 32, 3],stddev=0.01),name='marg_w_fc1')
            marg_out_fc1 = tf.nn.relu(tf.add(tf.matmul(marg_out_conv2, marg_w_fc1), tf.constant(0.01,shape=[3])))
            marg_out_fc1 = tf.nn.dropout(marg_out_fc1,keep_prob)

            attributes_res.append(marg_out_fc1)

            # lobulation = [] 
            # conv1 30 x 30 -> 28 x 28
            lobu_w_conv = tf.Variable(tf.random_normal([3, 3, 256, 128], stddev = 0.01), dtype = tf.float32, name = 'lobu_w_conv')
            lobu_b_conv = tf.Variable(tf.constant(0.01, shape = [128], dtype = tf.float32, name = 'lobu_b_conv'))
            lobu_out_conv = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv5, lobu_w_conv, strides = [1, 1, 1, 1], padding = 'VALID'), lobu_b_conv))

            # conv1 24 x 24 -> 22 x 22
            lobu_w_conv1 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev = 0.01), dtype = tf.float32, name = 'lobu_w_conv1')
            lobu_b_conv1 = tf.Variable(tf.constant(0.01, shape = [64], dtype = tf.float32, name = 'lobu_b_conv1'))
            lobu_out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(lobu_out_conv, lobu_w_conv1, strides = [1, 1, 1, 1], padding = 'VALID'), lobu_b_conv1))

            # pooling1 22 x 22 -> 11 x 11
            lobu_pooling1 = tf.nn.max_pool(lobu_out_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

            # conv2 11 x 11 -> 9 x 9
            lobu_w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev = 0.01), dtype = tf.float32, name = 'lobu_w_conv2')
            lobu_b_conv2 = tf.Variable(tf.constant(0.01, shape = [32], dtype = tf.float32, name = 'lobu_b_conv2'))
            lobu_out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(lobu_pooling1, lobu_w_conv2, strides = [1, 1, 1, 1], padding = 'VALID'), lobu_b_conv2))

            # conv3 9 x 9-> 7 x 7
            # lobu_w_conv3 = tf.Variable(tf.random_normal([3, 3, 32, 3], stddev = 0.01), dtype = tf.float32, name = 'lobu_w_conv3')
            # lobu_b_conv3 = tf.Variable(tf.constant(0.01, shape = [3], dtype = tf.float32, name = 'lobu_b_conv3'))
            # lobu_out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(lobu_out_conv2, lobu_w_conv3, strides = [1, 1, 1, 1], padding = 'VALID'), lobu_b_conv3))

            # lobu_pooling2 = tf.nn.avg_pool(lobu_out_conv3, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID')

            # lobu_netout = tf.reduce_mean(lobu_pooling2, axis=2, keep_dims=False)
            # lobu_netout = tf.reduce_mean(lobu_netout, axis=1, keep_dims=False)           

            # attributes_res.append(lobu_netout)
            lobu_out_conv2 = tf.reshape(lobu_out_conv2,[-1, 9 * 9 * 32 ])
            lobu_w_fc1 = tf.Variable(tf.random_normal([9 * 9 * 32, 3],stddev=0.01),name='lobu_w_fc1')
            lobu_out_fc1 = tf.nn.relu(tf.add(tf.matmul(lobu_out_conv2, lobu_w_fc1), tf.constant(0.01,shape=[3])))
            lobu_out_fc1 = tf.nn.dropout(lobu_out_fc1,keep_prob)

            attributes_res.append(lobu_out_fc1)

            # spiculation = [] 
            spicu_w_conv = tf.Variable(tf.random_normal([3, 3, 256, 128], stddev = 0.01), dtype = tf.float32, name = 'spicu_w_conv')
            spicu_b_conv = tf.Variable(tf.constant(0.01, shape = [128], dtype = tf.float32, name = 'spicu_b_conv'))
            spicu_out_conv = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv5, spicu_w_conv, strides = [1, 1, 1, 1], padding = 'VALID'), spicu_b_conv))

            # conv1 24 x 24 -> 22 x 22
            spicu_w_conv1 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev = 0.01), dtype = tf.float32, name = 'spicu_w_conv1')
            spicu_b_conv1 = tf.Variable(tf.constant(0.01, shape = [64], dtype = tf.float32, name = 'spicu_b_conv1'))
            spicu_out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(spicu_out_conv, spicu_w_conv1, strides = [1, 1, 1, 1], padding = 'VALID'), spicu_b_conv1))

            # pooling1 22 x 22 -> 11 x 11
            spicu_pooling1 = tf.nn.max_pool(spicu_out_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

            # conv2 11 x 11 -> 9 x 9
            spicu_w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev = 0.01), dtype = tf.float32, name = 'spicu_w_conv2')
            spicu_b_conv2 = tf.Variable(tf.constant(0.01, shape = [32], dtype = tf.float32, name = 'spicu_b_conv2'))
            spicu_out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(spicu_pooling1, spicu_w_conv2, strides = [1, 1, 1, 1], padding = 'VALID'), spicu_b_conv2))

            # conv3 9 x 9-> 7 x 7
            # spicu_w_conv3 = tf.Variable(tf.random_normal([3, 3, 32, 3], stddev = 0.01), dtype = tf.float32, name = 'spicu_w_conv3')
            # spicu_b_conv3 = tf.Variable(tf.constant(0.01, shape = [3], dtype = tf.float32, name = 'spicu_b_conv3'))
            # spicu_out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(spicu_out_conv2, spicu_w_conv3, strides = [1, 1, 1, 1], padding = 'VALID'), spicu_b_conv3))

            # spicu_pooling2 = tf.nn.avg_pool(spicu_out_conv3, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID')

            # spicu_netout = tf.reduce_mean(spicu_pooling2, axis=2, keep_dims=False)
            # spicu_netout = tf.reduce_mean(spicu_netout, axis=1, keep_dims=False)

            spicu_out_conv2 = tf.reshape(spicu_out_conv2,[-1, 9 * 9 * 32 ])
            spicu_w_fc1 = tf.Variable(tf.random_normal([9 * 9 * 32, 3],stddev=0.01),name='spicu_w_fc1')
            spicu_out_fc1 = tf.nn.relu(tf.add(tf.matmul(spicu_out_conv2, spicu_w_fc1), tf.constant(0.01,shape=[3])))
            spicu_out_fc1 = tf.nn.dropout(spicu_out_fc1,keep_prob)

            attributes_res.append(spicu_out_fc1)

            # texture = [] 
            # conv1 30 x 30 -> 28 x 28
            text_w_conv = tf.Variable(tf.random_normal([3, 3, 256, 128], stddev = 0.01), dtype = tf.float32, name = 'text_w_conv')
            text_b_conv = tf.Variable(tf.constant(0.01, shape = [128], dtype = tf.float32, name = 'text_b_conv'))
            text_out_conv = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv5, text_w_conv, strides = [1, 1, 1, 1], padding = 'VALID'), text_b_conv))

            # conv1 24 x 24 -> 22 x 22
            text_w_conv1 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev = 0.01), dtype = tf.float32, name = 'text_w_conv1')
            text_b_conv1 = tf.Variable(tf.constant(0.01, shape = [64], dtype = tf.float32, name = 'text_b_conv1'))
            text_out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(text_out_conv, text_w_conv1, strides = [1, 1, 1, 1], padding = 'VALID'), text_b_conv1))

            # pooling1 22 x 22 -> 11 x 11
            text_pooling1 = tf.nn.max_pool(text_out_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

            # conv2 11 x 11 -> 9 x 9
            text_w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev = 0.01), dtype = tf.float32, name = 'text_w_conv2')
            text_b_conv2 = tf.Variable(tf.constant(0.01, shape = [32], dtype = tf.float32, name = 'text_b_conv2'))
            text_out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(text_pooling1, text_w_conv2, strides = [1, 1, 1, 1], padding = 'VALID'), text_b_conv2))

            # conv3 9 x 9-> 7 x 7
            # text_w_conv3 = tf.Variable(tf.random_normal([3, 3, 32, 3], stddev = 0.01), dtype = tf.float32, name = 'text_w_conv3')
            # text_b_conv3 = tf.Variable(tf.constant(0.01, shape = [3], dtype = tf.float32, name = 'text_b_conv3'))
            # text_out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(text_out_conv2, text_w_conv3, strides = [1, 1, 1, 1], padding = 'VALID'), text_b_conv3))

            # text_pooling2 = tf.nn.avg_pool(text_out_conv3, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID')

            # text_netout = tf.reduce_mean(text_pooling2, axis=2, keep_dims=False)
            # text_netout = tf.reduce_mean(text_netout, axis=1, keep_dims=False)

            text_out_conv2 = tf.reshape(text_out_conv2,[-1, 9 * 9 * 32 ])
            text_w_fc1 = tf.Variable(tf.random_normal([9 * 9 * 32, 3],stddev=0.01),name='text_w_fc1')
            text_out_fc1 = tf.nn.relu(tf.add(tf.matmul(text_out_conv2, text_w_fc1), tf.constant(0.01,shape=[3])))
            text_out_fc1 = tf.nn.dropout(text_out_fc1,keep_prob)

            attributes_res.append(text_out_fc1)

            # malignancy = [] 
            # conv1 30 x 30 -> 28 x 28
            malig_w_conv = tf.Variable(tf.random_normal([3, 3, 256, 128], stddev = 0.01), dtype = tf.float32, name = 'malig_w_conv')
            malig_b_conv = tf.Variable(tf.constant(0.01, shape = [128], dtype = tf.float32, name = 'malig_b_conv'))
            malig_out_conv = tf.nn.relu(tf.add(tf.nn.conv2d(out_conv5, malig_w_conv, strides = [1, 1, 1, 1], padding = 'VALID'), malig_b_conv))

            # conv1 24 x 24 -> 22 x 22
            malig_w_conv1 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev = 0.01), dtype = tf.float32, name = 'malig_w_conv1')
            malig_b_conv1 = tf.Variable(tf.constant(0.01, shape = [64], dtype = tf.float32, name = 'malig_b_conv1'))
            malig_out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(malig_out_conv, malig_w_conv1, strides = [1, 1, 1, 1], padding = 'VALID'), malig_b_conv1))

            # pooling1 22 x 22 -> 11 x 11
            malig_pooling1 = tf.nn.max_pool(malig_out_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

            # conv2 11 x 11 -> 9 x 9
            malig_w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev = 0.01), dtype = tf.float32, name = 'malig_w_conv2')
            malig_b_conv2 = tf.Variable(tf.constant(0.01, shape = [32], dtype = tf.float32, name = 'malig_b_conv2'))
            malig_out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(malig_pooling1, malig_w_conv2, strides = [1, 1, 1, 1], padding = 'VALID'), malig_b_conv2))

            # conv3 9 x 9-> 7 x 7
            malig_w_conv3 = tf.Variable(tf.random_normal([3, 3, 32, 2], stddev = 0.01), dtype = tf.float32, name = 'malig_w_conv3')
            malig_b_conv3 = tf.Variable(tf.constant(0.01, shape = [2], dtype = tf.float32, name = 'malig_b_conv3'))
            malig_out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(malig_out_conv2, malig_w_conv3, strides = [1, 1, 1, 1], padding = 'VALID'), malig_b_conv3))

            malig_pooling2 = tf.nn.avg_pool(malig_out_conv3, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID')

            malig_netout = tf.reduce_mean(malig_pooling2, axis=2, keep_dims=False)
            malig_netout = tf.reduce_mean(malig_netout, axis=1, keep_dims=False)

            # malig_out_conv2 = tf.reshape(malig_out_conv2,[-1, 9 * 9 * 32 ])
            # malig_w_fc1 = tf.Variable(tf.random_normal([9 * 9 * 32, 3],stddev=0.01),name='malig_w_fc1')
            # malig_out_fc1 = tf.nn.relu(tf.add(tf.matmul(malig_out_conv2, malig_w_fc1), tf.constant(0.01,shape=[3])))
            # malig_out_fc1 = tf.nn.dropout(malig_out_fc1,keep_prob)

            # attributes_res.append(malig_netout)
           
            return attributes_res, malig_netout

    def classifylayer(self, input, inputshape):
        input_X = tf.reshape(input, [-1, inputshape])
        input_X_W = tf.Variable(tf.random_normal([inputshape, 2], stddev = 0.01))
        input_X_B = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32)
        out = tf.add(tf.matmul(input_X, input_X_W), input_X_B)
        return out

    def inference(self,npy_path,model_index,test_size,seed,train_flag=True):
        f = open('2019.txt', 'w')
        # some statistic index
        highest_acc = 0.0
        highest_iterator = 1

        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        # (3, 42, 42)
        # take placeholder as input
        x = tf.placeholder(tf.float32, [None, 42, 42, 3], name = 'x')
        x_image = tf.reshape(x, [-1, 42, 42, 3])
        net_out, malig_netout = self.archi_1(x, keep_prob)

        real_label0 = tf.placeholder(tf.float32, [None, 3], name = 'real_label0')
        real_label1 = tf.placeholder(tf.float32, [None, 3], name = 'real_label1')
        real_label2 = tf.placeholder(tf.float32, [None, 3], name = 'real_label2')
        real_label3 = tf.placeholder(tf.float32, [None, 3], name = 'real_label3')
        real_label4 = tf.placeholder(tf.float32, [None, 3], name = 'real_label4')
        real_label5 = tf.placeholder(tf.float32, [None, 3], name = 'real_label5')

        real_label6 = tf.placeholder(tf.float32, [None, 2], name = 'real_label6')
        # b = tf.constant(value=1,dtype=tf.float32)
        # net_out = tf.multiply(net_out,b,name='net_out') 
        # net_out = tf.convert_to_tensor(net_out, name = 'net_out')
        # 6 x 5
        saver = tf.train.Saver()  # default to save all variable,save mode or restore from path

        if train_flag:
            global_step = tf.Variable(0, name = 'global_step')  
            learning_rate = tf.train.exponential_decay(0.01, global_step, 500, 1, staircase=True, name = 'learning_rate')

            cross_entropy0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.nn.softmax(net_out[0]), labels=real_label0, name = 'cross_entropy0'))
            cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.nn.softmax(net_out[1]), labels=real_label1, name = 'cross_entropy1'))
            cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.nn.softmax(net_out[2]), labels=real_label2, name = 'cross_entropy2'))
            cross_entropy3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.nn.softmax(net_out[3]), labels=real_label3, name = 'cross_entropy3'))
            cross_entropy4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.nn.softmax(net_out[4]), labels=real_label4, name = 'cross_entropy4'))
            cross_entropy5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.nn.softmax(net_out[5]), labels=real_label5, name = 'cross_entropy5'))

            cross_entropy6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.nn.softmax(malig_netout), labels=real_label6, name = 'cross_entropy6'))
            print("cross_entropy0: ",cross_entropy0.get_shape())
        
            w0 = tf.Variable(initial_value=1.0, name = 'w0')
            w1 = tf.Variable(tf.random_normal([1], stddev = 0.01), name = 'w1')
            w2 = tf.Variable(tf.random_normal([1], stddev = 0.01), name = 'w2')
            w3 = tf.Variable(tf.random_normal([1], stddev = 0.01), name = 'w3')
            w4 = tf.Variable(tf.random_normal([1], stddev = 0.01), name = 'w4')
            w5 = tf.Variable(tf.random_normal([1], stddev = 0.01), name = 'w5')
            w6 = tf.Variable(tf.random_normal([1], stddev = 0.01), name = 'w6')

            # sum_of_entropy = (tf.matmul(w0, cross_entropy0) + tf.matmul(w1, cross_entropy1) + tf.matmul(w2, cross_entropy2) + tf.matmul(w3, cross_entropy3) + tf.matmul(w4, cross_entropy4) + tf.matmul(w5, cross_entropy5) + tf.matmul(w6, cross_entropy6))
            loss_of_entropy = cross_entropy0 +  cross_entropy1 + cross_entropy2 + cross_entropy3 + cross_entropy4 + cross_entropy5 + cross_entropy6

            classify0 = self.classifylayer(net_out[0], 3)
            classify1 = self.classifylayer(net_out[1], 3)
            classify2 = self.classifylayer(net_out[2], 3)
            classify3 = self.classifylayer(net_out[3], 3)
            classify4 = self.classifylayer(net_out[4], 3)
            classify5 = self.classifylayer(net_out[5], 3)
            classify6 = self.classifylayer(malig_netout, 2)

            wc0 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'wc0')
            wc1 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'wc1')
            wc2 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'wc2')
            wc3 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'wc3')
            wc4 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'wc4')
            wc5 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'wc5')
            wc6 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'wc6')
            print("test:")
            print("classify1: ",classify1.get_shape())
            print("wc1: ", wc1.get_shape())

            classifyres = (tf.matmul(classify0, wc0) + tf.matmul(classify1, wc1) + tf.matmul(classify2, wc2)+ tf.matmul(classify3, wc3)+ tf.matmul(classify4, wc4)+ tf.matmul(classify5, wc5) + tf.matmul(classify6, wc6))
            #classifyres = tf.matmul(classify6, wc6)
            print('classifyres: ', classifyres.get_shape())

            prediction = tf.nn.softmax(classifyres, name = 'prediction')
            print('prediction: ', prediction.get_shape())


            loss_of_classify = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=real_label6, name = 'loss_of_classify'))
            print('loss_of_classify: ', loss_of_classify.get_shape())
            print('loss_of_entropy: ', loss_of_entropy.get_shape())
            loss = loss_of_classify + loss_of_entropy
            #loss = loss_of_classify #loss_of_entropy
            #loss = loss_of_classify
            print('loss: ', loss.get_shape())
            train_step1 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_of_classify)
            train_step2 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_of_entropy)

            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label6, 1))
            accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accruacy')

            merged = tf.summary.merge_all()
            # # times = 5
            # acc_list = []
            # auc_list = []

#config=tf.ConfigProto(gpu_options=gpu_options)
            with tf.Session() as sess:
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                sess.run(tf.global_variables_initializer())
                train_writer = tf.summary.FileWriter('./fully_tensorboard/', sess.graph)
                global_step = tf.Variable(0, name = 'global_step')  

                acc_list_epochs = []
                auc_list_epochs = []
                K = 5
                for tk in range(0,1):
                    K_folders = K_Cross_Split(npy_path, K)

                    steps = 0

                    for m in range(K):
                        train_filenames,test_filenames = get_train_and_test_filenames(K, m, K_folders)
                        print('K: ',m)

                        # how many time should one epoch should loop to feed all data
                        times = len(train_filenames) // self.batch_size
                        if (len(train_filenames) % self.batch_size) != 0:
                            times = times + 1
                        # loop epoches
                        for i in tqdm.tqdm(range(self.epoch)):
                            epoch_start =time.time()
                            #  the data will be shuffled by every epoch
                            random.shuffle(train_filenames)
                            for t in range(times):
                                steps += 1
                                batch_files = train_filenames[t*self.batch_size:(t+1)*self.batch_size]
                                batch_data, subtletyt, sphericityt, margint, lobulationt, spiculationt, texturet, malignancy = get_batch_withlabels(batch_files)

                                feed_dict = {x: batch_data, real_label0:subtletyt, real_label1:sphericityt, real_label2:margint, real_label3:lobulationt, real_label4:spiculationt, real_label5:texturet, real_label6:malignancy,keep_prob: self.keep_prob}
                                sess.run([train_step1, train_step2],feed_dict =feed_dict)

                                # train_writer.add_summary(summary, i)
                            saver.save(sess, './fully_ckpt/fully', steps)

                            epoch_end = time.time()

                            highcount = 0
                            lowcount = 0
                            highfile = []
                            lowfile = []
                            bothfile = []
                            for testonefile in test_filenames:
                                if 'high' in testonefile:
                                    highcount += 1
                                    highfile.append(testonefile)
                                    bothfile.append(testonefile)
                                elif 'low' in testonefile:
                                    lowcount += 1
                                    lowfile.append(testonefile)
                                    bothfile.append(testonefile)

                            print(highcount, lowcount, len(bothfile), len(test_filenames))
                            batch_data, subtletyt, sphericityt, margint, lobulationt, spiculationt, texturet, malignancy = get_batch_withlabels(bothfile)
                            feed_dict = {x: batch_data, real_label0:subtletyt, real_label1:sphericityt, real_label2:margint, real_label3:lobulationt, real_label4:spiculationt, real_label5:texturet,real_label6:malignancy, keep_prob: self.keep_prob}
                            net_loss1,net_loss2, acc = sess.run([loss_of_classify, loss_of_entropy,accruacy], feed_dict =feed_dict)
                            # print(net_loss.shape)
                            lnrt = sess.run(learning_rate, feed_dict = {global_step : steps})
                            print("net_loss: " + str(net_loss1) + ' ' + str(net_loss2))
                            print("accuracy: ", acc)
                            print("learning_rate: ", lnrt, steps)
                            f.write("accuracy: " + str(acc))
                            f.write("net_loss: " + str(net_loss1) + ' ' + str(net_loss2))
                            f.write("\n")
                            batch_data, subtletyt, sphericityt, margint, lobulationt, spiculationt, texturet, malignancy = get_batch_withlabels(highfile)
                            feed_dict = {x: batch_data, real_label0:subtletyt, real_label1:sphericityt, real_label2:margint, real_label3:lobulationt, real_label4:spiculationt, real_label5:texturet,real_label6:malignancy, keep_prob: self.keep_prob}
                            acc = sess.run([accruacy], feed_dict =feed_dict)
                            print("accuracy: "+ str(acc))
                            f.write("high accuracy: " + str(acc))
                            f.write("\n")

                            batch_data, subtletyt, sphericityt, margint, lobulationt, spiculationt, texturet, malignancy = get_batch_withlabels(lowfile)
                            feed_dict = {x: batch_data, real_label0:subtletyt, real_label1:sphericityt, real_label2:margint, real_label3:lobulationt, real_label4:spiculationt, real_label5:texturet,real_label6:malignancy, keep_prob: self.keep_prob}
                            acc = sess.run([accruacy], feed_dict =feed_dict)
                            print("accuracy: " + str(acc))
                            f.write("low accuracy: " + str(acc))
                            f.write("\n")

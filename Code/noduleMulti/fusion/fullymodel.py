# -*- coding:utf-8 -*-
'''
 the idea of this script came from LUNA2016 champion paper.
 This model conmposed of three network,namely Archi-1(size of 10x10x6),Archi-2(size of 30x30x10),Archi-3(size of 40x40x26)

input: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half. Shape [batch, in_depth, in_height, in_width, in_channels].
filter: A Tensor. Must have the same type as input. Shape [filter_depth, filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
strides: A list of ints that has length >= 5. 1-D tensor of length 5. The stride of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1.
padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
name: A name for the operation (optional).

'''
import tensorflow as tf
import random
import time
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class modelfully(object):

    def __init__(self,learning_rate,keep_prob,batch_size,epoch):
        print("modelfully network begin...")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch

        self.cubic_shape = [[10, 20, 20], [6, 20, 20], [26, 40, 40]]
        model_index = 0

        f = open('fully.txt', 'w')
        # some statistic index
        highest_acc = 0.0
        highest_iterator = 1
        
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.import_meta_graph('../august_mali/fully_ckpt/fully-80.meta')  # default to save all variable,save mode or restore from path

        # self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, tf.train.latest_checkpoint('../august_mali/fully_ckpt/'))

        
        graph = tf.get_default_graph()

        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.x = graph.get_tensor_by_name("x:0")
        self.sphericity = graph.get_tensor_by_name("sphericity:0")
        self.margin = graph.get_tensor_by_name("margin:0")
        self.lobulation = graph.get_tensor_by_name("lobulation:0")
        self.spiculation = graph.get_tensor_by_name("spiculation:0")
        self.real_label = graph.get_tensor_by_name("real_label:0")
        self.spiculation = graph.get_tensor_by_name("spiculation:0")
        self.prediction = graph.get_tensor_by_name("prediction:0")
        self.accruacy = graph.get_tensor_by_name("accruacy:0")


    def pred(self, test_batch, sphericityt, margint, lobulationt, spiculationt, test_label):
        dropkeep = 0.8
        test_dict = {self.x: test_batch, self.sphericity: sphericityt, self.margin: margint, self.lobulation: lobulationt, 
            self.spiculation: spiculationt, self.real_label:test_label, self.keep_prob:dropkeep}

        predres, aacres = self.sess.run([self.prediction, self.accruacy], feed_dict = test_dict)
        # print(predres)
        # print(test_label)
        # print(aacres)
        return predres, aacres

    def test(self):
        print('succeed')
    
    def archi_1(self,input,sphericity, margin, lobulation, spiculation, keep_prob):
        # return out_fc2
        with tf.name_scope("Archi-1"):
            # input size is batch_sizex20x20x6
            # 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3, 5, 5, 1, 64],stddev=0.01),dtype=tf.float32,name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1,keep_prob)

            # max pooling ,pooling layer has no effect on the data size
            hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,1,1,1,1],ksize=[1,1,1,1,1],padding='SAME')

            # after conv1 ,the output size is batch_sizex4x16x16x64([batch_size,in_deep,width,height,output_deep])
            w_conv2 = tf.Variable(tf.random_normal([3, 5, 5, 64, 128], stddev=0.01), dtype=tf.float32,name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[128]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)


            # after conv2 ,the output size is batch_sizex2x12x12x64([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([3, 5, 5, 128, 256], stddev=0.01), dtype=tf.float32, name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[256]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)

            w_conv4 = tf.Variable(tf.random_normal([1, 1, 1, 256, 256], stddev=0.01), dtype=tf.float32, name='w_conv4')
            b_conv4 = tf.Variable(tf.constant(0.01, shape=[256]), dtype=tf.float32, name='b_conv4')
            out_conv4 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv3, w_conv4, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv4))
            out_conv4 = tf.nn.dropout(out_conv4, keep_prob)

            out_conv3_shape = tf.shape(out_conv4)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

            # after conv2 ,the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv4 = tf.reshape(out_conv4,[-1, 256 * 8 * 8 * 4])
            w_fc1 = tf.Variable(tf.random_normal([256 * 8 * 8 * 4, 200],stddev=0.01),name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv4, w_fc1), tf.constant(0.01,shape=[200])))
            out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

            
            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([200, 2], stddev=0.01), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.01, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)

            return out_fc2

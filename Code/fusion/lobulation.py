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
from dataprepare import get_batch,get_train_and_test_filename, get_batch_withlabels, get_high_data, get_batch_withlabels_high
import random
import time
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class modellobulation(object):

    def __init__(self,learning_rate,keep_prob,batch_size,epoch):
        print(" network begin...")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch

        self.cubic_shape = [[10, 20, 20], [6, 20, 20], [26, 40, 40]]
        model_index = 0

        f = open('lobulation1.txt', 'w')
        # some statistic index
        highest_acc = 0.0
        highest_iterator = 1

        self.graph = tf.Graph()

        with self.graph.as_default():
              # keep_prob used for dropout
            self.keep_prob = tf.placeholder(tf.float32)
            # take placeholder as input
            self.x = tf.placeholder(tf.float32, [None, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2]])

            # <sphericity>3</sphericity>
            # <margin>3</margin>
            # <lobulation>3</lobulation>
            # <spiculation>4</spiculation>
            self.sphericity = tf.placeholder(tf.float32)
            self.margin = tf.placeholder(tf.float32)
            self.lobulation = tf.placeholder(tf.float32)
            self.spiculation = tf.placeholder(tf.float32)

                # X = tf.placeholder(tf.float32)
                # Y = tf.placeholder(tf.float32)

                # W = tf.Variable(tf.random_normal([1]), name='weight')
                # b = tf.Variable(tf.random_normal([1]), name='bias')
            

            self.x_image = tf.reshape(self.x, [-1, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2], 1])
            self.net_out = self.archi_lobulation(self.x_image, self.sphericity, self.margin, self.lobulation, self.spiculation, self.keep_prob)


            print("restore model")
                # softmax layer
            self.real_label = tf.placeholder(tf.float32, [None, 2])
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.net_out, labels=self.real_label)
            #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
            self.net_loss = tf.reduce_mean(self.cross_entropy)

            self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 1).minimize(self.net_loss)

            self.prediction = tf.nn.softmax(self.net_out)
            self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.real_label, 1))

            self.accruacy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.saver = tf.train.import_meta_graph('./lobulation_ckpt/fully-80.meta')  # default to save all variable,save mode or restore from path

            self.sess = tf.Session(graph = self.graph)
            # self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./lobulation_ckpt/'))
            self.sess.run(tf.global_variables_initializer())
        # saver = tf.train.import_meta_graph('ckpt/archi-1-40. ')
        #saver.restore("/ckpt/archi-1-40.data-00000-of-00001")
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # test_filenames = get_high_data(npy_path)
        print('done')

    def pred(self, test_batch, sphericityt, margint, lobulationt, spiculationt, test_label):
        dropkeep = 1
        test_dict = {self.x: test_batch, self.sphericity: sphericityt, self.margin: margint, self.lobulation: lobulationt, 
            self.spiculation: spiculationt, self.real_label:test_label, self.keep_prob:dropkeep}

        netout = self.sess.run([self.net_out], feed_dict = test_dict)

        return netout

    def test(self):
        print('succeed')
    
    def archi_lobulation(self,input,sphericity, margin, lobulation, spiculation, keep_prob):
        # with tf.name_scope("Archi-1"):
        #     # input size is batch_sizex20x20x6
        #     # 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
        #     w_conv1 = tf.Variable(tf.random_normal([3, 5, 5, 1, 64],stddev=0.01),dtype=tf.float32,name='w_conv1')
        #     b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
        #     out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
        #     out_conv1 = tf.nn.dropout(out_conv1,keep_prob)

        #     # max pooling ,pooling layer has no effect on the data size
        #     hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,2,2,2,1],ksize=[1,2,2,2,1],padding='SAME')

        #     out_conv3_shape = tf.shape(hidden_conv1)
        #     tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

        #     # after conv2 ,the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
        #     # all feature map flatten to one dimension vector,this vector will be much long
        #     hidden_conv1 = tf.reshape(hidden_conv1,[-1, 64 * 8 * 8 * 4])
        #     w_fc1 = tf.Variable(tf.random_normal([64 * 8 * 8 * 4, 150],stddev=0.01),name='w_fc1')
        #     out_fc1 = tf.nn.relu(tf.add(tf.matmul(hidden_conv1, w_fc1),tf.constant(0.01,shape=[150])))
        #     out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

        #     out_fc1_shape = tf.shape(out_fc1)
        #     tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

        #     w_fc2 = tf.Variable(tf.random_normal([150, 2], stddev=0.01), name='w_fc2')
        #     out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.01, shape=[2])))
        #     out_fc2 = tf.nn.dropout(out_fc2, keep_prob)


        # return out_fc2
        with tf.name_scope("Archi-Lobulation"):
            # input size is batch_sizex20x20x6
            # 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3, 5, 5, 1, 64],stddev=0.01),dtype=tf.float32,name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1,keep_prob)

            # max pooling ,pooling layer has no effect on the data size
            hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,1,1,1,1],ksize=[1,1,1,1,1],padding='SAME')

            # after conv1 ,the output size is batch_sizex4x16x16x64([batch_size,in_deep,width,height,output_deep])
            w_conv2 = tf.Variable(tf.random_normal([3, 5, 5, 64, 64], stddev=0.01), dtype=tf.float32,name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)


            # after conv2 ,the output size is batch_sizex2x12x12x64([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([3, 5, 5, 64, 64], stddev=0.01), dtype=tf.float32, name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)

            out_conv3_shape = tf.shape(out_conv3)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

            # after conv2 ,the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv3 = tf.reshape(out_conv3,[-1, 64 * 8 * 8 * 4])
            w_fc1 = tf.Variable(tf.random_normal([64 * 8 * 8 * 4, 150],stddev=0.01),name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3, w_fc1), tf.constant(0.01,shape=[150])))
            out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([150, 2], stddev=0.01), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.01, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)
            # sphericity, margin, lobulation, spiculation
            # w_sphericity = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_sphericity')
            # w_margin = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_margin')
            # w_lobulation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_lobulation')
            # w_spiculation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_spiculation')

            # b_sphericity = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_sphericity')
            # b_margin = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_margin')
            # b_lobulation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_lobulation')
            # b_spiculation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_spiculation')

            # out_sphericity = tf.nn.relu(tf.add(tf.matmul(sphericity, w_sphericity), b_sphericity))
            # out_margin = tf.nn.relu(tf.add(tf.matmul(margin, w_margin), b_margin))
            # out_lobulation = tf.nn.relu(tf.add(tf.matmul(lobulation, w_lobulation), b_lobulation))
            # out_spiculation = tf.nn.relu(tf.add(tf.matmul(spiculation, w_spiculation), b_spiculation))

            # out_fc2 = tf.add(out_fc2, out_sphericity)
            # out_fc2 = tf.add(out_fc2, out_margin)
            # out_fc2 = tf.add(out_fc2, out_lobulation)
            # out_fc2 = tf.add(out_fc2, out_spiculation)

            return out_fc2


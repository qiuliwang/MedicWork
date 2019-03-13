# -*- coding:utf-8 -*-
'''
sample code for restore model

Created by WangQL

1/21/2019

'''
import tensorflow as tf
import random
import time
import datetime
import os
from dataprepare import get_train_and_test_filenames, get_batch_withlabels, K_Cross_Split,get_batch_withlabels_restoremodel
import tqdm
from sklearn import svm  

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

class restoremodel(object):
    def __init__(self):
        print("'***********************\nRestore!!!\n***********************'")

        # model_index = 0

        # some statistic index
        highest_acc = 0.0
        highest_iterator = 1
        self.epoch = 40
        self.sess = tf.Session()
        self.batch_size = 64
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.import_meta_graph('/home/wangqiuli/raid/multiattri_nodule/fully_ckpt/fully-37600.meta')  
        # default to save all variable,save mode or restore from path
        self.saver.restore(self.sess, tf.train.latest_checkpoint('/home/wangqiuli/raid/multiattri_nodule/fully_ckpt/'))

        
        graph = tf.get_default_graph()

        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.x = graph.get_tensor_by_name("x:0")
        self.subtlety = graph.get_tensor_by_name("subtlety:0")
        self.sphericity = graph.get_tensor_by_name("sphericity:0")
        self.margin = graph.get_tensor_by_name("margin:0")
        self.lobulation = graph.get_tensor_by_name("lobulation:0")
        self.spiculation = graph.get_tensor_by_name("spiculation:0")
        self.texture = graph.get_tensor_by_name("texture:0")
        self.net_out = graph.get_tensor_by_name("net_out:0")

        self.real_label0 = graph.get_tensor_by_name("real_label0:0")
        self.real_label1 = graph.get_tensor_by_name("real_label1:0")      
        self.real_label2 = graph.get_tensor_by_name("real_label2:0")     
        self.real_label3 = graph.get_tensor_by_name("real_label3:0")      
        self.real_label4 = graph.get_tensor_by_name("real_label4:0")     
        self.real_label5 = graph.get_tensor_by_name("real_label5:0")

        print('***********************\nRestore Done!!!\n***********************')

    def pred(self, npydata):
        # return self.net_out(npydata, 1)
        feed_dict = {self.x: npydata, self.keep_prob: 1}
        res = self.sess.run([self.net_out, self.keep_prob], feed_dict = feed_dict)
        return res

    def inference(self, path):
        filelist = os.listdir(path)
        K = 5
        K_folders = K_Cross_Split(path, K)

        input_X1 = tf.placeholder(tf.float32, [None, 3], name = 'input_X1')
        input_X1 = tf.reshape(input_X1, [-1, 3])
        input_X_W1 = tf.Variable(tf.random_normal([3, 2], stddev = 0.01), name = 'input_X_W1')
        input_X_B1 = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='input_X_B1')
        out1 = tf.add(tf.matmul(input_X1, input_X_W1), input_X_B1)

        input_X2 = tf.placeholder(tf.float32, [None, 3], name = 'input_X2')
        input_X2 = tf.reshape(input_X2, [-1, 3])
        input_X_W2 = tf.Variable(tf.random_normal([3, 2], stddev = 0.01), name = 'input_X_W2')
        input_X_B2 = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='input_X_B2')
        out2 = tf.add(tf.matmul(input_X2, input_X_W2), input_X_B2)

        input_X3 = tf.placeholder(tf.float32, [None, 3], name = 'input_X3')
        input_X3 = tf.reshape(input_X3, [-1, 3])
        input_X_W3 = tf.Variable(tf.random_normal([3, 2], stddev = 0.01), name = 'input_X_W3')
        input_X_B3 = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='input_X_B3')
        out3 = tf.add(tf.matmul(input_X3, input_X_W3), input_X_B3)

        input_X4 = tf.placeholder(tf.float32, [None, 3], name = 'input_X4')
        input_X4 = tf.reshape(input_X4, [-1, 3])
        input_X_W4 = tf.Variable(tf.random_normal([3, 2], stddev = 0.01), name = 'input_X_W4')
        input_X_B4 = tf.Variable (tf.constant(0.01, shape=[2]), dtype=tf.float32, name='input_X_B4')
        out4 = tf.add(tf.matmul(input_X4, input_X_W4), input_X_B4)

        input_X5 = tf.placeholder(tf.float32, [None, 3], name = 'input_X5')
        input_X5 = tf.reshape(input_X5, [-1, 3])
        input_X_W5 = tf.Variable(tf.random_normal([3, 2], stddev = 0.01), name = 'input_X_W5')
        input_X_B5 = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='input_X_B5')
        out5 = tf.add(tf.matmul(input_X5, input_X_W5), input_X_B5)

        input_X6 = tf.placeholder(tf.float32, [None, 3], name = 'input_X6')
        input_X6 = tf.reshape(input_X6, [-1, 3])
        input_X_W6 = tf.Variable(tf.random_normal([3, 2], stddev = 0.01), name = 'input_X_W6')
        input_X_B6 = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='input_X_B6')
        out6 = tf.add(tf.matmul(input_X6, input_X_W6), input_X_B6)
    
        input_X7 = tf.placeholder(tf.float32, [None, 3], name = 'input_X7')
        input_X7 = tf.reshape(input_X7, [-1, 3])
        input_X_W7 = tf.Variable(tf.random_normal([3, 2], stddev = 0.01), name = 'input_X_W7')
        input_X_B7 = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='input_X_B7')
        out7 = tf.add(tf.matmul(input_X7, input_X_W7), input_X_B7)

        w1 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w1')
        w2 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w2')
        w3 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w3')
        w4 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w4')
        w5 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w5')
        w6 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w6')
        w7 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w7')
        
        res = tf.matmul(out1, w1) + tf.matmul(out2, w2) + tf.matmul(out3, w3)+ tf.matmul(out4, w4)+ tf.matmul(out5, w5)+ tf.matmul(out6, w6) + tf.matmul(out7, w7)
        second_net_out = tf.nn.sigmoid(res)

        print('second_net_out: ',second_net_out.get_shape())

        # # softmax layer
        sec_real_label = tf.placeholder(tf.float32, [None, 2], name = 'sec_real_label')
        print('real_label: ', sec_real_label.get_shape())
        sec_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=second_net_out, labels=sec_real_label, name = 'second_net_out')
        # #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
        sec_net_loss = tf.reduce_mean(sec_cross_entropy)

        train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(sec_net_loss)

        prediction = tf.nn.sigmoid(second_net_out, name = 'prediction')
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(sec_real_label, 1))
        accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accruacy')

        # _,auc =  tf.metrics.auc(tf.cast(tf.argmax(prediction, 1),tf.float32),tf.cast(tf.argmax(real_label, 1),tf.float32))

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(tf.global_variables_initializer())

        for m in range(K):
            train_filenames,test_filenames = get_train_and_test_filenames(K, m, K_folders)
            print('K: ',len(train_filenames))
            times = len(train_filenames) // self.batch_size
            if (len(train_filenames) % self.batch_size) != 0:
                times = times + 1
            # loop epoches
            for i in tqdm.tqdm(range(self.epoch)):
            # for i in tqdm.tqdm(range(2)):
                epoch_start =time.time()
                #  the data will be shuffled by every epoch
                random.shuffle(train_filenames)
                for t in range(times):
                    batch_files = train_filenames[t*self.batch_size:(t+1)*self.batch_size]
                    batchdata, batchlabel = get_batch_withlabels_restoremodel(batch_files)
                    batchx = self.pred(batchdata)
                    batchdata = batchx[0]

                    batchvol1 = batchdata[0]
                    batchvol2 = batchdata[1]
                    batchvol3 = batchdata[2]
                    batchvol4 = batchdata[3]
                    batchvol5 = batchdata[4]
                    batchvol6 = batchdata[5]

                    feed_dict = {input_X1:batchvol1, input_X2:batchvol2, input_X3:batchvol3, input_X4:batchvol4, input_X5:batchvol5, input_X6:batchvol6, sec_real_label: batchlabel}

                    _ = sess.run(train_step, feed_dict = feed_dict)

                    # print(batchdata.shape) #(64, 6, 5)
                
                batch_files = test_filenames
                batchdata, batchlabel = get_batch_withlabels_restoremodel(batch_files)
                batchx = self.pred(batchdata)
                batchdata = batchx[0]

                batchvol1 = batchdata[0]
                batchvol2 = batchdata[1]
                batchvol3 = batchdata[2]
                batchvol4 = batchdata[3]
                batchvol5 = batchdata[4]
                batchvol6 = batchdata[5]

                feed_dict = {input_X1: batchvol1, input_X2: batchvol2, input_X3: batchvol3, input_X4: batchvol4, 
                    input_X5: batchvol5, input_X6: batchvol6, sec_real_label: batchlabel}

                loss, acc = sess.run([sec_net_loss, accruacy], feed_dict = feed_dict)
                print('loss: ', loss)
                print('acc: ', acc)

'''
Created by Wang Qiu Li

fusion model 

7/15/2018

'''

import tensorflow as tf
from dataprepare import get_batch,get_train_and_test_filename, get_batch_withlabels, get_high_data, get_batch_withlabels_high
import random
import time
import datetime
from lobulation import model

class fusion(object):
    def __init__(self, ckpt_path, meta_name):

        print('restore ', ckpt_path)
        self.keep_prob = 1
        self.graph=tf.Graph()
        with self.graph.as_default():
             self.saver=tf.train.import_meta_graph(ckpt_path + meta_name)
        self.sess=tf.Session(graph=self.graph)
        with self.sess.as_default():
             with self.graph.as_default():
                 self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_path))


        tempmodel = model(1,1,32,80)
        
        
        x = tf.placeholder(tf.float32, [None, 10, 20, 20])
        x_image = tf.reshape(x, [-1, 10, 20, 20, 1])

        res = tf.placeholder(tf.float32, [None, 2])
        keep_prob = tf.placeholder(tf.float32)
        sphericity = tf.placeholder(tf.float32)
        margin = tf.placeholder(tf.float32)
        lobulation = tf.placeholder(tf.float32)
        spiculation = tf.placeholder(tf.float32)
        net_out = tempmodel.archi_1(x_image, sphericity, margin, lobulation, spiculation, 1)
        real_label = tf.placeholder(tf.float32, [None, 2])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label)
        #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
        net_loss = tf.reduce_mean(cross_entropy)

        train_step = tf.train.MomentumOptimizer(self.learning_rate, 1).minimize(net_loss)

        prediction = tf.nn.softmax(net_out)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
        accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())

    def predict(self,test_batch, sphericityt, margint, lobulationt, spiculationt, test_label):

        # self.sess.run(tf.global_variables_initializer())
        test_dict = {x: test_batch, sphericity: sphericityt, margin: margint, lobulation: lobulationt, 
            spiculation: spiculationt, real_label: test_label, keep_prob:1}

        acc_test = self.sess.run(res, feed_dict=test_dict)
        print('accuracy  is %f' % acc_test)
# -*- coding:utf-8 -*-
'''
this is the enterance of this project
'''

import tensorflow as tf
import os
# from fusionmodel import fusion
import numpy as np
import random
from lobulation import modellobulation
from spiculation import modelspiculation
from sphercity import modelsphercity
from fully import modelfully

from dataprepare import get_batch,get_train_and_test_filename, get_batch_withlabels, get_high_data, get_batch_withlabels_high


if __name__ =='__main__':
    batch_size =32
    learning_rate = 0.01
    keep_prob = 1
    epoch = 5
    # path = '/data0/LUNA/cubic_normalization_npy'
    path = 'NPY/'

    train_filenames,test_filenames = get_train_and_test_filename(path, 0.1, 121)
    times = len(train_filenames) // batch_size
    if (len(train_filenames) % batch_size) != 0:
        times = times + 1

    templobulation = modellobulation(learning_rate, 1, batch_size, epoch)
    tempspiculation =  modelspiculation(learning_rate, 1, batch_size, epoch)
    tempsphercity =  modelsphercity(learning_rate, 1, batch_size, epoch)
    tempfully = modelfully(learning_rate, 1, batch_size, epoch)
# lobulation_ckpt
    # lobulationrestore = fusion('lobulation_ckpt/', 'fully-80.meta')

    inputsphercity = tf.placeholder(tf.float32, [None, 2])
    inputlobulation = tf.placeholder(tf.float32, [None, 2])
    inputspiculation = tf.placeholder(tf.float32, [None, 2])
    inputfully = tf.placeholder(tf.float32, [None, 2])


    w_sphericity = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_sphericity')
    w_fully = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_fully')
    w_lobulation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_lobulation')
    w_spiculation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_spiculation')

    b_sphericity = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_sphericity')
    b_fully = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_fully')
    b_lobulation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_lobulation')
    b_spiculation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_spiculation')

    out_sphericity = tf.nn.relu(tf.add(tf.matmul(inputsphercity, w_sphericity), b_sphericity))
    out_fully = tf.nn.relu(tf.add(tf.matmul(inputfully, w_fully), b_fully))
    out_lobulation = tf.nn.relu(tf.add(tf.matmul(inputlobulation, w_lobulation), b_lobulation))
    out_spiculation = tf.nn.relu(tf.add(tf.matmul(inputspiculation, w_spiculation), b_spiculation))

    net_out = tf.relu(tf.add (tf.add(out_fully , out_lobulation), tf.add(out_sphericity, out_spiculation)))

    real_label = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label))
    #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
    net_loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(net_loss)

    correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(real_label, 1))
    accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(epoch):
        print(i)
        random.shuffle(train_filenames)
        for t in range(times):
            batch_files = train_filenames[t * batch_size:(t+1) * batch_size]
            batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(batch_files)
            out1 = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            # out2 = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            

            # lobulationrestore.predict(batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            # test_dict = {x: batch_data, sphercity: sphercityt, margin: margint, lobulation: lobulationt, spiculation: spiculationt, real_label: batch_label, keep_prob:1}

            # acc_test,loss, aucpred = sess.run([accruacy,net_loss, prediction],feed_dict=test_dict)
            # aucscore = tf.metric.auc(test_label, prediction)
            # print('accuracy  is %f' % acc_test)


    #test_path = '../../data/cubic_normalization_test'
    # test_size = 0.1
    # seed=121

    # print(" begin...")
    # model = model(learning_rate,keep_prob,batch_size,80)
    # model.inference(path,0, test_size, seed, True)

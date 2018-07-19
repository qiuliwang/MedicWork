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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ =='__main__':
    batch_size =32
    learning_rate = 0.1
    keep_prob = 1
    epoch = 40
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

    net_out = tf.nn.relu(out_fully + out_lobulation + out_sphericity + out_spiculation)

    # net_out = tf.nn.relu(tf.add (tf.add(out_fully , out_lobulation), tf.add(out_sphericity, out_spiculation)))

    real_label = tf.placeholder(tf.float32, [None, 2])

    # print('net out ', net_out.shape)
    # print('real',real_label.shape)
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label))
    #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
    net_loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.AdadeltaOptimizer(learning_rate, 0.9).minimize(net_loss)
    prediction = tf.nn.softmax(net_out)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
    accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    times = 1
    #        with tf.Session() as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('training times: ', times)
        for i in range(epoch):
            print(i)
            random.shuffle(train_filenames)
            for t in range(times):
                batch_files = train_filenames[t * batch_size:(t+1) * batch_size]
                batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(batch_files)
                out_lobulation = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                out_spiculation = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                out_sphericity = tempsphercity.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                out_fully = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                # print('====')
                # print(len(out_lobulation))
                # print(out_lobulation[0])
                # print(out_spiculation[0])
                # print(out_sphericity[0])
                # print(out_fully[0])

                # print(batch_label.shape)
                feed_dict = {inputsphercity: out_sphericity[0], inputlobulation: out_lobulation[0], inputspiculation: out_spiculation[0],
                            inputfully: out_fully[0], real_label: batch_label}
                res, _ = sess.run([net_out, train_step],feed_dict =feed_dict)
                # print('res shape ', res.shape)


            batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(test_filenames)

            out_lobulation = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            out_spiculation = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            out_sphericity = tempsphercity.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            out_fully = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)

            test_dict = {inputsphercity: out_sphericity[0], inputlobulation: out_lobulation[0], inputspiculation: out_spiculation[0],
                            inputfully: out_fully[0], real_label: batch_label}
            # print(len(out_sphericity))
            # print(out_sphericity[0].shape)


            countx = 0
            for one in test_filenames:
                if 'low' in one:
                    countx += 1
            print('precent: ', countx / len(test_filenames))
            acc_test,loss = sess.run([accruacy,net_loss],feed_dict=test_dict)
            print('acc is %f, loss it %f '%(acc_test, loss))



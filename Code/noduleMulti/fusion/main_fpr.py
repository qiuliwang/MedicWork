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
from subtlety import modelsubtlety
from fprmodel import modelfpr

from dataprepare import get_batch,get_train_and_test_filename, get_batch_withlabels, get_high_data, get_batch_withlabels_high
from data_prepare_fpr import get_batch,get_train_and_test_filename_fpr
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ =='__main__':
    batch_size =32
    learning_rate = 0.1
    keep_prob = 1
    epoch = 40
    # path = '/data0/LUNA/cubic_normalization_npy'
    path_fpr = '/data0/LUNA/fpr_cubic_normalization_npy/'
    path_mal = 'NPY/'

    train_filenames,test_filenames = get_train_and_test_filename_fpr(path_fpr, 0.1, 121)
    print('traing: ', len(train_filenames))
    print('test: ', len(test_filenames))
    times = len(train_filenames) // batch_size
    if (len(train_filenames) % batch_size) != 0:
        times = times + 1

    train_filenames_mal,test_filenames_mal = get_train_and_test_filename(path_mal, 0.2, 121)
    print('traing: ', len(train_filenames_mal))
    print('test: ', len(test_filenames_mal))
    times_fpr = len(train_filenames_mal) // batch_size
    if (len(test_filenames_mal) % batch_size) != 0:
        times_fpr = times_fpr + 1

    templobulation = modellobulation(learning_rate, 1, batch_size, epoch)
    tempspiculation =  modelspiculation(learning_rate, 1, batch_size, epoch)
    tempsubtlety =  modelsubtlety(learning_rate, 1, batch_size, epoch)
    tempfully = modelfpr(learning_rate, 1, batch_size, epoch)

# lobulation_ckpt
    # lobulationrestore = fusion('lobulation_ckpt/', 'fully-80.meta')

    inputsubtlety = tf.placeholder(tf.float32, [None, 2])
    inputlobulation = tf.placeholder(tf.float32, [None, 2])
    inputspiculation = tf.placeholder(tf.float32, [None, 2])
    inputfully = tf.placeholder(tf.float32, [None, 2])
    
    testsubtlety = tf.nn.sigmoid(inputsubtlety)
    testlobulation = tf.nn.softmax(inputlobulation)
    testspiculation = tf.nn.softmax(inputspiculation)
    testfully = tf.nn.sigmoid(inputfully)

    w_subtlety = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_subtlety')
    w_fully = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_fully')
    w_lobulation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_lobulation')
    w_spiculation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_spiculation')

    b_subtlety = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_subtlety')
    b_fully = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_fully')
    b_lobulation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_lobulation')
    b_spiculation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_spiculation')

    out_subtlety = (tf.add(tf.matmul(testsubtlety, w_subtlety), b_subtlety))
    out_fully = (tf.add(tf.matmul(testfully, w_fully), b_fully))
    out_lobulation = (tf.add(tf.matmul(testlobulation, w_lobulation), b_lobulation))
    out_spiculation = (tf.add(tf.matmul(testspiculation, w_spiculation), b_spiculation))

    net_out = (out_fully + out_lobulation + out_subtlety + out_spiculation)

    # net_out = tf.nn.relu(tf.add (tf.add(out_fully , out_lobulation), tf.add(out_subtlety, out_spiculation)))

    real_label = tf.placeholder(tf.float32, [None, 2])

    # print('net out ', net_out.shape)
    # print('real',real_label.shape)
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label))
    #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
    net_loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(net_loss)
    prediction = tf.nn.softmax(net_out)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
    accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _,auc =  tf.metrics.auc(tf.cast(tf.argmax(prediction, 1),tf.float32),tf.cast(tf.argmax(real_label, 1),tf.float32))

    #        with tf.Session() as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('training times: ', times)
        for i in range(epoch):
            print(i)
            random.shuffle(train_filenames)
            random.shuffle(train_filenames_mal)
            for t in range(times):
                batch_files = train_filenames[t * batch_size:(t+1) * batch_size]
                batch_files_mal = train_filenames_mal[t * batch_size:(t+1) * batch_size]
                batch_data, batch_label = get_batch(batch_files)
                batch_data_mal, sphercityt, margint, lobulationt, spiculationt, batch_label_mal = get_batch_withlabels(batch_files_mal)
                print("batch_data_shape:",batch_data.shape)
                print("fpr_label_shape:",batch_label.shape)
                print("mal_label_shape:",batch_label.shape)
                # predres, accres
                lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                fullypred, fullyacc = tempfully.pred(batch_data, batch_label)
                # print('====')
                # print(len(out_lobulation))

                # print(batch_label.shape)

                feed_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, inputsubtlety: subtletypred,
                            inputfully: fullypred, real_label: batch_label}
                _ = sess.run([train_step],feed_dict =feed_dict)
                
                # print(out_fully)
                # print('XXXXXXXXX\n', resfully)
                # print('XXXXXXXXX\n', res)
#                print(test1)
                # print(test2)
                if t % 10 == 0:
                    print(t, times)

            batch_data, batch_label = get_batch(test_filenames)
            batch_data_mal, sphercityt, margint, lobulationt, spiculationt, batch_label_mal = get_batch_withlabels(test_filenames_mal)

            lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            fullypred, fullyacc = tempfully.pred( batch_data, batch_label)

            test_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, 
                inputsubtlety: subtletypred, inputfully: fullypred, real_label: batch_label}
            # print(len(out_subtlety))
            # print(out_subtlety[0].shape)


            countx = 0
            for one in test_filenames:
                if 'real' in one:
                    countx += 1
            print('precent: ', countx / len(test_filenames))
            acc_test,loss,aucscore = sess.run([accruacy,net_loss, auc],feed_dict=test_dict)
            print('acc is %f, loss is %f , auc is %f '%(acc_test, loss, aucscore))

            # lowfile = []
            # highfile = []
            # for one in test_filenames:
            #     if 'real' in one:
            #         highfile.append(one)
            #     else:
            #         lowfile.append(one)
            
            # batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(lowfile)

            # lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            # spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            # subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            # fullypred, fullyacc = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)

            # test_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, 
            #     inputsubtlety: subtletypred, inputfully: fullypred, real_label: batch_label}
            # acc_test,loss,aucscore = sess.run([accruacy,net_loss, auc],feed_dict=test_dict)
            # print('spc acc is %f, loss is %f , auc is %f '%(acc_test, loss, aucscore))
        
            # batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(highfile)

            # lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            # spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            # subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
            # fullypred, fullyacc = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)

            # test_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, 
            #     inputsubtlety: subtletypred, inputfully: fullypred, real_label: batch_label}
            # acc_test,loss,aucscore = sess.run([accruacy,net_loss, auc],feed_dict=test_dict)
            # print('sen acc is %f, loss is %f , auc is %f '%(acc_test, loss, aucscore))
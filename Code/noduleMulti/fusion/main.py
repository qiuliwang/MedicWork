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
from fullymodel import modelfully
from fprmodel import modelfpr

from dataprepare import get_batch,get_train_and_test_filename, get_batch_withlabels, get_high_data, get_batch_withlabels_high
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ =='__main__':
    batch_size =32
    learning_rate = 0.1
    keep_prob = 1
    epoch = 40
    # path = '/data0/LUNA/cubic_normalization_npy'
    path = 'npy_and_fpr_data/'

    train_filenames,test_filenames = get_train_and_test_filename(path, 0.2, 121)
    print('traing: ', len(train_filenames))
    print('test: ', len(test_filenames))
    times = len(train_filenames) // batch_size
    if (len(train_filenames) % batch_size) != 0:
        times = times + 1

    templobulation = modellobulation(learning_rate, 1, batch_size, epoch)
    tempspiculation =  modelspiculation(learning_rate, 1, batch_size, epoch)
    tempsubtlety =  modelsubtlety(learning_rate, 1, batch_size, epoch)
    tempfully = modelfully(learning_rate, 1, batch_size, epoch)
    tempfpr = modelfpr(learning_rate, 1, batch_size, epoch)

# lobulation_ckpt
    # lobulationrestore = fusion('lobulation_ckpt/', 'fully-80.meta')

    inputsubtlety = tf.placeholder(tf.float32, [None, 2])
    inputlobulation = tf.placeholder(tf.float32, [None, 2])
    inputspiculation = tf.placeholder(tf.float32, [None, 2])
    inputfully = tf.placeholder(tf.float32, [None, 2])
    inputfpr = tf.placeholder(tf.float32, [None, 2])
    
    # testsubtlety = tf.nn.sigmoid(inputsubtlety)
    # testlobulation = tf.nn.softmax(inputlobulation)
    # testspiculation = tf.nn.softmax(inputspiculation)
    # testfully = tf.nn.sigmoid(inputfully)
    # testfpr = tf.nn.sigmoid(inputfpr)

    w_subtlety = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_subtlety')
    w_fully = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_fully')
    w_lobulation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_lobulation')
    w_spiculation = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_spiculation')
    w_fpr = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_fpr')

    b_subtlety = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_subtlety')
    b_fully = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_fully')
    b_lobulation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_lobulation')
    b_spiculation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_spiculation')
    b_fpr = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_fpr')

    out_subtlety = (tf.add(tf.matmul(inputsubtlety, w_subtlety), b_subtlety))
    out_fully = (tf.add(tf.matmul(inputfully, w_fully), b_fully))
    out_lobulation = (tf.add(tf.matmul(inputlobulation, w_lobulation), b_lobulation))
    out_spiculation = (tf.add(tf.matmul(inputspiculation, w_spiculation), b_spiculation))
    out_fpr = (tf.add(tf.matmul(inputfpr, w_fpr), b_fpr))

    net_out_mal = (out_fully + out_lobulation + out_subtlety + out_spiculation)
    net_out_fpr= (out_fpr + out_lobulation + out_subtlety + out_spiculation)

    # net_out = tf.nn.relu(tf.add (tf.add(out_fully , out_lobulation), tf.add(out_subtlety, out_spiculation)))

    real_label_mal = tf.placeholder(tf.float32, [None, 2])
    real_label_fpr = tf.placeholder(tf.float32, [None, 2])

    # print('net out ', net_out.shape)
    # print('real',real_label.shape)
    cross_entropy_mal = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out_mal, labels=real_label_mal))
    cross_entropy_fpr = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out_fpr, labels=real_label_fpr))
    #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
    net_loss_mal = tf.reduce_mean(cross_entropy_mal)
    net_loss_fpr = tf.reduce_mean(cross_entropy_fpr)

    train_step_mal = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(net_loss_mal)
    print("=======================",net_out_mal)
    print("======================================",real_label_mal)
    prediction_mal = tf.nn.softmax(net_out_mal)
    train_step_fpr = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(net_loss_fpr)
    prediction_fpr = tf.nn.softmax(net_out_fpr)

    correct_prediction_mal = tf.equal(tf.argmax(prediction_mal, 1), tf.argmax(real_label_mal, 1))
    accruacy_mal = tf.reduce_mean(tf.cast(correct_prediction_mal, tf.float32))
    _,auc_mal =  tf.metrics.auc(tf.cast(tf.argmax(prediction_mal, 1),tf.float32),tf.cast(tf.argmax(real_label_mal, 1),tf.float32))
    correct_prediction_fpr = tf.equal(tf.argmax(prediction_fpr, 1), tf.argmax(real_label_fpr, 1))
    accruacy_fpr = tf.reduce_mean(tf.cast(correct_prediction_fpr, tf.float32))
    _,auc_fpr =  tf.metrics.auc(tf.cast(tf.argmax(prediction_fpr, 1),tf.float32),tf.cast(tf.argmax(real_label_fpr, 1),tf.float32))

    #        with tf.Session() as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('training times: ', times)
        for i in range(epoch):
            print(i)
            random.shuffle(train_filenames)
            for t in range(times):
                batch_files = train_filenames[t * batch_size:(t+1) * batch_size]
                batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_0, batch_label_1 = get_batch_withlabels(batch_files)
                # predres, accres
                lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_1)
                spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_1)
                subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_1)
                fullypred, fullyacc = tempfully.pred(batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_1)
                fprpred, fpracc = tempfpr.pred(batch_data, batch_label_0)
                # print('====')
                # print(len(out_lobulation))

                # print(batch_label.shape)

                feed_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, inputsubtlety: subtletypred,
                            inputfully: fullypred, inputfpr: fprpred,real_label_mal: batch_label_1,real_label_fpr:batch_label_0}
                _ = sess.run([train_step_mal,train_step_fpr],feed_dict =feed_dict)
                
                # print(out_fully)
                # print('XXXXXXXXX\n', resfully)
                # print('XXXXXXXXX\n', res)
#                print(test1)
                # print(test2)
                if t % 10 == 0:
                    print(t, times)

            batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_0, batch_label_1= get_batch_withlabels(test_filenames)

            lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_1)
            spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_1)
            subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_1)
            fullypred, fullyacc = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label_1)
            fprpred, fpracc = tempfpr.pred(batch_data, batch_label_0)

            test_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, 
                inputsubtlety: subtletypred, inputfully: fullypred, inputfpr: fprpred,real_label_mal: batch_label_1,real_label_fpr:batch_label_0}
            # print(len(out_subtlety))
            # print(out_subtlety[0].shape)


            countx = 0;county = 0
            for one in test_filenames:
                if 'high' in one:
                    countx += 1
            print('high precent: ', countx / len(test_filenames))
            for one in test_filenames:
                if 'fake' in one:
                    county += 1
            print('fake precent: ', county / len(test_filenames))
            acc_test_mal,loss_mal,aucscore_mal,acc_test_fpr,loss_fpr,aucscore_fpr = sess.run([accruacy_mal,net_loss_mal, auc_mal,accruacy_fpr,net_loss_fpr, auc_fpr],feed_dict=test_dict)
            print('acc_mal is %f, loss_mal is %f , auc_mal is %f '%(acc_test_mal, loss_mal, aucscore_mal))
            print('acc_fpr is %f, loss_fpr is %f , auc_fpr is %f '%(acc_test_fpr, loss_fpr, aucscore_fpr))

            # lowfile = []
            # highfile = []
            # for one in test_filenames:
            #     if 'high' in one:
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
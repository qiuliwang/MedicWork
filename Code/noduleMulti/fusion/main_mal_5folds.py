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
import time
import datetime

from dataprepare import get_batch,get_train_and_test_filename, get_batch_withlabels,get_train_and_test_filenamex,get_train_and_test_filenames,K_Cross_Split
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ =='__main__':        
    f = open('fusion0001X.txt', 'w')
    
    global_step = tf.Variable(0, name = 'global_step')  

    batch_size =64
    keep_prob = 1
    epoch = 40
    learning_rate = tf.train.exponential_decay(0.001, global_step * batch_size, 100, 1, staircase=True)

    # path = '/data0/LUNA/cubic_normalization_npy'
    path = '/raid/data/wangqiuli/Documents/noduleMulti/august_mali/train'
    # pathtest = '/home/wangqiuli/Documents/august_mali/test/'
    # train_filenames, test_filenames = get_train_and_test_filenamex(path, 0.3, 121)
    # # test_filenames = get_train_and_test_filenames(pathtest)

    # print('traing: ', len(train_filenames))
    # print('test: ', len(test_filenames))
    # times = len(train_filenames) // batch_size
    # if (len(train_filenames) % batch_size) != 0:
    #     times = times + 1

    templobulation = modellobulation(0.01, 1, batch_size, epoch)
    tempspiculation =  modelspiculation(0.01, 1, batch_size, epoch)
    tempsubtlety =  modelsubtlety(0.01, 1, batch_size, epoch)
    tempfully = modelfully(0.01, 1, batch_size, epoch)

# lobulation_ckpt
    # lobulationrestore = fusion('lobulation_ckpt/', 'fully-80.meta')

    inputsubtlety = tf.placeholder(tf.float32, [None, 2])
    inputlobulation = tf.placeholder(tf.float32, [None, 2])
    inputspiculation = tf.placeholder(tf.float32, [None, 2])
    inputfully = tf.placeholder(tf.float32, [None, 2])
    
    # testsubtlety = tf.nn.sigmoid(inputsubtlety)
    # testlobulation = tf.nn.softmax(inputlobulation)
    # testspiculation = tf.nn.softmax(inputspiculation)
    # testfully = tf.nn.sigmoid(inputfully)

    w_subtlety1 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_subtlety1')
    w_subtlety2 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_subtlety2')
    w_fully1 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_fully1')
    w_fully2 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_fully2')
    w_lobulation1 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_lobulation1')
    w_lobulation2 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_lobulation2')
    w_spiculation1 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_spiculation1')
    w_spiculation2 = tf.Variable(tf.random_normal([2, 2], stddev = 0.01), name = 'w_spiculation2')

    b_subtlety = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_subtlety')
    b_fully = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_fully')
    b_lobulation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_lobulation')
    b_spiculation = tf.Variable(tf.constant(0.01, shape=[2]), dtype=tf.float32, name='b_spiculation')

    out_subtlety = tf.add(tf.add(tf.matmul(tf.pow(inputsubtlety, 2.0), w_subtlety1), tf.matmul(inputsubtlety, w_subtlety2)), b_subtlety)
    out_fully = tf.add(tf.add(tf.matmul(tf.pow(inputfully, 2.0), w_fully1), tf.matmul(inputfully, w_fully2)), b_fully)
    out_lobulation = tf.add(tf.add(tf.matmul(tf.pow(inputlobulation, 2.0), w_lobulation1), tf.matmul(inputlobulation, w_lobulation2)), b_lobulation)
    out_spiculation = tf.add(tf.add(tf.matmul(tf.pow(inputspiculation, 2.0), w_spiculation1), tf.matmul(inputspiculation, w_spiculation2)), b_spiculation)
    
    
    # Y_pred = tf.add(tf.add(tf.multiply(tf.pow(X, 2.0), W1), tf.multiply(X, W2)), b)

    net_out = (out_fully + out_lobulation + out_subtlety + out_spiculation)
    # net_out = out_fully
    # net_out = tf.nn.relu(tf.add (tf.add(out_fully , out_lobulation), tf.add(out_subtlety, out_spiculation)))

    real_label = tf.placeholder(tf.float32, [None, 2])

    # print('net out ', net_out.shape)
    # print('real',real_label.shape)
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label))
    #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
    net_loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(net_loss, global_step=global_step)
    # print("=======================",net_out)
    # print("======================================",real_label)
    prediction = tf.nn.softmax(net_out)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
    accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _,auc =  tf.metrics.auc(tf.cast(tf.argmax(prediction, 1),tf.float32),tf.cast(tf.argmax(real_label, 1),tf.float32))

    #        with tf.Session() as sess:
    acc_list = []
    auc_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        acc_list_epochs = []
        auc_list_epochs = []

        sign = 0
        K = 5
        K_folders = K_Cross_Split(path, K)

        for m in range(K):
            train_filenames,test_filenames = get_train_and_test_filenames(K, m, K_folders)
            # how many time should one epoch should loop to feed all data
            times = len(train_filenames) // batch_size
            if (len(train_filenames) % batch_size) != 0:
                times = times + 1
                    # loop epoches
            for i in range(epoch):
                epoch_start =time.time()
                        #  the data will be shuffled by every epoch
                random.shuffle(train_filenames)
                for t in range(times):
                    sign += 1
                    batch_files = train_filenames[t * batch_size:(t+1) * batch_size]
                    batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(batch_files)
                    # predres, accres
                    lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    fullypred, fullyacc = tempfully.pred(batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    # print('====')
                    # print(len(out_lobulation))

                    # print(batch_label.shape)
                    # print('Learning rate: ', learning_rate)
                    # print(type(fullypred))
                    feed_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, inputsubtlety: subtletypred,
                                inputfully: fullypred, real_label: batch_label}
                    _ = sess.run([train_step],feed_dict =feed_dict)
                    lnrt = sess.run(learning_rate, feed_dict = {global_step: sign})
                    
                    # print('XXXXXXXXX\n', resfully)
                    # print('XXXXXXXXX\n', res)
                    # print(test2)
                    if t % 10 == 0:
                        print(lnrt, t, times)

                epoch_end = time.time()

                batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(test_filenames)

                lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                fullypred, fullyacc = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)

                test_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, 
                    inputsubtlety: subtletypred, inputfully: fullypred, real_label: batch_label}
                # print(len(out_subtlety))
                # print(out_subtlety[0].shape)


                countx = 0
                for one in test_filenames:
                    if 'high' in one:
                        countx += 1
                print('precent: ', countx / len(test_filenames))
                acc_test_single,loss,auc_score_single = sess.run([accruacy,net_loss,auc],feed_dict=test_dict)
                acc_list_epochs.append(acc_test_single)
                auc_list_epochs.append(auc_score_single)
                print('accuracy  is %f ' % acc_test_single)
                f.write('accuracy  is %f ' % acc_test_single)

                print('loss is ', loss)
                f.write('loss is %f ' % loss)
                print('auc_score is ', auc_score_single)
                f.write('auc_score  is %f ' % auc_score_single)
                print("epoch %d time consumed %f seconds "%(i,(epoch_end-epoch_start)))
                f.write("epoch %d time consumed %f seconds "%(i,(epoch_end-epoch_start)))
                # if acc_test  > highest_acc:
                #     highest_acc = acc_test
                #     highest_iterator = i
                f.write('\n')
            ave_acc_epochs = sum(acc_list_epochs[-10:-1])/len(acc_list_epochs[-10:-1])
            ave_auc_epochs = sum(auc_list_epochs[-10:-1])/len(auc_list_epochs[-10:-1])
            print('epoch average accuracy  is %f ' % ave_acc_epochs)
            print('epoch average auc  is %f ' % ave_auc_epochs)
            acc_list.append(ave_acc_epochs)
            auc_list.append(ave_auc_epochs)
            print("acc_list",acc_list)
            print("auc_list",auc_list)

        acc_test = sum(acc_list)/len(acc_list)
        print('average accuracy  is %f ' % acc_test)
        f.write('average accuracy  is %f ' % acc_test)
        auc_test = sum(auc_list)/len(auc_list)
        print('average auc  is %f ' % auc_test)
        f.write('average auc  is %f ' % auc_test)

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
                # f.write('spc acc is %f, loss is %f , auc is %f \n'%(acc_test, loss, aucscore))

                # batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(highfile)

                # lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                # spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                # subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                # fullypred, fullyacc = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)

                # test_dict = {inputspiculation: spiculationpred, inputlobulation: lobulationpred, 
                #     inputsubtlety: subtletypred, inputfully: fullypred, real_label: batch_label}
                # acc_test,loss,aucscore = sess.run([accruacy,net_loss, auc],feed_dict=test_dict)
                # print('sen acc is %f, loss is %f , auc is %f '%(acc_test, loss, aucscore))
                # f.write('sen acc is %f, loss is %f , auc is %f \n'%(acc_test, loss, aucscore))

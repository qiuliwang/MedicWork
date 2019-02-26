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
from margin import modelmargin
 

import time
import datetime

from dataprepare import get_batch,get_train_and_test_filename, get_batch_withlabels,get_train_and_test_filenamex
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

if __name__ =='__main__':        
    f = open('fusion09.txt', 'w')
    
    global_step = tf.Variable(0, name = 'global_step')  

    batch_size =64
    keep_prob = 1
    epoch = 40
    learning_rate = tf.train.exponential_decay(0.005, global_step * batch_size, 64*40*81, 1, staircase=True)

    # path = '/data0/LUNA/cubic_normalization_npy'
    path = 'fdata/'
    # pathtest = '/home/wangqiuli/Documents/august_mali/test/'
    train_filenames, test_filenames = get_train_and_test_filenamex(path, 0.1, 121)
    # # test_filenames = get_train_and_test_filenames(pathtest)

    print('traing: ', len(train_filenames))
    print('test: ', len(test_filenames))
    times = len(train_filenames) // batch_size
    if (len(train_filenames) % batch_size) != 0:
        times = times + 1

    templobulation = modellobulation(0.01, 1, batch_size, epoch)
    tempspiculation =  modelspiculation(0.01, 1, batch_size, epoch)
    tempsubtlety =  modelsubtlety(0.01, 1, batch_size, epoch)
    tempfully = modelfully(0.01, 1, batch_size, epoch)
    tempfpr = modelfpr(0.01, 1, batch_size, epoch)

    inputplace = tf.placeholder(tf.float32, [None, 10])

    w_fc1 = tf.Variable(tf.random_normal([10, 10],stddev=0.01),name='w_fc1')
    out_fc1 = tf.nn.relu(tf.add(tf.matmul(inputplace, w_fc1), tf.constant(0.01,shape=[10])))
    out_fc1 = tf.nn.dropout(out_fc1,1)

    w_fc2 = tf.Variable(tf.random_normal([10, 2], stddev=0.01), name='w_fc2')
    out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.01, shape=[2])))
    net_out = tf.nn.dropout(out_fc2, 1)


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
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        acc_list_epochs = []
        auc_list_epochs = []
        K = 5
        sign = 0
        for m in range(K):
            train_filenames,test_filenames = get_train_and_test_filename(path,K,m)
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
                    fprpred, fpracc = tempfpr.pred(batch_data, batch_label)
                    inputdata = np.concatenate([lobulationpred, spiculationpred, subtletypred, fullypred, fprpred], 1)

                    feed_dict = {inputplace: inputdata, real_label: batch_label}
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
                fprpred, fpracc = tempfpr.pred(batch_data, batch_label)
                inputdata = np.concatenate([lobulationpred, spiculationpred, subtletypred, fullypred, fprpred], 1)

                test_dict = {inputplace: inputdata, real_label: batch_label}
                # print(len(out_subtlety))
                # print(out_subtlety[0].shape)


                countx = 0
                for one in test_filenames:
                    if 'high' in one:
                        countx += 1
                print('precent: ', countx / len(test_filenames))
                acc_test,loss,aucscore = sess.run([accruacy,net_loss, auc],feed_dict=test_dict)
                print('acc is %f, loss is %f , auc is %f '%(acc_test, loss, aucscore))
                f.write('acc is %f, loss is %f , auc is %f \n'%(acc_test, loss, aucscore))

                fakefiles = []
                lowfiles = []
                highfiles = []

                for onefile in test_filenames:
                    if 'fake' in onefile:
                        fakefiles.append(onefile)
                    elif 'low' in onefile:
                        lowfiles.append(onefile)
                    elif 'high' in onefile:
                        highfiles.append(onefile)

                print('xxxx ',len(fakefiles))
                if len(fakefiles) != 0:
                    batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(fakefiles)
                    print(batch_data)
                    print(batch_label)
                    lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    fullypred, fullyacc = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    fprpred, fpracc = tempfpr.pred(batch_data, batch_label)
                    inputdata = np.concatenate([lobulationpred, spiculationpred, subtletypred, fullypred, fprpred], 1)

                    test_dict = {inputplace: inputdata, real_label: batch_label}
                    # print(len(out_subtlety))
                    # print(out_subtlety[0].shape)
                
                if len(lowfiles) != 0:

                    acc_test,loss,aucscore = sess.run([accruacy,net_loss, auc],feed_dict=test_dict)
                    print('fake acc is %f, loss is %f , auc is %f '%(acc_test, loss, aucscore))
                    f.write('fake acc is %f, loss is %f , auc is %f \n'%(acc_test, loss, aucscore))

                    batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(lowfiles)

                    lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    fullypred, fullyacc = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    fprpred, fpracc = tempfpr.pred(batch_data, batch_label)
                    inputdata = np.concatenate([lobulationpred, spiculationpred, subtletypred, fullypred, fprpred], 1)

                    test_dict = {inputplace: inputdata, real_label: batch_label}
                    # print(len(out_subtlety))
                    # print(out_subtlety[0].shape)

                    acc_test,loss,aucscore = sess.run([accruacy,net_loss, auc],feed_dict=test_dict)
                    print('low acc is %f, loss is %f , auc is %f '%(acc_test, loss, aucscore))
                    f.write('low acc is %f, loss is %f , auc is %f \n'%(acc_test, loss, aucscore))

                if len(highfiles) != 0:

                    batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(highfiles)

                    lobulationpred, lobulationacc = templobulation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    spiculationpred, spiculationacc = tempspiculation.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    subtletypred, subtletyacc = tempsubtlety.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    fullypred, fullyacc = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)
                    fprpred, fpracc = tempfpr.pred(batch_data, batch_label)
                    inputdata = np.concatenate([lobulationpred, spiculationpred, subtletypred, fullypred, fprpred], 1)

                    test_dict = {inputplace: inputdata, real_label: batch_label}
                    # print(len(out_subtlety))
                    # print(out_subtlety[0].shape)

                    acc_test,loss,aucscore = sess.run([accruacy,net_loss, auc],feed_dict=test_dict)
                    print('high acc is %f, loss is %f , auc is %f '%(acc_test, loss, aucscore))
                    f.write('high acc is %f, loss is %f , auc is %f \n'%(acc_test, loss, aucscore))

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


class model(object):

    def __init__(self,learning_rate,keep_prob,batch_size,epoch):
        print(" network begin...")
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.epoch = epoch

        self.cubic_shape = [[40, 40], [6, 20, 20], [26, 40, 40]]
    
    def archi_1(self,input,sphericity, margin, lobulation, spiculation, keep_prob):
        with tf.name_scope("Archi-1"):
            print(input)
            w_conv1 = tf.Variable(tf.random_normal([5, 5, 3, 64],stddev=0.01),dtype=tf.float32,name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(input,w_conv1,strides=[1,1,1,1],padding='VALID'),b_conv1))
            
            pooling1 = tf.nn.max_pool(out_conv1, ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'VALID')

            w_conv2 = tf.Variable(tf.random_normal([5, 5, 64, 128],stddev=0.01),dtype=tf.float32,name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01,shape=[128]),dtype=tf.float32,name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(pooling1, w_conv2, strides=[1,1,1,1],padding='VALID'), b_conv2))

            pooling2 = tf.nn.max_pool(out_conv2, ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'VALID')

            w_conv3 = tf.Variable(tf.random_normal([1, 1, 128, 256],stddev=0.01),dtype=tf.float32,name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01,shape=[256]),dtype=tf.float32,name='b_conv3')
            out_conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(pooling2, w_conv3, strides=[1,1,1,1], padding='VALID'),b_conv3))
            
            pooling3 = tf.reshape(out_conv3 ,[-1, 256 * 7 * 7])
            w_fc1 = tf.Variable(tf.random_normal([256 * 7 * 7, 1024],stddev=0.01),name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(pooling3, w_fc1), tf.constant(0.01,shape=[1024])))
            out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

            w_fc2 = tf.Variable(tf.random_normal([1024, 2], stddev=0.01), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.01, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)

            out_fc1_shape = tf.shape(out_fc2)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])
            
            print(out_fc2)

            return out_fc2

    def inference(self,npy_path,model_index,test_size,seed,train_flag=True):
        f = open('1245_001_final.txt', 'w')
        highest_acc = 0.0
        highest_iterator = 1

        train_filenames,test_filenames = get_train_and_test_filename(npy_path,test_size,seed)

        # how many time should one epoch should loop to feed all data
        times = len(train_filenames) // self.batch_size
        if (len(train_filenames) % self.batch_size) != 0:
            times = times + 1

        print("Training examples: ", len(train_filenames))
        print("Testing examples: ", len(test_filenames))
        keep_prob = tf.placeholder(tf.float32)
        x = tf.placeholder(tf.float32, [None, 40, 40, 3])

        sphericity = tf.placeholder(tf.float32)
        margin = tf.placeholder(tf.float32)
        lobulation = tf.placeholder(tf.float32)
        spiculation = tf.placeholder(tf.float32)
        
        x_image = tf.reshape(x, [-1, 40, 40, 3])
        net_out = self.archi_1(x_image, sphericity, margin, lobulation, spiculation, keep_prob)

        saver = tf.train.Saver()  # default to save all variable,save mode or restore from path

        if train_flag:
            global_step = tf.Variable(0)  
            learning_rate = tf.train.exponential_decay(0.01, global_step, times * 30, 1, staircase=True)
            # softmax layer
            real_label = tf.placeholder(tf.float32, [None, 2])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label)
            #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
            net_loss = tf.reduce_mean(cross_entropy)

            train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(net_loss)

            prediction = tf.nn.sigmoid(net_out)
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
            accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            merged = tf.summary.merge_all()
            # times = 5
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                train_writer = tf.summary.FileWriter('./fully_tensorboard/', sess.graph)
                for i in range(self.epoch):
                    epoch_start =time.time()
                    random.shuffle(train_filenames)
                    for t in range(times):
                        if t % 10 == 0:
                            print(t, times)
                        batch_files = train_filenames[t*self.batch_size:(t+1)*self.batch_size]
                        batch_data, sphericityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(batch_files)

                        feed_dict = {x: batch_data, sphericity: sphericityt, margin: margint, lobulation: lobulationt, 
                            spiculation: spiculationt, real_label: batch_label, keep_prob: self.keep_prob}
                        _,summary,out_res = sess.run([train_step, merged, net_out],feed_dict =feed_dict)

                        lnrt = sess.run(learning_rate, feed_dict = {global_step : t + i * times})
                        if t == times - 1:
                            print(lnrt)
                        train_writer.add_summary(summary, i)
                        saver.save(sess, './fully_ckpt/fully', global_step=i + 1)
                    epoch_end = time.time()
                    test_batch, sphericityt, margint, lobulationt, spiculationt, test_label = get_batch_withlabels(test_filenames)

                    x10 = 0
                    x01 = 0
                    for label in test_label:
                        if label[0] == 1:
                            x10 += 1
                        else:
                            x01 += 1
                    print('percent: ', x10 / len(test_label))
                    test_dict = {x: test_batch,sphericity: sphericityt, margin: margint, lobulation: lobulationt, spiculation: spiculationt, real_label: test_label, keep_prob:self.keep_prob}
                    acc_test,loss = sess.run([accruacy,net_loss],feed_dict=test_dict)

                    print('accuracy  is %f ' % acc_test)
                    f.write('accuracy  is %f ' % acc_test)

                    print('loss is ', loss)
                    f.write('loss is %f ' % loss)
                    print("epoch %d time consumed %f seconds "%(i,(epoch_end-epoch_start)))
                    f.write("epoch %d time consumed %f seconds "%(i,(epoch_end-epoch_start)))
                    if acc_test  > highest_acc:
                        highest_acc = acc_test
                        highest_iterator = i
                    f.write('\n')
            print("training finshed..highest accuracy is %f,the iterator is %d " % (highest_acc, highest_iterator))
            f.write("training finshed..highest accuracy is %f,the iterator is %d "  % (highest_acc, highest_iterator))

            
        else:
            print("restore model")
             # softmax layer
            real_label = tf.placeholder(tf.float32, [None, 2])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label)
            #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
            net_loss = tf.reduce_mean(cross_entropy)

            train_step = tf.train.MomentumOptimizer(self.learning_rate, 1).minimize(net_loss)

            prediction = tf.nn.softmax(net_out)
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
            accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            merged = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                train_writer = tf.summary.FileWriter('./tensorboard/', sess.graph)                                
                saver = tf.train.import_meta_graph('./fully_ckpt/fully-120.meta')
                saver.restore(sess, tf.train.latest_checkpoint('./fully_ckpt/'))
                # test_filenames = get_high_data(npy_path)
                test_filenamesX = []
                for onefile in test_filenames:
                    if 'low' in onefile:
                       test_filenamesX.append(onefile)

                print('test ', len(test_filenamesX))
                # loop epoches
                for i in range(10):
                    epoch_start =time.time()
                    #  the data will be shuffled by every epoch
                    for t in range(times):
                        print(t)
                        batch_files = test_filenamesX[t*self.batch_size:(t+1)*self.batch_size]
                        test_batch, sphericityt, margint, lobulationt, spiculationt, test_label = get_batch_withlabels(test_filenamesX)                   
                        test_dict = {x: test_batch,sphericity: sphericityt, margin: margint, lobulation: lobulationt, spiculation: spiculationt, real_label: test_label, keep_prob:self.keep_prob}

                        acc_test,loss, aucpred = sess.run([accruacy,net_loss, prediction],feed_dict=test_dict)
                        print('accuracy  is %f' % acc_test)

                        if acc_test  > highest_acc:
                            highest_acc = acc_test
                            highest_iterator = i
                    epoch_end = time.time()


            print("training finshed..highest accuracy is %f,the iterator is %d " % (highest_acc, highest_iterator))
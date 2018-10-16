# -*- coding:utf-8 -*-
'''
maligancy

'''
import tensorflow as tf
from dataprepare import get_batch,get_train_and_test_filename, get_batch_withlabels, get_high_data, get_batch_withlabels_high
import random
import time
import datetime
import os
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)


class model(object):

    def __init__(self,learning_rate,keep_prob,batch_size,epoch):
        print(" network begin...")
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.epoch = epoch

        self.cubic_shape = [[10, 20, 20], [6, 20, 20], [26, 40, 40]]
    
    def archi_1(self,input,sphericity, margin, lobulation, spiculation, keep_prob):
        # return out_fc2
        with tf.name_scope("Archi-1"):
            # input size is batch_sizex20x20x6
            # 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3, 5, 5, 1, 64],stddev=0.01),dtype=tf.float32,name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1,keep_prob)

            # max pooling ,pooling layer has no effect on the data size
            hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,1,1,1,1],ksize=[1,1,1,1,1],padding='SAME')

            # after conv1 ,the output size is batch_sizex4x16x16x64([batch_size,in_deep,width,height,output_deep])
            w_conv2 = tf.Variable(tf.random_normal([3, 5, 5, 64, 128], stddev=0.01), dtype=tf.float32,name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[128]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)

            # after conv2 ,the output size is batch_sizex2x12x12x64([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([3, 5, 5, 128, 256], stddev=0.01), dtype=tf.float32, name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[256]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)

            w_conv4 = tf.Variable(tf.random_normal([1, 1, 1, 256, 256], stddev=0.01), dtype=tf.float32, name='w_conv4')
            b_conv4 = tf.Variable(tf.constant(0.01, shape=[256]), dtype=tf.float32, name='b_conv4')
            out_conv4 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv3, w_conv4, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv4))
            out_conv4 = tf.nn.dropout(out_conv4, keep_prob)

            out_conv3_shape = tf.shape(out_conv4)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

            # after conv2 ,the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv4 = tf.reshape(out_conv4,[-1, 256 * 8 * 8 * 4])
            w_fc1 = tf.Variable(tf.random_normal([256 * 8 * 8 * 4, 200],stddev=0.01),name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv4, w_fc1), tf.constant(0.01,shape=[200])))
            out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

            
            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([200, 2], stddev=0.01), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.01, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob, name = 'net_out')
           
            return out_fc2

    def inference(self,npy_path,model_index,test_size,seed,train_flag=True):
        f = open('1245_001_final.txt', 'w')
        # some statistic index
        highest_acc = 0.0
        highest_iterator = 1

        # train_filenames,test_filenames = get_train_and_test_filename(npy_path,test_size,seed)
        # # how many time should one epoch should loop to feed all data
        # times = len(train_filenames) // self.batch_size
        # if (len(train_filenames) % self.batch_size) != 0:
        #     times = times + 1

        # print("Training examples: ", len(train_filenames))
        # print("Testing examples: ", len(test_filenames))
        # keep_prob used for dropout
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        # take placeholder as input
        x = tf.placeholder(tf.float32, [None, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2]], name = 'x')

        # <sphericity>3</sphericity>
        # <margin>3</margin>
        # <lobulation>3</lobulation>
        # <spiculation>4</spiculation>
        sphericity = tf.placeholder(tf.float32, name = 'sphericity')
        margin = tf.placeholder(tf.float32, name = 'margin')
        lobulation = tf.placeholder(tf.float32, name = 'lobulation')
        spiculation = tf.placeholder(tf.float32, name = 'spiculation')

        x_image = tf.reshape(x, [-1, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2], 1], name = 'x_image')
        net_out = self.archi_1(x_image, sphericity, margin, lobulation, spiculation, keep_prob)
        print(net_out)
        saver = tf.train.Saver()  # default to save all variable,save mode or restore from path

        if train_flag:
            global_step = tf.Variable(0, name = 'global_step')  
            learning_rate = tf.train.exponential_decay(0.01, global_step, 5159, 1, staircase=True, name = 'learning_rate')

            # softmax layer
            real_label = tf.placeholder(tf.float32, [None, 2], name = 'real_label')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label, name = 'cross_entropy')
            #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
            net_loss = tf.reduce_mean(cross_entropy)

            train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(net_loss)

            prediction = tf.nn.sigmoid(net_out, name = 'prediction')
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
            accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accruacy')

            _,auc =  tf.metrics.auc(tf.cast(tf.argmax(prediction, 1),tf.float32),tf.cast(tf.argmax(real_label, 1),tf.float32))

            merged = tf.summary.merge_all()
            # times = 5
            acc_list = []
            auc_list = []
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                train_writer = tf.summary.FileWriter('./fully_tensorboard/', sess.graph)

                acc_list_epochs = []
                auc_list_epochs = []
                K = 5
                for m in range(K):
                    train_filenames,test_filenames = get_train_and_test_filename(npy_path,K,m)
                # how many time should one epoch should loop to feed all data
                    times = len(train_filenames) // self.batch_size
                    if (len(train_filenames) % self.batch_size) != 0:
                        times = times + 1
                    # loop epoches
                    for i in range(self.epoch):
                        epoch_start =time.time()
                        #  the data will be shuffled by every epoch
                        random.shuffle(train_filenames)
                        # times = 5
                        for t in range(times):
                            if t % 10 == 0:
                                print(t, times)
                            # print(t, times)
                            batch_files = train_filenames[t*self.batch_size:(t+1)*self.batch_size]
                            batch_data, sphericityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(batch_files)
                            feed_dict = {x: batch_data, sphericity: sphericityt, margin: margint, lobulation: lobulationt, 
                                spiculation: spiculationt, real_label: batch_label, keep_prob: self.keep_prob}
                            _,summary,out_res = sess.run([train_step, merged, net_out],feed_dict =feed_dict)
                            # print(len(out_res))
                            # print(out_res[0])
                            # print(sess.run(tf.nn.sigmoid(out_res[0])))
                            # feed_dict = {global_step: t + i * times}
                            lnrt = sess.run(learning_rate, feed_dict = {global_step : t + i * times})
                            if t == times - 1:
                                print(lnrt)

                            train_writer.add_summary(summary, i)
                            saver.save(sess, './fully_ckpt/fully', global_step=i + 1)

                        epoch_end = time.time()
                        # randomtestfiles = random.sample(test_filenames, 32)
                        test_batch, sphericityt, margint, lobulationt, spiculationt, test_label = get_batch_withlabels(test_filenames)
                        # print(test_label)

                        x10 = 0
                        x01 = 0
                        for label in test_label:
                            if label[0] == 1:
                                x10 += 1
                            else:
                                x01 += 1
                        print('percent: ', x10 / len(test_label))
                        # f.write('percent: %f ' % x10 / 16)
                        test_dict = {x: test_batch,sphericity: sphericityt, margin: margint, lobulation: lobulationt, spiculation: spiculationt, real_label: test_label, keep_prob:self.keep_prob}
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
                # print("training finshed..highest accuracy is %f,the iterator is %d " % (highest_acc, highest_iterator))
                # f.write("training finshed..highest accuracy is %f,the iterator is %d "  % (highest_acc, highest_iterator))

             
        else:
            print("restore model")
             # softmax layer
            real_label = tf.placeholder(tf.float32, [None, 2])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label)
            net_loss = tf.reduce_mean(cross_entropy)

            train_step = tf.train.MomentumOptimizer(self.learning_rate, 1).minimize(net_loss)

            prediction = tf.nn.softmax(net_out)
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(real_label, 1))
            accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            merged = tf.summary.merge_all()

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                train_writer = tf.summary.FileWriter('./tensorboard/', sess.graph)                                
                saver = tf.train.import_meta_graph('./fully_ckpt/fully-80.meta')
                saver.restore(sess, tf.train.latest_checkpoint('./fully_ckpt/'))
                test_filenames = os.listdir(npy_path)

                test_filenamesX = []
                for onefile in test_filenames:
                    if 'low' in onefile:
                       test_filenamesX.append(onefile)
                    if 'high' in onefile:
                        test_filenamesX.append(onefile)

                print('test ', len(test_filenamesX))
                # loop epoches
                for i in range(2):
                    epoch_start =time.time()
                    #  the data will be shuffled by every epoch
                    for t in range(10):
                        print(t)
                        batch_files = test_filenamesX[t*self.batch_size:(t+1)*self.batch_size]
                        print(len(batch_files))
                        test_batch, sphericityt, margint, lobulationt, spiculationt, test_label = get_batch_withlabels(test_filenamesX)                   
                        test_dict = {x: test_batch,sphericity: sphericityt, margin: margint, lobulation: lobulationt, spiculation: spiculationt, real_label: test_label, keep_prob:self.keep_prob}

                        acc_test,loss, aucpred = sess.run([accruacy,net_loss, prediction],feed_dict=test_dict)
                        print('accuracy  is %f' % acc_test)

                        if acc_test  > highest_acc:
                            highest_acc = acc_test
                            highest_iterator = i
                    epoch_end = time.time()


            print("training finshed..highest accuracy is %f,the iterator is %d " % (highest_acc, highest_iterator))


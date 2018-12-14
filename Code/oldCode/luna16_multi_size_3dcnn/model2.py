# -*- coding:utf-8 -*-
'''
 the idea of this script came from LUNA2016 champion paper.
 This model conmposed of three network,namely Archi-1(size of 10x10x6),Archi-2(size of 30x30x10),Archi-3(size of 40x40x26)

'''
import tensorflow as tf
from data_prepare import get_train_batch,get_all_filename,get_test_batch
import random
import time
#import tensorflow.python.debug as tf_debug
class model2(object):

    def __init__(self,learning_rate,keep_prob,batch_size,epoch):
        print(" network begin...")
        print(learning_rate)
        print(keep_prob)
        print(batch_size)
        print(epoch)
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.epoch = epoch

        self.cubic_shape = [[6, 20, 20], [10, 30, 30], [26, 40, 40]]

    def archi_1(self,input,keep_prob):
        with tf.name_scope("Archi-1"):
            # input size is batch_sizex20x20x6
            # 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3,5,5,1,64],stddev=0.001),dtype=tf.float32,name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1,keep_prob)

            # max pooling ,pooling layer has no effect on the data size
            hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,1,1,1,1],ksize=[1,1,1,1,1],padding='SAME')

            # after conv1 ,the output size is batch_sizex4x16x16x64([batch_size,in_deep,width,height,output_deep])
            w_conv2 = tf.Variable(tf.random_normal([3, 5, 5, 64,64], stddev=0.001), dtype=tf.float32,name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)

            # after conv2 ,the output size is batch_sizex2x12x12x64([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([1, 5, 5, 64,64], stddev=0.001), dtype=tf.float32,
                                  name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(
                tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)

            out_conv3_shape = tf.shape(out_conv3)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

            # after conv2 ,the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv3 = tf.reshape(out_conv3,[-1,64*8*8*2])
            w_fc1 = tf.Variable(tf.random_normal([64*8*8*2,150],stddev=0.001),name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3,w_fc1),tf.constant(0.001,shape=[150])))
            out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([150, 2], stddev=0.001), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.001, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)

            w_sm = tf.Variable(tf.random_normal([2, 2], stddev=0.001), name='w_sm')
            b_sm = tf.constant(0.001, shape=[2])
            out_sm = tf.nn.softmax(tf.add(tf.matmul(out_fc2, w_sm), b_sm))

            return out_sm

    def archi_2(self,input,keep_prob=0.7):
        with tf.name_scope("Archi-2"):
            # input size is batch_sizex20x20x6
            # 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3,5,5,1,64],stddev=0.001),dtype=tf.float32,name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1,keep_prob)

            # max pooling ,pooling layer has no effect on the data size
            hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,1,1,1,1],ksize=[1,1,2,2,1],padding='SAME')

            # after conv1 ,the output size is batch_sizex4x16x16x64([batch_size,in_deep,width,height,output_deep])
            w_conv2 = tf.Variable(tf.random_normal([3 ,5, 5, 64,64], stddev=0.001), dtype=tf.float32,name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)

            # after conv2 ,the output size is batch_size x 2 x 12 x 12 x 64 ([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([3 ,5, 5, 64,64], stddev=0.001), dtype=tf.float32,
                                  name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(
                tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)

            out_conv3_shape = tf.shape(out_conv3)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

            # after conv2 ,the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv3 = tf.reshape(out_conv3,[-1,64*18*18*4])
            w_fc1 = tf.Variable(tf.random_normal([64*18*18*4,250],stddev=0.001),name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3,w_fc1),tf.constant(0.001,shape=[250])))
            out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([250, 2], stddev=0.001), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.001, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)

            w_sm = tf.Variable(tf.random_normal([2, 2], stddev=0.001), name='w_sm')
            b_sm = tf.constant(0.001, shape=[2])
            out_sm = tf.nn.softmax(tf.add(tf.matmul(out_fc2, w_sm), b_sm))
            return out_sm

    def archi_3(self,input,keep_prob):
        with tf.name_scope("Archi-3"):
            # input size is batch_sizex[26, 40, 40]
            # 5x5x3 is the kernel size of conv1,1 is the input depth,64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3,5,5,1,64],stddev=0.001),dtype=tf.float32,name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1,keep_prob)

            # max pooling ,pooling layer has no effect on the data size
            hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,1,1,1,1],ksize=[1,2,2,2,1],padding='SAME')

            # after conv1 ,the output size is batch_size x 24 x 36 x 36 x 64([batch_size,in_deep,width,height,output_deep])
            w_conv2 = tf.Variable(tf.random_normal([3 ,5, 5, 64,64], stddev=0.001), dtype=tf.float32,name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)

            # after conv2 ,the output size is batch_size x 22 x 32 x 32 x 64 ([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([3 ,5, 5, 64,64], stddev=0.001), dtype=tf.float32,
                                  name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(
                tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)

            out_conv3_shape = tf.shape(out_conv3)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

            # after conv2 ,the output size is batch_sizex20x28x28x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv3 = tf.reshape(out_conv3,[-1,64*28*28*20])
            w_fc1 = tf.Variable(tf.random_normal([64*28*28*20,250],stddev=0.001),name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3,w_fc1),tf.constant(0.001,shape=[250])))
            out_fc1 = tf.nn.dropout(out_fc1,keep_prob)

            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([250, 2], stddev=0.001), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.001, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)

            w_sm = tf.Variable(tf.random_normal([2, 2], stddev=0.001), name='w_sm')
            b_sm = tf.constant(0.001, shape=[2])
            out_sm = tf.nn.softmax(tf.add(tf.matmul(out_fc2, w_sm), b_sm))
            return out_sm

    def inference(self,npy_path,test_path,model_index,train_flag=True):
        # some statistic index
        highest_acc = 0.0
        highest_iterator = 1

        all_filenames = get_all_filename(npy_path,self.cubic_shape[model_index][1])
        all_test_filenames = get_all_filename(test_path,self.cubic_shape[model_index][1])
        random.shuffle(all_test_filenames)
        random.shuffle(all_filenames)

        # how many time should one epoch should loop to feed all data
        times = int(len(all_filenames) / self.batch_size)
        if (len(all_filenames) % self.batch_size) != 0:
            times = times + 1

        print("training times: " + str(times))
        # keep_prob used for dropout
        keep_prob = tf.placeholder(tf.float32)
        # take placeholder as input
      

        # print("shape of x :" )
        # print(x.shape)
        # print("shape of x_image :")
        # print(x_image.shape)
        # print("shape of net_out :")
        # print(net_out.shape)
        '''
        shape of x :
        (?, 6, 20, 20)
        shape of x_image :
        (?, 6, 20, 20, 1)
        shape of net_out :
        (?, 2)
        '''

        # saver = tf.train.Saver()  # default to save all variable,save mode or restore from path
        print("train:")
        if train_flag:
            # softmax layer
            # logits_size=[486,2] labels_size=[32,2]
            x = tf.placeholder(tf.float32, [None, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2]])
            x_image = tf.reshape(x, [-1, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2], 1])
            net_out = self.archi_2(x_image)
            real_label = tf.placeholder(tf.float32, [None, 2])

            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label))

            # cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label))
            # #cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
            optimizer = tf.train.AdamOptimizer().minimize(cost)

            # net_loss = tf.reduce_mean(cross_entropy)

            # train_step = tf.train.AdamOptimizer(self.learning_rate, 0.9).minimize(net_loss)

            # correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(real_label, 1))
            # accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            merged = tf.summary.merge_all()

            
            with tf.Session() as sess:
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                sess.run(tf.global_variables_initializer())
                train_writer = tf.summary.FileWriter('./tensorboard/', sess.graph)
                # loop epoches
                for i in range(self.epoch):
                    epoch_loss = 0

                    print("Training " + str(i))
                    epoch_start =time.time()
                    #  the data will be shuffled by every epoch
                    for t in range(times):
                        print("times " + str(t))
                        batch_files = all_filenames[t*self.batch_size:(t+1)*self.batch_size]
                        batch_data, batch_label = get_train_batch(batch_files)

                        # here is the problem
                        feed_dict = {x: batch_data, real_label: batch_label}
                        _, c = sess.run([optimizer, cost],feed_dict =feed_dict)
                        epoch_loss += c
                        # saver.save(sess, './ckpt/archi-2', global_step=i + 1)
                    print( 'loss: ', epoch_loss)
                    test_batch,test_label = get_test_batch(all_test_filenames)
                    feed_dict = {x: test_batch, real_label: test_label}

                    correct = tf.equal(tf.argmax(net_out, 1), tf.argmax(test_label, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    print('Accrracy: ', accuracy.eval({x: test_batch, real_label: test_label}))
                    # epoch_end = time.time()
                    # test_batch,test_label = get_test_batch(all_test_filenames)

                    # test_dict = {x: test_batch, real_label: test_label}
                    # acc_test,loss = sess.run([accruacy,net_loss],feed_dict=test_dict)

                    # print('accuracy is %f' % acc_test)
                    # print("epoch %d time consumed %f seconds"%(i,(epoch_end-epoch_start)))

            print("training finshed..highest accuracy is %f,the iterator is %d " % (highest_acc, highest_iterator))





























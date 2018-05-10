'''
Created by WangQL

py file for 3D conv

Medic
'''

import tensorflow as tf
import numpy as np

x1 = tf.placeholder('float')
x2 = tf.placeholder('float')
x3 = tf.placeholder('float')

y = tf.placeholder('float')
    
# return tf.nn.max_pool3d(x, ksize = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1], padding = 'SAME')

# archi-1
x1 = tf.reshape(x, shape = [-1, 20, 20, 6, 1])
archi1_C1_weight1 = tf.Variable(tf.random_normal([5, 5, 3, 1, 64]))
archi1_C1_bias1 = tf.Variable(tf.constant(0.1, shape = [64]))
archi1_C1_kernal1 = tf.nn.conv3d(x, archi1_C1_weight, strides = [1,1,1,1,1], padding = 'SAME')
# bias2 = tf.Variable(tf.constant(0.1, shape = [64]))
# conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
archi1_C1_conv1 = tf.nn.relu(tf.nn.bias_add(archi1_C1_kernal1, archi1_C1_bias1))
archi1_C1_maxpooling = tf.nn.max_pool3d(archi1_C1_conv1, ksize = [1, 1, 1, 1, 1], 
    strides = [1, 1, 1, 1, 1], padding = 'SAME')

archi1_C1_weight2 = tf.Variable(tf.random_normal([5, 5, 3, 64, 64]))
archi1_C1_bias2 = tf.Variable(tf.constant(0.1, shape = [64]))
archi1_C1_kernal2 = tf.nn.conv3d(archi1_C1_maxpooling, archi1_C1_weight2, strides = [1, 1, 1, 1, 1], padding = 'SAME')
archi1_C1_conv2 = tf.nn.relu(tf.nn.bias_add(archi1_C1_kernal2, archi1_C1_bias2))

archi1_C1_weight3 = tf.Variable(tf.random_normal([5, 5, 1, 64, 64]))
archi1_C1_bias3 = tf.Variable(tf.constant(0.1, shape = [64]))
archi1_C1_kernal3 = tf.nn.conv3d(archi1_C1_conv2, archi1_C1_weight3, strides = [1, 1, 1, 1, 1], padding = 'SAME')
archi1_C1_conv3 = tf.nn.relu(tf.nn.bias_add(archi1_C1_kernal3, archi1_C1_bias3))

# 20 20 6 64 = 153500
# 'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
# 'out': tf.Variable(tf.random_normal([1024, n_classes]))
archi1_C1_FC1 = tf.reshape(archi1_C1_conv3, [-1, 153500])
archi1_C1_FC1_weight = tf.Variable(tf.random_normal([153500, 150))
archi1_C1_FC1_bias = tf.Variable(tf.constant(0.1, shape = [150]))
archi1_C1_FC1_out = tf.nn.relu(tf.matmul(archi1_C1_FC1, archi1_C1_FC1_weight) + archi1_C1_FC1_bias)

archi1_C1_FC2_weight = tf.Variable(tf.random_normal([150, 2]))
archi1_C1_FC2_bias = tf.Variable(tf.constant(0.1, shape = [2]))
archi1_C1_FC2_out = tf.nn.relu(tf.matmul(archi1_C1_out, archi1_C1_FC2_weight) + archi1_C1_FC2_bias)

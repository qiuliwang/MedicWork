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

# archi-1
x1 = tf.reshape(x, shape = [-1, 20, 20, 6, 1])

archi1_C1_conv = tf.nn.conv3d(x, W, strides = [1,5,5,3,1], padding = 'SAME')
archi1_C1_weight = tf.Variable(tf.random_normal([3, 3, 3, 1, 32]))
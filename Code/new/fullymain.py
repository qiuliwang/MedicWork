# -*- coding:utf-8 -*-
'''
this is the enterance of this project
'''

import tensorflow as tf
import os
from fully import model
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ =='__main__':
    batch_size = 128
    learning_rate = 0.01
    keep_prob = 1
    path = 'train/'
    #path = 'fpr_npy/'

    #test_path = '../../data/cubic_normalization_test'
    test_size = 0.1
    seed=121

    print(" begin...")
    model = model(learning_rate,keep_prob,batch_size,40)
    model.inference(path,0, test_size, seed, True)

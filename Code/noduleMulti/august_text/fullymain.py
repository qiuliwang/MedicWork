# -*- coding:utf-8 -*-
'''
this is the enterance of this project
'''

import tensorflow as tf
import os
from fully import model
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ =='__main__':
    batch_size =64
    learning_rate = 0.01
    keep_prob = 0.7
    # path = '/data0/LUNA/cubic_normalization_npy'
    path = 'train/'

    #test_path = '../../data/cubic_normalization_test'
    test_size = 0.1
    seed=121

    print(" begin...")
    model = model(learning_rate,keep_prob,batch_size, 80)
    model.inference(path,0, test_size, seed, True)

# -*- coding:utf-8 -*-
'''
this is the enterance of this project
'''

# import tensorflow as tf
import os
# from fusionmodel import fusion
import numpy as np
import random

from fullymodel import modelfully

from dataprepare import get_batch,get_train_and_test_filenames, get_batch_withlabels
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    batch_size = 32
    learning_rate = 0.1
    keep_prob = 1
    epoch = 40
    # path = '/data0/LUNA/cubic_normalization_npy'
    path = 'train/'

    train_filenames,test_filenames = get_train_and_test_filenames(path, 0.1, 121)
    print('traing: ', len(train_filenames))
    print('test: ', len(test_filenames))
    times = len(train_filenames) // batch_size
    if (len(train_filenames) % batch_size) != 0:
        times = times + 1

    tempfully = modelfully(learning_rate, 1, batch_size, epoch)

    for i in range(2):
        random.shuffle(train_filenames)
        for t in range(2):
            batch_files = test_filenames
            batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label = get_batch_withlabels(batch_files)
            out_fully = tempfully.pred( batch_data, sphercityt, margint, lobulationt, spiculationt, batch_label)

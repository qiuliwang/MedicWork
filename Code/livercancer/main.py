from dataprepare import Data
# from model import CaptionGenerator
from config import Config
import tensorflow as tf
import os
import numpy as np 
from tqdm import tqdm
from newmodel import RCNNMODEL


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


datapath1 = '/home/wangqiuli/Data/liver_cancer_dataset/train_dataset/'
datapath2 = '/home/wangqiuli/Data/liver_cancer_dataset/train_data2/'

labelpath1 = './train_label.csv'
labelpath2 = './train2_label.csv'
data = Data(datapath1, labelpath1)
patients = data.patients
labels = data.labels

data2 = Data(datapath2, labelpath2)
patients2 = data2.patients
labels2 = data2.labels

print(len(patients))
print(len(labels))
print(len(patients2))
print(len(labels2))

# for onepatient in patients2:
#     try:
#         for onelabel in labels2:
#             if onepatient == onelabel[0]:
#                 break
#     except:
#         print onepatient

config = Config()

load = False
load_cnn = False


with tf.Session() as sess:
    model = RCNNMODEL(config)

    sess.run(tf.global_variables_initializer())   
    if load:
        model.load(sess, './models/18146.npy')
    if load_cnn:
        model.load_cnn(sess, './vgg16_no_fc.npy')
    tf.get_default_graph().finalize()
    model.train(sess, patients, labels, data)
    model.train(sess, patients2, labels2, data2)

    testdatapath = '/home/wangqiuli/Data/liver_cancer_dataset/test_data/'
    model.test(sess, testdatapath, data)

from dataprepare import Data
# from model import CaptionGenerator
from config import Config
import tensorflow as tf
import os
import numpy as np 
from tqdm import tqdm
from newmodel import RCNNMODEL


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpuconfig = tf.ConfigProto()

gpuconfig.gpu_options.per_process_gpu_memory_fraction = 0.99

datapath = '/home/wangqiuli/Data/liver_cancer_dataset/train_dataset/'

labelpath = './train_label.csv'
data = Data(datapath, labelpath)
patients = data.patients
labels = data.labels

print(len(patients))
print(len(labels))

# for onepatient in patients2:
#     try:
#         for onelabel in labels2:
#             if onepatient == onelabel[0]:
#                 break
#     except:
#         print onepatient

config = Config()

load = True
load_cnn =False


with tf.Session() as sess:
    model = RCNNMODEL(config)

    sess.run(tf.global_variables_initializer())   
    if load:
        model.load(sess, './models/52801.npy')
    if load_cnn:
        model.load_cnn(sess, './resnet50_no_fc.npy')
    tf.get_default_graph().finalize()
    #model.train(sess, patients, labels, data, False)
    model.train(sess, patients, labels, data, True)
    model.train(sess, patients, labels, data, True)
    model.train(sess, patients, labels, data, True)

    testdatapath = '/home/wangqiuli/Data/liver_cancer_dataset/test_data/'
    model.test(sess, testdatapath, data)

from dataprepare import Data
# from model import CaptionGenerator
from config import Config
import tensorflow as tf
import os
import numpy as np 
from tqdm import tqdm
from newmodel import RCNNMODEL
import pydicom
from PIL import Image
datapath1 = '/home/wangqiuli/Data/liver_cancer_dataset/train_dataset/'

labelpath1 = './train_label.csv'
data = Data(datapath1, labelpath1)
patients = data.patients
labels = data.labels

print(len(patients))
print(len(labels))


# print(len(filelist))
# print(filelist[1])
# onefile = filelist[1]

# import pydicom
# ds = pydicom.dcmread('./data/' + onefile)
# direct = ds.dir()
# print ds.data_element('InstanceNumber').value
# for onedir in direct:
#     # print(onedir)
#     # print(ds.data_element(onedir).value)
#     if ds.data_element(InstanceNumber).value == 96:
#         print onedir
# for onefile in filelist:
#     print onefile

# filelist.sort()
# patient = patients[0]
# npy = data.getOnePatient(patient, 0)
# print(npy.shape)

# t = 0
# for i in range(32):
#     if i + 2 < 30:
#         print(i, i + 1, i + 2)

for onepat in patients[:1]:
        pix = data.getOnePatient(onepat)
        print(pix.shape)
        pix = data.getOnePatient(onepat)
        print(pix.shape)

        pix = data.getOnePatient(onepat, False)
        print(pix.shape)
        pix = data.getOnePatient(onepat,False)
        print(pix.shape)
# from random import choice

# indexlist = [0,1,2,3]
# print(choice(indexlist))
# print(choice(indexlist))
# print(choice(indexlist))
# print(choice(indexlist))
# print(choice(indexlist))
# print(choice(indexlist))
# print(choice(indexlist))
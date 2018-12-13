'''
Created by Wang Qiu Li
7/3/2018

get dicom info according to malignancy.csv and ld_scan.txt
'''

import csvTools
import os
import pandas as pd
import pydicom
import scipy.misc
import cv2
import numpy as np
import random
basedir = '/data0/LIDC/DOI/'
resdir = '1245/'

noduleinfo = csvTools.readCSV('files/malignancy.csv')
idscaninfo = csvTools.readCSV('files/id_scan.txt')

print('normal')

lenoftrain = round(len(noduleinfo) * 0.7)
lenoftest = len(noduleinfo) - lenoftrain

print(lenoftrain)
print(lenoftest)
print(len(noduleinfo))

traindata = random.sample(noduleinfo , lenoftrain)
testdata = [i for i in noduleinfo if i not in traindata]

print(len(traindata))
print(len(testdata))
csvTools.writeCSV('traindata.csv', traindata)
csvTools.writeCSV('testdata.csv', testdata)

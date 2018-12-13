'''
Created by Wang Qiu Li

7/4/2018

prepare data for malignancy model
'''

import os
import csvTools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import numpy as np
import re

basedir = 'train/'

# 10477 - 10456

datas = os.listdir(basedir)

labels = csvTools.readCSV('files/malignancy.csv')
'''
id = 0
malignancy level = 29
'''

def get_train_and_test_filenames(path,test_size,seed):
    filelist = os.listdir(path)
    return train_test_split(filelist, test_size=test_size, random_state=seed)

def get_train_and_test_index(path,K):
    high_filenames,low_filenames = get_high_and_low_filename(path)
    print("high_filenames: ", len(high_filenames))
    print("low_filenames: ", len(low_filenames))
    kf = KFold(n_splits=K)
    train_high_index = [];test_high_index = []
    train_low_index = [];test_low_index = []
    # random.shuffle(high_filenames)
    # random.shuffle(low_filenames)
    for train, test in kf.split(high_filenames):
        train_high_index.append(train)
        test_high_index.append(test)
    for train, test in kf.split(low_filenames):
        train_low_index.append(train)
        test_low_index.append(test)
    return high_filenames,low_filenames,train_high_index,test_high_index,train_low_index,test_low_index

def get_train_and_test_filename(path,K,i):
    high_filenames,low_filenames,train_high_index,test_high_index,train_low_index,test_low_index = get_train_and_test_index(path,K)
    train_filenames = [];test_filenames = []
    print ("========================  "+ str(i)+ "  ========================")
    print ("train_high_index:",train_high_index[i],"test_high_index",test_high_index[i])
    print ("train_low_index:",train_low_index[i],"test_low_index",test_low_index[i])
    for m in train_high_index[i]:
        train_filenames.append(high_filenames[m])
    for n in train_low_index[i]:
        train_filenames.append(low_filenames[n])
    print("train_filenames",len(train_filenames))
    for m in test_high_index[i]:
        test_filenames.append(high_filenames[m])
    for n in test_low_index[i]:
        test_filenames.append(low_filenames[n])
    print("test_filenames",len(test_filenames))

    return train_filenames,test_filenames

def get_high_and_low_filename(path):
    filelist = os.listdir(path)
    highlist = []
    lowlist = []
    for onefile in filelist:
        if 'high' in onefile:
            highlist.append(onefile)
        if 'low' in onefile:
            lowlist.append(onefile)
    return highlist,lowlist

def get_high_data(path):
    filelist = os.listdir(path)
    returnlist = []
    for onefile in filelist:
        if 'low' in onefile:
            returnlist.append(onefile)
    return returnlist

def get_batch_withlabels_high(batch_filename):
    '''
    get batch
    return data and label
    '''
    batch_array = []
    batch_label = []

    sphercity = []
    margin = []
    lobulation =[]
    spiculation = []

    temp_filename = []

    for one in batch_filename:
        if 'high' in one:
            temp_filename.append(one)

    for onefile in temp_filename:
        try:
            # print(onefile)
            index = onefile[:onefile.find('_')]
            # print(index)


            # print(index)
            chara_list = []
            for onenodule in labels:
                if onenodule[0] == index:
                    sphercity.append([float(onenodule[24]), float(onenodule[24])])
                    margin.append([float(onenodule[25]), float(onenodule[25])])
                    lobulation.append([float(onenodule[26]), float(onenodule[26])])
                    spiculation.append([float(onenodule[27]), float(onenodule[27])])
            arr = np.load(basedir + onefile)
            batch_array.append(arr)
            # id = re.findall("\d+",onefile)[0]
            if 'high' in onefile:
                batch_label.append([0, 1])
            elif 'low' in onefile:
                batch_label.append([1, 0])

        except Exception as e:
            print("file not exists! %s"%onefile)
            batch_array.append(batch_array[-1]) 

    return np.array(batch_array), np.array(sphercity), np.array(margin), np.array(lobulation), np.array(spiculation), np.array(batch_label)


def get_batch_withlabels(batch_filename):
    '''
    get batch
    return data and label
    '''
    batch_array = []
    batch_label = []

    sphercity = []
    margin = []
    lobulation =[]
    spiculation = []

    for onefile in batch_filename:
        try:
            # print(onefile)
            index = onefile[:onefile.find('_')]
            # print(index)


            # print(index)
            chara_list = []
            for onenodule in labels:
                if onenodule[0] == index:
                    sphercity.append([float(onenodule[24]), float(onenodule[24])])
                    margin.append([float(onenodule[25]), float(onenodule[25])])
                    lobulation.append([float(onenodule[26]), float(onenodule[26])])
                    spiculation.append([float(onenodule[27]), float(onenodule[27])])
            arr = np.load(basedir + onefile)
            batch_array.append(arr)
            # id = re.findall("\d+",onefile)[0]
            if 'high' in onefile:
                batch_label.append([0, 1])
            elif 'low' in onefile:
                batch_label.append([1, 0])

        except Exception as e:
            print("file not exists! %s"%onefile)
            batch_array.append(batch_array[-1]) 
    return np.array(batch_array), np.array(sphercity), np.array(margin), np.array(lobulation), np.array(spiculation), np.array(batch_label)



def get_batch(batch_filename):
    '''
    get batch
    return data and label
    '''
    batch_array = []
    batch_label = []

    for onefile in batch_filename:
        try:
            arr = np.load(basedir + onefile)
            batch_array.append(arr)
            # id = re.findall("\d+",onefile)[0]
            if 'high' in onefile:
                batch_label.append([0, 1])
            elif 'low' in onefile:
                batch_label.append([1, 0])

        except Exception as e:
            print("file not exists! %s"%onefile)
            batch_array.append(batch_array[-1]) 
    return np.array(batch_array), np.array(batch_label)

# trainbatch, testbatch = get_train_and_test_filename(basedir, 0.1, 121)
# print(len(trainbatch))
# print(len(testbatch))
# batch_filename = trainbatch[:32]
# print(len(batch_filename))
# batch_array, sphercity, margin, lobulation, spiculation, batch_label = get_batch_withlabels(batch_filename)
# print(len(batch_array))
# print((sphercity))
# print((margin))
# print((lobulation))
# print((spiculation))

# print(len(batch_label))

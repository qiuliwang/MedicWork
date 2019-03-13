import os
import csvTools
import pydicom
import numpy as np
import tensorflow as tf
import math
import scipy.misc
import operator
cmpfun = operator.attrgetter('InstanceNumber')
from PIL import Image
from random import choice
import cv2

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 70] = 0
    image_array[image_array < 30] = 0
    return image_array
    
# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def get_pixels_hu(ds):
    image = ds.pixel_array
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    image = image * slope
    image += intercept
    return image


def angle_transpose(file,degree):
    '''
     @param file : a npy file which store all information of one cubic
     @param degree: how many degree will the image be transposed,90,180,270 are OK
    '''
    array = file

    newarr = np.zeros(array.shape,dtype=np.float32)
    for depth in range(array.shape[0]):
        jpg = array[depth]
        jpg.reshape((jpg.shape[0],jpg.shape[1],1))
        img = Image.fromarray(jpg)
        out = img.rotate(degree)
        newarr[depth,:,:] = np.array(out).reshape(array.shape[1],-1)[:,:]
    return newarr

class Data(object):
    def __init__(self, datapath, labelpath = ''):
        self.datapath = datapath
        self.labelpath  = labelpath
        self.patients = os.listdir(datapath)
        lines = csvTools.readCSV(self.labelpath)
        lines = lines[1:]
        self.labels = lines
        self.slicethicknesscount = []
	self.count = 0

    def getOnePatient(self, patientName, transsign = True):
        dcmfiles = os.listdir(self.datapath + patientName)
        dcmfiles.sort()
        slices = [pydicom.dcmread(os.path.join(self.datapath, patientName, s)) for s in dcmfiles]
        slices.sort(key = cmpfun)
        slicethickness = slices[0].data_element('SliceThickness').value
        dcmkeep = []
        keeprate = 5.0 / slicethickness
        keeprate = int(math.floor(keeprate))
        if keeprate < 1:
            keeprate = 1
        tempsign = 0
        for onedcm in slices:
            if tempsign % keeprate == 0:
                dcmkeep.append(onedcm)
            tempsign += 1
        if len(dcmkeep) > 32:
            x = len(dcmkeep)
            dcmkeep = dcmkeep[(x // 2 - 16) : (x // 2 + 16)]
        if len(dcmkeep) < 32:
            temp = []
            for i in range(0, 32 - len(dcmkeep)):
                temp.append(dcmkeep[0])
            dcmkeep = temp + dcmkeep
        indexlist = [0,0,0,1,2,3]

        if transsign == True:
            index = choice(indexlist)
        else:
            index = 0
        pixels = []
        for temp in dcmkeep:
            temp = get_pixels_hu(temp)
            temp = truncate_hu(temp)
            temp = cv2.resize(temp,(256,256))
            pixels.append(temp)
        threeDSlices = []
        for i in range(30):
            temp = []
            temp.append(pixels[i])
            temp.append(pixels[i + 1])
            temp.append(pixels[i + 2])
            temppix = np.array(temp)
            temppix = temppix.transpose(1,2,0)
            if index == 1:
                temppix = angle_transpose(temppix, 90)
            if index == 2:
                temppix = angle_transpose(temppix, 180)        
            if index == 3:
                temppix = angle_transpose(temppix, 270)       
            threeDSlices.append(temppix)

        pixels = np.array(threeDSlices, dtype=np.float)
        return pixels

    def getOnePatient2(self, testdatapath, patientName):
        dcmfiles = os.listdir(testdatapath + patientName)
        dcmfiles.sort()
        slices = [pydicom.dcmread(os.path.join(testdatapath, patientName, s)) for s in dcmfiles]
        slices.sort(key = cmpfun)

        slicethickness = slices[0].data_element('SliceThickness').value
        dcmkeep = []
        keeprate = 5.0 / slicethickness
        keeprate = int(math.floor(keeprate))
        if keeprate < 1:
            keeprate = 1

        tempsign = 0
        for onedcm in slices:
            if tempsign % keeprate == 0:
                dcmkeep.append(onedcm)
            tempsign += 1
        if len(dcmkeep) > 32:
            x = len(dcmkeep)
            dcmkeep = dcmkeep[(x // 2 - 16) : (x // 2 + 16)]
        if len(dcmkeep) < 32:
            temp = []
            for i in range(0, 32 - len(dcmkeep)):
                temp.append(dcmkeep[0])
            dcmkeep = temp + dcmkeep
        pixels = []

        for temp in dcmkeep:
            temp = get_pixels_hu(temp)
            temp = truncate_hu(temp)
            temp = cv2.resize(temp,(256,256))
            pixels.append(temp)

        threeDSlices = []
        for i in range(30):
            temp = []
            temp.append(pixels[i])
            temp.append(pixels[i + 1])
            temp.append(pixels[i + 2])
            temppix = np.array(temp)
            temppix = temppix.transpose(1,2,0)
            threeDSlices.append(temppix)

        pixels = np.array(threeDSlices, dtype=np.float)      
        return pixels
        

    def getThickness(self):
        return self.slicethicknesscount
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

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 300] = 0
    image_array[image_array < 10] = 0
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
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # Convert to Hounsfield units (HU)
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    # print(intercept)
    # print(slope)
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
        #img.show()
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
        # print(dcmfiles)
        dcmfiles.sort()
        # print(dcmfiles)
        slices = [pydicom.dcmread(os.path.join(self.datapath, patientName, s)) for s in dcmfiles]
        slices.sort(key = cmpfun)
        # for onefile in dcmfiles:
        #     ds = pydicom.dcmread(os.path.join(self.datapath, patientName, onefile))
        #     # pixel = ds.pixel_array
        #     temp = get_pixels_hu(ds)
        #     temp = truncate_hu(temp)
        #     scipy.misc.imsave('./res/' + onefile + 'r.jpeg', temp)
        # pixels = []

        slicethickness = slices[0].data_element('SliceThickness').value
        # print(slicethickness)
        dcmkeep = []
        keeprate = 5.0 / slicethickness
        keeprate = int(math.floor(keeprate))
        # print(keeprate)
        if keeprate < 1:
            keeprate = 1

        # print('rate: ', keeprate)
        tempsign = 0
        for onedcm in slices:
            # print(onedcm)
            if tempsign % keeprate == 0:
                # print(tempsign)
                # print(onedcm)
                dcmkeep.append(onedcm)
            tempsign += 1
        # print(len(dcmkeep))
        if len(dcmkeep) > 30:
            x = len(dcmkeep)
            dcmkeep = dcmkeep[(x // 2 - 15) : (x // 2 + 15)]
        if len(dcmkeep) < 30:
            temp = []
            for i in range(0, 30 - len(dcmkeep)):
                temp.append(dcmkeep[0])
            dcmkeep = temp + dcmkeep
        # sign = 0
        # print('len of dcmkeep',len(dcmkeep))
        indexlist = [0,1,2,3]

        if transsign == True:
            index = choice(indexlist)
        else:
            index = 0
        # print(index)
        pixels = []
        sign = 1
        for temp in dcmkeep:
            temp = get_pixels_hu(temp)
            temp = truncate_hu(temp)
            # scipy.misc.imsave('./res/' + str(sign) + '.jpeg', temp)
            np.save('./res/' + str(sign) + '.npy', temp)
            sign += 1
            pixels.append(temp)

        pixels = np.array(pixels, dtype=np.float)
        if index == 1:
            pixels = angle_transpose(pixels, 90)
        if index == 2:
            pixels = angle_transpose(pixels, 180)        
        if index == 3:
            pixels = angle_transpose(pixels, 270)       

        pixels = np.expand_dims(pixels, -1)

        return pixels
        # patient_pixels = get_pixels_hu(slices)#.transpose(2,1,0)
        # print(patient_pixels.shape)

    def getOnePatient2(self, testdatapath, patientName):
        dcmfiles = os.listdir(testdatapath + patientName)
        # print(dcmfiles)
        dcmfiles.sort()
        # print(dcmfiles)
        slices = [pydicom.dcmread(os.path.join(testdatapath, patientName, s)) for s in dcmfiles]
        slices.sort(key = cmpfun)
        # for onefile in dcmfiles:
        #     ds = pydicom.dcmread(os.path.join(self.datapath, patientName, onefile))
        #     # pixel = ds.pixel_array
        #     temp = get_pixels_hu(ds)
        #     temp = truncate_hu(temp)
        #     scipy.misc.imsave('./res/' + onefile + 'r.jpeg', temp)
        # pixels = []

        slicethickness = slices[0].data_element('SliceThickness').value
        # print(slicethickness)
        dcmkeep = []
        keeprate = 5.0 / slicethickness
        keeprate = int(math.floor(keeprate))
        # print(keeprate)
        if keeprate < 1:
            keeprate = 1

        # print('rate: ', keeprate)
        tempsign = 0
        for onedcm in slices:
            # print(onedcm)
            if tempsign % keeprate == 0:
                # print(tempsign)
                # print(onedcm)
                dcmkeep.append(onedcm)
            tempsign += 1
        # print(len(dcmkeep))
        if len(dcmkeep) > 30:
            x = len(dcmkeep)
            dcmkeep = dcmkeep[(x // 2 - 15) : (x // 2 + 15)]
        if len(dcmkeep) < 30:
            temp = []
            for i in range(0, 30 - len(dcmkeep)):
                temp.append(dcmkeep[0])
            dcmkeep = temp + dcmkeep
        # sign = 0
        # print('len of dcmkeep',len(dcmkeep))
        pixels = []

        # # print(type(pixels[0]))
        for temp in dcmkeep:
            # sign += 1
            temp = get_pixels_hu(temp)
            temp = truncate_hu(temp)
            # scipy.misc.imsave('./res/' + str(sign) + 'r.jpeg', temp)
            pixels.append(temp)

        pixels = np.array(pixels, dtype=np.float)
        pixels = np.expand_dims(pixels, -1)
        
        return pixels
        
        # patient_pixels = get_pixels_hu(slices)#.transpose(2,1,0)
        # print(patient_pixels.shape)

    def getThickness(self):
        return self.slicethicknesscount

    # def getAllPatients(self):
    #     return os.listdir(self.datapath)
    
    # def getAllLabels(self):
    #     lines = csvTools.readCSV(self.labelpath + 'train_label.csv')
    #     lines = lines[1:]
    #     return lines

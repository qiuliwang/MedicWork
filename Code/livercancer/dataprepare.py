import os
import csvTools
import pydicom
import numpy as np
import tensorflow as tf
import math

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

class Data(object):
    def __init__(self, datapath, labelpath = ''):
        self.datapath = datapath
        self.labelpath  = labelpath
        self.patients = os.listdir(datapath)
        lines = csvTools.readCSV(self.labelpath)
        lines = lines[1:]
        self.labels = lines
        self.slicethicknesscount = []


    def getOnePatient(self, patientName):
        dcmfiles = os.listdir(self.datapath + patientName)
        # print(dcmfiles)
        dcmfiles.sort()
        slices = [pydicom.dcmread(os.path.join(self.datapath, patientName, s)) for s in dcmfiles]
        slices.sort(key = lambda x : float(x.ImagePositionPatient[2]),reverse=True)
        pixels = []
        # if slices[0].data_element('SliceThickness').value < 5:
        #     if slices[0].data_element('SliceThickness').value not in self.slicethicknesscount:
        #         self.slicethicknesscount.append(slices[0].data_element('SliceThickness').value)
        slicethickness = slices[0].data_element('SliceThickness').value
        keeprate = 5.0 / slicethickness
        keeprate = int(math.floor(keeprate))
        if keeprate < 1:
            keeprate = 1
        # print('rate: ', keeprate)
        tempsign = 0
        for oneslice in slices:
            if tempsign % keeprate == 0:
                temppix = oneslice.pixel_array
                # temppix = tf.expand_dims(temppix, -1)
                pixels.append(temppix)
            tempsign += 1
        if len(pixels) > 70:
            print patientName
        pixels = np.array(pixels, dtype=np.float)
        pixels = np.expand_dims(pixels, -1)
        return pixels
        # patient_pixels = get_pixels_hu(slices)#.transpose(2,1,0)
        # print(patient_pixels.shape)

    def getOnePatient2(self, datapath, patientName):
        dcmfiles = os.listdir(datapath + patientName)
        # print(dcmfiles)
        dcmfiles.sort()
        slices = [pydicom.dcmread(os.path.join(datapath, patientName, s)) for s in dcmfiles]
        pixels = []
        slicethickness = slices[0].data_element('SliceThickness').value
        keeprate = 5.0 / slicethickness
        keeprate = int(math.floor(keeprate))
        if keeprate < 1:
            keeprate = 1
        # print('rate: ', keeprate)
        tempsign = 0
        for oneslice in slices:
            if tempsign % keeprate == 0:
                temppix = oneslice.pixel_array
                # temppix = tf.expand_dims(temppix, -1)
                pixels.append(temppix)
            tempsign += 1
        if len(pixels) > 70:
            print patientName
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
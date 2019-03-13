'''
Created by Wang Qiu Li
7/5/2018

get dicom info according to malignancy.csv and ld_scan.txt
get one dicom
'''
import os
import pandas as pd
import pydicom
import scipy.misc
import cv2
import numpy as np

basedir = './'
resdir = './res/'
resdir2 = 'D:/Data/LIDC-IDRI/NNNPY/'


# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 300] = 0
    image_array[image_array < -20] = 0
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

def cutTheImage(x, y, pix):
    temp = 16
    x1 = x - temp
    x2 = x + temp
    y1 = y - temp
    y2 = y + temp
    img_cut = pix[x1:x2, y1:y2]
    return img_cut

files = os.listdir('./data/')
for onefile in files:
    ds = pydicom.dcmread('./data/' + onefile)
    pix = ds.pixel_array
    # cutpix = cutTheImage(y_loc, x_loc, pix)
    pix = get_pixels_hu(ds)
    # pix = normalazation(pix)
    pix = truncate_hu(pix)
    # print(pix[0][0])
    scipy.misc.imsave('./res/' + onefile + 'r.jpeg', pix)

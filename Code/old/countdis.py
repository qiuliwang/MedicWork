x = [36, 44, 38, 34, 40, 50, 42, 30, 48, 46, 32, 54, 52, 56, 64]

'''
Created by Wang Qiu Li
7/3/2018

get dicom info according to malignancy.csv and ld_scan.txt
for all data
labels will be set in dataprepare.py
'''

import csvTools
import os
import pandas as pd
import pydicom
import scipy.misc
import cv2
import numpy as np
import tqdm

basedir = '/raid/data/LIDC/DOI/'
resdir = 'allnpy/'

noduleinfo = csvTools.readCSV('files/testdata.csv')
idscaninfo = csvTools.readCSV('files/id_scan.txt')

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 400] = 0
    image_array[image_array <-1000] = 0
    return image_array
    
# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work



def cutTheImage(x, y, pix, pixelspacing):
    temp = int((30 / pixelspacing) / 2)
    x1 = x - temp
    x2 = x + temp
    y1 = y - temp
    y2 = y + temp
    img_cut = pix[x1:x2, y1:y2]
    return img_cut

def caseid_to_scanid(caseid):
    returnstr = ''
    if caseid < 10:
        returnstr = '000' + str(caseid)
    elif caseid < 100:
        returnstr = '00' + str(caseid)
    elif caseid < 1000:
        returnstr = '0' + str(caseid)
    else:
        returnstr = str(caseid)
    return 'LIDC-IDRI-' + returnstr

f = open('errlist.txt', 'w')
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0

def dict2list(dic:dict):
    ''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst


errorcount = 0
sizelist = {}
for onenodule in tqdm.tqdm(noduleinfo):
    scanid = onenodule[1]
    scanid = caseid_to_scanid(int(scanid))
    # print(scanid)
    scanpath = ''
    for idscan in idscaninfo:
        if scanid in idscan[0]:
            scanpath = idscan[0]
            break
        
    filelist1 = os.listdir(basedir + scanpath)
    filelist2 = []
    # print(len(filelist1))
    for onefile in filelist1:
        if '.dcm' in onefile:
            filelist2.append(onefile)
    ds = pydicom.dcmread(basedir + scanpath + '/' + filelist2[1])
    # print(type(ds))
    item = ds.data_element('PixelSpacing')
    pixelspacing = item.value[0]
    spacing = int((30 / pixelspacing) / 2)
    spacing *= 2
    if spacing not in sizelist.keys():
        sizelist[spacing] = 1
    else:
        sizelist[spacing] += 1

keys = sizelist.items()
items = sorted(dict2list(sizelist), key=lambda x:x[0], reverse=False) # 按照第0个元素降序排列
print(items)
#     slices = [pydicom.dcmread(basedir + scanpath + '/' + s) for s in filelist2]
#     slices.sort(key = lambda x : float(x.ImagePositionPatient[2]),reverse=True)
#     x_loc = int(onenodule[6])
#     y_loc = int(onenodule[7])
#     z_loc = int(onenodule[8])
#     # print(x_loc, y_loc, z_loc)
#     pix = slices[z_loc - 1].pixel_array
#     cut_img = []
#     # print(np.min(cut_img))

#     # add z loc
#     zstart = z_loc - 1 - 1
#     zend = z_loc - 1 + 2

#     # tempsign = 0
#     # for zslice in slices[zstart : zend]:
#     #     pix = zslice.pixel_array
#     #     pix.flags.writeable = True

#     #     pix = truncate_hu(pix)
#     #     pix = normalazation(pix)
#     #     cutpix = cutTheImage(y_loc, x_loc, pix, pixelspacing)
#     #     cutpix = cv2.resize(cutpix, (20, 20))

#     #     # scipy.misc.imsave(str(tempsign) + '.jpeg', cutpix)
#     #     print(cutpix.shape)
#     #     if cutpix.shape[0] not in sizelist:
#     #         sizelist.append(cutpix.shape[0])
#     #     tempsign += 1
#     #     cut_img.append(cutpix)
    
#     # level = round(float(onenodule[29]))
#     try:
#         # print(sizelist)
#         # np.save(resdir + onenodule[0] + '.npy', cut_img)
        

#     except BaseException:
#         print(onenodule)
#         errorcount += 1

# # print('Done! ', errorcount)
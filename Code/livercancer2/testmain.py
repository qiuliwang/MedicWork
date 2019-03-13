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
import cv2
import scipy 
import scipy.misc

datapath1 = '/home/wangqiuli/Data/liver_cancer_dataset/train_dataset/'

labelpath1 = './train_label.csv'
data = Data(datapath1, labelpath1)
patients = data.patients
labels = data.labels

# print(len(patients))
# print(len(labels))

data.getOnePatient('0013EDC2-8D7A-4A41-AEB5-D3BB592306D2', False)
basedir = './res/'
pics = os.listdir(basedir)
pics.sort(key = lambda x: int(x[:-4]))
for onepic in pics:
    print(onepic)



def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T
 
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
 
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis
 
for i in range(len(pics) - 1):
    pic1 = np.load(basedir + pics[i])
    pic2 = np.load(basedir + pics[i + 1])
    # print(type(pic1))
    # flow = cv2.calcOpticalFlowFarneback(prevImg, nextImg, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
    flow = cv2.calcOpticalFlowFarneback(pic1, pic2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(type(flow))
    print(flow.shape)
    img1 = flow[:, :, 0]
    img2 = flow[:, :, 1]

    scipy.misc.imsave('./res1/' + str(i) + 'flow1.jpeg', img1)
    scipy.misc.imsave('./res1/' + str(i) + 'flow2.jpeg', img2)
    scipy.misc.imsave('./res2/' + str(i) + 'pic1.jpeg', pic1)
    scipy.misc.imsave('./res2/' + str(i) + 'pic2.jpeg', pic2)


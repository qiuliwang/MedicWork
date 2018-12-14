import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import traceback
import random
from PIL import Image

base_dir = "/data0/LUNA/cubic_normalization_npy"
annatation_file = base_dir + 'CSVFILES/annotations.csv'
candidate_file =base_dir +  'CSVFILES/candidates_V2.csv'

# logfile = "/home/wangqiuli/Documents/luna16_multi_size_3dcnn/loginfo.txt"
# f = open(logfile,'w')
# f.write("abcdef\n")
# f.write("ABCDEF\n")

# filelist = os.listdir(base_dir)
# real = []
# print(len(filelist))
# countreal = 0
# countfake = 0
# for filename in filelist:
#     if 'real' in filename:
#         countreal += 1
#         real.append(filename)
#     if 'fake' in filename:
#         countfake += 1
# print(countreal)
# print(real[:30])
# print(countfake)
listnum = [ 0.00426978,0.01160646,0.03154963,0.08576079,0.23312201,0.63369131]
num = 0
for number in listnum:
    num += number

print(num)
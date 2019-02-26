import os
import numpy as np 
path = 'fdata/'

filelist = os.listdir(path)
counthigh = 0
countlow = 0
countfake = 0
for onefile in filelist:
    if 'high' in onefile:
        counthigh += 1
    elif 'low' in onefile:
        countlow += 1
    elif 'fake' in onefile:
        countfake += 1
print(counthigh)
print(countlow)
print(countfake)
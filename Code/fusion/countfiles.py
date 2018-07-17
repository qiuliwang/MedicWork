import os
import numpy as np 
datadir = 'lobulation/'

filelist = os.listdir(datadir)
print(len(filelist))

counthigh = 0
countlow = 0
for onefile in filelist:
    if 'high' in onefile:
        counthigh += 1
    elif 'low' in onefile:
        countlow += 1
print(counthigh)
print(countlow)
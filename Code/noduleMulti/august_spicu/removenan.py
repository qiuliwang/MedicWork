import os
import numpy as np

path = 'train/'

filelist = os.listdir(path)

for one in filelist:
    x = np.load(path + one)
    if x.shape[0] != 10:
        print(x.shape)
        os.remove(path + one)

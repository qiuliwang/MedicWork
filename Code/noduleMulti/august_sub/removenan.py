import os
import numpy as np

path = 'train/'

filelist = os.listdir(path)

for one in filelist:
    x = np.load(path + one)
    # print(x.shape)
    if x.shape[0] != 10:
        print(x.shape)
        os.remove(path + one)

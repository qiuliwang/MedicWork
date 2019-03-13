import numpy as np 

for i in range(0, 5):
    x = np.ones([3,42,42])
    np.save(str(i) + '.npy', x)
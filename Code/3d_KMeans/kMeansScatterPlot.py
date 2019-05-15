from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.cluster import KMeans
import numpy as np
import csvTools
from scipy.spatial.distance import cdist

data = csvTools.readCSV('annotations.csv')
data = data[1:]

def getK(mat):
    km_n_clusters = range(1,30)
    meandistortions = []
    for k in km_n_clusters:
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(mat)
        meandistortions.append(sum(np.min(
                cdist(mat,kmeans.cluster_centers_,
                     'euclidean'),axis=1))/mat.shape[0])
    print(meandistortions)

    kt = 0
    for i in range(len(meandistortions) - 1):
        x = (meandistortions[i] - meandistortions[i + 1])
        if x < 1:
            print (i)
            kt = i
            break

    return kt 

def getClusterCenter(data):
    coordinate = []
    xx = []
    yy = []
    zz = []
    for one in data:
        z = int(one[2])
        x = int(one[3])
        y = int(one[4])
        zz.append(z)
        yy.append(y)
        xx.append(x)
        temp = [z,x,y]
        coordinate.append(temp)
        
    mat = np.array(coordinate)
    k = getK(mat)
    clt = KMeans(n_clusters=k)
    clt.fit(mat)

    flatClt = clt.cluster_centers_
    labels = clt.labels_
    print(flatClt)
    print(labels)
    for onedata, labels in zip(data, labels):
        onedata.append(labels)
    
    return flatClt, data

centers, data = getClusterCenter(data)
print(len(data))
print(data[0])
print(centers)

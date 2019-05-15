from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.cluster import KMeans
import numpy as np
import csvTools
from scipy.spatial.distance import cdist

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
    # kmeans=KMeans(n_clusters=kt)

    # plt.plot(km_n_clusters,meandistortions,'bx-')
    # plt.xlabel('k')
    # plt.ylabel(u'平均畸变程度')
    # plt.title(u'用肘部法则来确定最佳的K值')
    # plt.show()
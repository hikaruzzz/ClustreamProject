import numpy as np
import pandas as pd
import math
import scipy.io as scio
import random
import operator
from sklearn.decomposition import PCA
from CluStream import CluStream
from outlierdenstream import Sample, MicroCluster, Cluster, OutlierDenStream
from scipy.spatial import distance

from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN

from utils import *


# load dataset & data transform
#S1_X,S1_Y,S2_X,S2_Y,totalnum = transformer_bbcsport_2()
#S1_X,S1_Y,S2_X,S2_Y,totalnum = transformer_3Source()
#S1_X,S1_Y,S2_X,S2_Y,totalnum = transformer_Caltech101_7()
#S1_X,S1_Y,S2_X,S2_Y,totalnum = transformer_Caltech101_20()
S1_X,S1_Y,S2_X,S2_Y,totalnum = transformer_Mfeat()
#S1_X,S1_Y,S2_X,S2_Y,totalnum = transformer_MSRC_v1()

# pca处理S1(训练集）
pca = PCA(n_components=0.95, svd_solver='full')
S1_reduction_Xo = pca.fit_transform(S1_X)  # PCA: shape (n_samples, n_features)
normalizer = Normalizer().fit(S1_reduction_Xo)
S1_reduction_X = normalizer.transform(S1_reduction_Xo)


# # 进行初始的microcluster训练
# clu = CluStream(nb_initial_points=100)
# clu.fit(S1_reduction_X)
#
# den = OutlierDenStream(lamb=0.00025, epsilon=0.5, startingBuffer=S1_reduction_X, numberInitialSamples=totalnum,
#                        beta=0.6, mu=3.4, tp=500)
# den.runDBSCanInitialization()


# show
S1_result = np.zeros(int(S1_Y.max())+1,dtype=np.int)
print("total cluster points {}".format(int(S1_Y.max())))
for i in range(totalnum):
    S1_result[int(S1_Y[i][0])] += 1
print(S1_result)




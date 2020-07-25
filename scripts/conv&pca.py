#!/usr/bin/env python
# coding: utf-8

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import signal
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

scharr=np.array([[1,2,1],
                [2,4,2],
                [1,2,1]])

def Two_layer_conv2d(old_arr):
    mid=signal.convolve2d(old_arr,scharr,boundary='fill',mode='valid')
    new_arr=signal.convolve2d(mid,scharr,boundary='fill',mode='valid')
    return new_arr

data = None
first_time = True
for i in range(1979, 2019):
    ds = Dataset('./data/ERA5/download_ ' + str(i) + '.nc', "r+", format="NETCDF4")
    sec = np.ndarray(shape=(ds.variables['t'].shape[0],7,9,13), dtype=float)
    k=0
    for j in list(ds.variables.keys())[3:]:
        if ds.variables[j][:].mask == False:
            for index,a in enumerate(ds.variables[j][:]):
                sec[index, k, :, :] = Two_layer_conv2d(a)
            k+=1
    
    if first_time:
        data = sec
        first_time =False
    else:
        data = np.append(data, sec, axis=0)
    print("epoch " + str(i-1978) + ": current shape= " + str(data.shape))    


input_X = np.reshape(data, (data.shape[0], -1))

pca=PCA(n_components= 30)
tran_X=pca.fit_transform(input_X)

X_train, X_test = train_test_split(tran_X, test_size=0.2, random_state=42)


kmeans = KMeans(n_clusters=12, random_state= 100, max_iter=50000 ).fit(X_train)
print(kmeans.n_iter_)


# In[78]:


print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          %(kmeans.n_iter_,
      homogeneity_score(y_test, y_pred),
      completeness_score(y_test, y_pred),
      v_measure_score(y_test, y_pred),
      adjusted_rand_score(y_test, y_pred),
      adjusted_mutual_info_score(y_test, y_pred),
      silhouette_score(X_test, y_pred, metric='euclidean')))


# In[37]:


unique, counts = np.unique(kmeans.labels_, return_counts=True)
print(dict(zip(unique, counts)))




from netCDF4 import Dataset
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import pandas as pd
# from scipy import signal
# from sklearn.preprocessing import normalize

# time series data downsampling
# ARIMA
<<<<<<< HEAD
# ARMA
=======
>>>>>>> e52308794f738e4055f6ad89169fcf03463e1fc3
# DBSCAN
# LSTM
# PCA
# MDS

window_size = 6

for i in range(1979, 2020):
<<<<<<< HEAD
    data = Dataset("../data/ERA5/download_ " + str(i) + ".nc", "r+", format="NETCDF4")
=======
    data = Dataset("/Volumes/GoogleDrive/我的雲端硬碟/Research/Data/ERA5/download_ " + str(i) + ".nc", "r+", format="NETCDF4")
>>>>>>> e52308794f738e4055f6ad89169fcf03463e1fc3
    new_first_dim_num = data.variables['t'].shape[0] - window_size + 1
    var_num = len(list(data.variables.keys())[3:])
    new_input = np.ndarray(shape=(new_first_dim_num,window_size *var_num), dtype=float)
    for k in range(new_first_dim_num):
        for l in range(window_size):
            for j_index, j in enumerate(list(data.variables.keys())[3:]):
                new_input[k, l * var_num + j_index] = data.variables[j][k+l].mean()
		
    df_csv = pd.DataFrame(new_input)
<<<<<<< HEAD
    print((df_csv.notna()==True).count())
    df_csv.to_csv('../data/remake/remake_' + str(i) + '.csv')
=======
    df_csv.to_csv('/Volumes/GoogleDrive/我的雲端硬碟/Research/Data/remake/remake_' + str(i) + '.csv')
>>>>>>> e52308794f738e4055f6ad89169fcf03463e1fc3

from netCDF4 import Dataset
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pdb
# from scipy import signal
# from sklearn.preprocessing import normalize

# time series data downsampling
# ARIMA
# ARMA
# DBSCAN
# LSTM
# PCA
# MDS

window_size = 24

for i in range(1979, 2020):
    data = Dataset("/Volumes/GoogleDrive/我的雲端硬碟/Research/Data/ERA5/download_ " + str(i) + ".nc", "r+", format="NETCDF4")
    var_num = len(list(data.variables.keys())[3:])
    #先transfer into (hour number , 7)
    #空間上做平均
    new_data = np.ndarray(shape=(data.variables['t'].shape[0], var_num), dtype=float)
    for hour in range(data.variables['t'].shape[0]):
        for var_index, var in enumerate(list(data.variables.keys())[3:]):
            new_data[hour, var_index] = data.variables[var][hour].mean()
    #計算統計特性參數（mean, std, ar1)
    new_first_dim_num = data.variables['t'].shape[0] - window_size + 1
    new_output = np.ndarray(shape=(new_first_dim_num, 3 * var_num), dtype=float)
    pdb.set_trace()
    for row in range(new_first_dim_num):
        for col in range(0, var_num, 3):
            sin = new_data[row:(row + window_size)][:, col]
            new_output[row, col] = cal_mean(sin)
            new_output[row, col + 1] = cal_std(sin)
            new_output[row, col + 2] = cal_ar1(sin)
		
    df_csv = pd.DataFrame(new_output)
    df_csv.to_csv('/Volumes/GoogleDrive/我的雲端硬碟/Research/Data/remake/remake_' + str(i) + '.csv')
    

def cal_mean(target_array):
    return np.mean(target_array)

def cal_std(target_array):
    return np.std(target_array)

def cal_ar1(target_array):
    #let alpha = 0.6
    w = np.random.normal(size=len(target_array))
    
    return 1

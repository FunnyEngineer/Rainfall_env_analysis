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

def cal_mean(target_array):
    return np.mean(target_array)

def cal_std(target_array):
    return np.std(target_array)

def cal_lag1(target_array):    
    return pd.Series(target_array).autocorr(lag=1)

def main():
	window_size = 24

	for i in range(1979, 2020):
	    data = Dataset("../data/ERA5/download_ " + str(i) + ".nc", "r+", format="NETCDF4")
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
	    for row in range(new_first_dim_num):
	        for col in range(var_num):
	            si = new_data[row:(row + window_size)][:, col]
	            new_output[row, 3 * col] = cal_mean(si)
	            new_output[row, 3 * col + 1] = cal_std(si)
	            new_output[row, 3 * col + 2] = cal_lag1(si)

	    df_csv = pd.DataFrame(new_output)
	    df_csv.to_csv('../data/tran_3_var/3var_' + str(i) + '.csv')
	    print('year {} finished.'.format(i))
	    
if __name__ == "__main__":
	main()

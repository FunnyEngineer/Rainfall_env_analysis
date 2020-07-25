from netCDF4 import Dataset
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pdb
import pickle
from datetime import *
# from scipy import signal
# from sklearn.preprocessing import normalize

# time series data downsampling
# ARIMA
# ARMA
# DBSCAN
# LSTM
# PCA
# MDS

def main():
	window_size = 12

	for i in range(1979, 2020):
	    data = Dataset("../data/ERA5/download_ " + str(i) + ".nc", "r+", format="NETCDF4")
	    var_num = len(list(data.variables.keys())[3:])
	    #先transfer into (hour number, 7)
	    #空間上做平均
	    new_first_dim_num = data.variables['t'].shape[0]
	    new_data = np.ndarray(shape=(new_first_dim_num, var_num), dtype=float)
	    for hour in range(new_first_dim_num):
	        for var_index, var in enumerate(list(data.variables.keys())[3:]):
	        	new_data[hour, var_index] = data.variables[var][hour].mean()

	    df_csv = pd.DataFrame(new_data)
	    time_list = []
	    since_date = datetime(1900,1,1)
	    for j in data.variables['time'][:]:
	    	time_list.append(since_date + timedelta(hours = int(j)))
	    df_csv['Datetime'] = time_list
	    df_csv = df_csv.set_index('Datetime')
	    df_csv.to_csv('../data/raw_mean/raw_mean_' + str(i) + '.csv')
	    print('year {} finished.'.format(i))
	    # df_csv = pd.DataFrame(new_output)
	    # df_csv.to_csv('../data/tran_3_var/3var_' + str(i) + '.csv')
	    # print('year {} finished.'.format(i))
	    
if __name__ == "__main__":
	main()

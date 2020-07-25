from netCDF4 import Dataset
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pdb
import pickle
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
	    #先transfer into (hour number , window_size, 7)
	    #空間上做平均
	    new_first_dim_num = data.variables['t'].shape[0] - window_size + 1
	    new_data = np.ndarray(shape=(new_first_dim_num , window_size, var_num), dtype=float)
	    for hour in range(new_first_dim_num):
	        for var_index, var in enumerate(list(data.variables.keys())[3:]):
	        	for sec_dim in range(window_size):
	        		new_data[hour, sec_dim, var_index] = data.variables[var][hour + sec_dim].mean()
	    file = open('../data/multi_dim/multi_dim_' + str(i) + '.pickle', 'wb')
	    pickle.dump(new_data, file)
	    file.close()
	    # df_csv = pd.DataFrame(new_output)
	    # df_csv.to_csv('../data/tran_3_var/3var_' + str(i) + '.csv')
	    # print('year {} finished.'.format(i))
	    
if __name__ == "__main__":
	main()

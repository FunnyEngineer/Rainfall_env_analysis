from netCDF4 import Dataset
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import pandas as pd
# from scipy import signal
# from sklearn.preprocessing import normalize

window_szie = 6

for i in range(1979, 2019):
		data = Dataset("../data/ERA5/download_ " + str(i) + ".nc", "r+", format="NETCDF4")
		new_first_dim_num = data.variables['t'].shape[0] - window_szie + 1
		new_input = np.ndarray(shape=(new_first_dim_num,7), dtype=float)
		for k in range(new_first_dim_num):
			for j_index, j in enumerate(list(data.variables.keys())[3:]):
				new_input[k, j_index] = data.variables[j][k:(k+6)].mean()
		
		df_csv = pd.DataFrame(new_input, columns = list(data.variables.keys())[3:])
		df_csv.to_csv('../data/remake/remake_' + str(i) + '.csv')  
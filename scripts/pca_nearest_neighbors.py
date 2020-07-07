import  numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pdb
from netCDF4 import Dataset

pca = PCA(n_components=7)
neigh = NearestNeighbors(n_neighbors=30, algorithm='brute')
total_data = None
for i in range(1979, 2019):
	data = np.genfromtxt('../data/remake/remake_' + str(i)+'.csv', delimiter=',')
	data = np.delete(data[1:], 0, 1)
	if total_data is None:
		total_data = data
	else:
		total_data = np.concatenate((total_data, data), axis=0)
print("Total data shpae: " + str(total_data.shape))
total_tran = pca.fit_transform(total_data)
neigh.fit(total_tran)
# for i in range(1979, 2019):
# 	data = np.genfromtxt('../data/remake/remake_' + str(i)+'.csv', delimiter=',')
# 	data = np.delete(data[1:], 0, 1)
# 	new_data = pca.transform(data)
# 	neigh.fit(new_data)

new_test = pd.read_csv('../data/Events_Data.csv')
sin_test_date = new_test['start datetime'][0]
date = datetime.strptime(sin_test_date, '%Y/%m/%d %I:%M')
time_index_table = pd.read_csv('../data/time_index.csv')
time_index_table['datetime'] = pd.to_datetime(time_index_table['datetime'], format="%m/%d/%Y, %H:%M") 
time_index_table['datetime'] = abs(time_index_table['datetime'] - date)
#pdb.set_trace()

new_input_index = time_index_table['datetime'].idxmin()
new_input = total_tran[new_input_index]
new_input = np.reshape(new_input, (1, 7))
distances, indices = neigh.kneighbors(new_input)
time_index_table = pd.read_csv('../data/time_index.csv')
time_index_table['datetime'] = pd.to_datetime(time_index_table['datetime'], format="%m/%d/%Y, %H:%M") 
for rank, i in enumerate(indices[0]):
	if time_index_table['datetime'][i] > datetime(2010, 1, 1):
		print("Rank {}: ".format(rank) +str(time_index_table['datetime'][i]))

# # test example 2014/07/08 10:15
# since = datetime(1900, 1, 1)
# test_data = Dataset("../data/ERA5/download_ 2014.nc", "r+", format="NETCDF4")
# new_input = np.ndarray(shape=(1,7), dtype=float)
# index = 0
# for time in test_data.variables['time']:
# 	temp = since + timedelta(hours = time.data.item(0))
# 	index+=1
# 	if temp == datetime(2014, 7, 8, hour=10):
# 		for j_index, j in enumerate(list(test_data.variables.keys())[3:]):
# 				new_input[0][j_index] = test_data.variables[j][index:(index+6)].mean()
# 		break

# tran_input = pca.transform(new_input)
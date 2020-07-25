import  numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pdb
from netCDF4 import Dataset


##using pca and knn first fit model and transfer data
pca = PCA(n_components=7)
neigh = NearestNeighbors(n_neighbors=100, algorithm='brute')
total_data = None
for i in range(1979, 2019):
	data = pd.read_csv('../data/tran_3_var/3var_' + str(i)+'.csv', index_col=0).fillna(0)
	#data = np.delete(data[1:], 0, 1)
	if total_data is None:
		total_data = data
	else:
		total_data = pd.concat([total_data, data])
print("Total data shpae: " + str(total_data.shape))
total_tran = pca.fit_transform(total_data)
neigh.fit(total_tran)


# for i in range(1979, 2019):
# 	data = np.genfromtxt('../data/remake/remake_' + str(i)+'.csv', delimiter=',')
# 	data = np.delete(data[1:], 0, 1)
# 	new_data = pca.transform(data)
# 	neigh.fit(new_data)

new_test = pd.read_csv('../data/Events_Data.csv')
new_test['start datetime'] = pd.to_datetime(new_test['start datetime'] , format='%Y/%m/%d %H:%M')
new_test['start datetime'] = [i.round('h') for i in new_test['start datetime']]
new_test['end datetime'] = pd.to_datetime(new_test['end datetime'] , format='%Y/%m/%d %H:%M')
new_test['end datetime'] = [i.round('h') for i in new_test['end datetime']]

#set earliest date for check
earliest_date = new_test['start datetime'][new_test['start datetime'].idxmin()]

# set test date for prediction
text_file = open("Output.txt", "w")
# pdb.set_trace()
for event_index in range(len(new_test)):
	hour_count = int((new_test['end datetime'][event_index] - new_test['start datetime'][event_index]).seconds / 3600)
	text_file.write("Event {}: start from {}:\n".format(event_index, new_test['start datetime'][event_index]))
	in_per= 0
	for hour in range(hour_count):
		date = new_test['start datetime'][event_index] + timedelta(hours = hour)
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
			new_date = time_index_table['datetime'][i]
			if new_date > earliest_date:
				for gg in range(len(new_test)):
					if new_date > new_test['start datetime'][gg] and new_test['end datetime'][gg] > new_date:
						text_file.write("		Time {} Rank {} : {} is in the period.\n".format(date, rank, time_index_table['datetime'][i]))
						in_per += 1
						break
				#print("Rank {}: ".format(rank)+str(time_index_table['datetime'][i])  + " is not in the period.")
	text_file.write('For event {}: There are {} / {} prediction in the period.\n\n'.format(event_index, in_per * hour_count,in_per))
	print("Event {} finished.".format(event_index))
		
#pdb.set_trace()
text_file.close()
	# if time_index_table['datetime'][i] > datetime(2005, 7, 3):
	# 	print("Rank {}: ".format(rank) +str(time_index_table['datetime'][i]))

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
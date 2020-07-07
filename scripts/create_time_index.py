from datetime import datetime, timedelta
from netCDF4 import Dataset, date2num, num2date
import numpy as np
import csv

index = 0
since = datetime(1900, 1, 1)
with open('time_index.csv',  'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['index', 'datetime'])
    for i in range(1979, 2019):
        ds = Dataset('/Volumes/GoogleDrive/我的雲端硬碟/Research/Data/ERA5/download_ ' + str(i)+'.nc', "r+", format="NETCDF4")
        for time in ds.variables['time']:
            temp = since + timedelta(hours = time.data.item(0))
            writer.writerow([index, temp.strftime("%m/%d/%Y, %H:%M")]) #.strftime("%m/%d/%Y, %H:%M")
            index += 1
            # so if you want to convert string to datetime
            # datetime.strptime(test_dat_str, '%Y/%m/%d %I:%M')
            # for version 1 date time use 
            # df['datetime'] = pd.to_datetime(df['datetime'], format="%m/%d/%Y, %H:%M") 
            # to convert string to datetime

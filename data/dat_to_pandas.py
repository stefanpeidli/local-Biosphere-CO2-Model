import numpy as np
import pandas as pd

raw_data = np.loadtxt('zugspitze2016.dat', usecols=4, skiprows=33)
raw_dates = np.loadtxt('zugspitze2016.dat', dtype=str, usecols=0, skiprows=33)
raw_times = np.loadtxt('zugspitze2016.dat', dtype=str, usecols=1, skiprows=33)


# some cleaning
good_apples = raw_data > 0
clean_data = raw_data[good_apples]
clean_dates = raw_dates[good_apples]
clean_times_str = raw_times[good_apples]

clean_times = np.zeros(shape=clean_times_str.shape, dtype=int)
# data for whole year:
for i in range(len(clean_times)):
    clean_times[i] = int(clean_times_str[i][0:2])

data = pd.DataFrame(clean_data, columns=['CO2'], index=clean_dates)
data['times'] = clean_times
print(data)
data.to_csv('2016_zugspitze')

# script for reading, filtering/cleaning and viewing data

import numpy as np
import matplotlib.pyplot as plt


raw_data = np.loadtxt('zugspitze2016.dat', usecols=4, skiprows=33)
raw_dates = np.loadtxt('zugspitze2016.dat', dtype=str, usecols=0, skiprows=33)
raw_times = np.loadtxt('zugspitze2016.dat', dtype=str, usecols=1, skiprows=33)

# some cleaning
good_apples = raw_data > 0
clean_data = raw_data[good_apples]
clean_dates = raw_dates[good_apples]
clean_times_str = raw_times[good_apples]

clean_times = np.zeros(shape=clean_times_str.shape)
# data for whole year:
for i in range(len(clean_times)):
    clean_times[i] = int(clean_times_str[i][0:2])


for i in range(4):
    day = clean_dates[24*30*4+23*i]
    day_data = clean_dates == day
    plt.plot(clean_times[day_data], clean_data[day_data])

plt.show()


#######
# Sun position data
raw_sun_pos = np.loadtxt('sundata.txt', dtype=float, usecols=1, skiprows=6)
raw_sun_pos2 = np.loadtxt('sundata.txt', dtype=str, usecols=3, skiprows=6)
raw_sun_times = np.loadtxt('sundata.txt', dtype=str, usecols=0, skiprows=6)

sun_arc = []
for i in range(len(raw_sun_times)):
    if raw_sun_pos2[i] == 'n/a':
        sun_arc.append(180)
    else:
        sun_arc.append(float(raw_sun_pos2[i][0:-1]))
sun_arc = np.array(sun_arc)

full_hours = []
clean_sun_times_hours =[]
for i in range(len(raw_sun_times)):
    if raw_sun_times[i][3:5] == '00':
        full_hours.append(True)
        clean_sun_times_hours.append(float(raw_sun_times[i][0:2]))
    else:
        full_hours.append(False)
clean_sun_pos = sun_arc[full_hours]
clean_sun_times = raw_sun_times[full_hours]
intensity = np.arctan(1/clean_sun_pos)

#plt.plot(clean_sun_times, intensity)
#plt.show()
print(intensity)
print(clean_sun_times_hours)


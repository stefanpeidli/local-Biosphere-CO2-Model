# A very simple and basic model for local CO2 concentration

import numpy as np
import matplotlib.pyplot as plt
import solar as sol


# Photosynthesis rate at intensity, Michaelis-Menden-like
def pi_curve(intensity, p_max=10, ki=1/2):
    # p_max     maximal photosynthesis rate
    # ki        half-saturation intensity
    return p_max * intensity / (ki + intensity)


# Model parameters
C = [406]  # init val
r_d = 6  # Day CO2 production factor by respiration
r_n = 7  # Day CO2 production factor by respiration
p_max = 15
ki = 1/16

# Data values for some July day
intensity = [0.0055555, 0.1296468,  0.30467442, 0.48017503, 0.64833199, 0.80062541,0.91461761, 0.97361967, 0.95339044, 0.86052532, 0.72444091, 0.56142098,0.38895688, 0.21271527, 0.04335676]

times = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
inte = sol.SPA_day(20, 5, 2018, 1, 47, 10, 3000)
print(inte)
inte = np.arctan(1/inte)
print(inte)
plt.plot(range(24), inte)
plt.show()

# The Model, Euler Method
for i in range(14*60):  # Daytime
    cur_hour = round(i/60)
    C.append(C[-1] + (-pi_curve(intensity[cur_hour], p_max, ki) + r_d) / (14*60))
for i in range(10*60):  # Night
    C.append(C[-1] + r_n / (10 * 60))

# Formating
A = np.arange(6*60, 24*60+1)
B = np.arange(0, 6*60)
tim = np.concatenate((A, B))
tim = np.round(tim/60)

# Plot
plt.scatter(tim, C)
plt.show()



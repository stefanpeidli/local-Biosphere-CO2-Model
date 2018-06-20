# A very simple and basic model for local CO2 concentration

import numpy as np
import matplotlib.pyplot as plt
from utils import solar as sol


# Photosynthesis rate at intensity, Michaelis-Menden-like
def pi_curve(intensity, p_max=10, ki=1/2):
    # p_max     maximal photosynthesis rate
    # ki        half-saturation intensity
    return p_max * intensity / (ki + intensity)


def tomato(intensity, temperature, cotwo, a, b, c, d):
    # a is constant factor
    # b, c, d half saturation for intensity, temperature, cotwo
    return a * intensity * temperature * cotwo/((b+intensity)*(c + temperature)*(d + cotwo))


def nelken(intensity, temperature, cotwo, a, b, c, d):
    return a * intensity**b * temperature**c * cotwo**d


# The model

def evolve(init_concentration, init_time=0, init_date=(6,6,2018), duration=24, stepsize=0.1, p=10, k=1, c_e=0.5, ce1=1, ce2=1, Absorption=0, Emission=0):
    # EVOLVE : evolves an initial concentration in time.
    # init_concentration: initial concentration at init_time at the measurement site
    # init_time: Time in hours on the clock when evolving starts (0 means 0:00 in the morning)
    # init_date: Initial date in format (Day,month,year)
    # duration: Denotes the time in hours how long the system will be evolved
    # stepsize: Stepsize for the integration method
    # Absorption: Absorption model ID
    # Emission: Emission model ID

    C = [init_concentration]
    time = np.arange(init_time, init_time + duration, stepsize) % 24
    for t in time[1:]:
        cur_hour = round(t)
        absorp = absorption(0, cur_hour, p=p, k=k)
        emiss = emission(0, c_e)
        net_change = absorp + emiss
        new_value = C[-1] + net_change * stepsize  # eulerian method
        C.append(new_value)
    return [time, C]


def absorption(ID, cur_hour, p=1, k=1/16):
    if ID == 0:  # PI-Curve / Michaelis-Menten-Model
        return -pi_curve(sol.solar_intensity(hour=cur_hour), p, k)
    elif ID == 1:  # Tomato
        return
    elif ID == 2:  # Nelke
        return


def emission(ID, c_e=0.5, temperature=10):
    if ID == 0:  # Constant model
        constant = c_e
        return constant
    if ID == 1:  # Gent & Enoch model
        c1 = 1
        c2 = 1
        return c1*np.exp(c2 * (temperature - 25))


# Simulations
if __name__ == "__main__":
    stepsize = 0.1
    #Day1 = model_day(stepsize)
    #Day2 = model_day(stepsize, init_val=Day1[-1])
    res = evolve(406, 0, (6, 6, 2018), 23, stepsize, p=10, k=1)

    # Plot
    #plt.plot(np.arange(0, 24, stepsize), Day1)
    #plt.plot(np.arange(0, 24, stepsize), Day2)
    plt.plot(res[0], res[1])
    plt.show()

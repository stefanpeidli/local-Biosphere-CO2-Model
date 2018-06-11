"""Coded by Patrick
This code describes the Solar Position Algorithm (SPA) based on a paper written
by Ibrahim Reda abd Afshin Andreas. This algorithm aims to compute the solar
zenith and azimuth angles in the period from -2000 to 6000 years, with the 
uncertainties of +/- 0.0003°. It also computes the angle of incidence of sunlight
for a surface that is tilted to any horizontal and verticle angle.
"""
import numpy as np
from math import pow

L0_data = np.matrix('175347046 0 0; 3341656 4.6692568 6283.07585; 34894 4.6261 12566.1517;3497 2.7441 5753.3849; \
                3418 2.8289 3.5231; 3136 3.6277 77713.7715; 2676 4.4181 7860.4194; \
                2343 6.1352 3930.2097; 1324 0.7425 11506.7698; 1273 2.0371 529.691; \
                1199 1.1096 1577.3435; 990 5.233 5884.927; 902 2.045 26.298; 857 3.508 398.149;\
                780 1.179 5223.694; 753 2.533 5507.553; 505 4.583 18849.228; 492 4.205 775.523;\
                357 2.92 0.067; 317 5.849 11790.629; 284 1.899 796.298; 271 0.315 10977.079;\
                243 0.345 5486.778; 206 4.806 2544.314; 205 1.869 5573.143; 202 2.458 6069.777;\
                156 0.833 213.299; 132 3.411 2942.463; 126 1.083 20.775; 115 0.645 0.98;\
                103 0.636 4694.003; 102 0.976 15720.839; 102 4.267 7.114; 99 6.21 2146.17;\
                98 0.68 155.42; 86 5.98 161000.69; 85 1.3 6275.96; 85 3.67 71430.7;\
                80 1.81 17260.15; 79 3.04 12036.46; 75 1.76 5088.63; 74 3.5 3154.69;\
                74 4.68 801.82; 70 0.83 9437.76; 62 3.98 8827.39; 61 1.82 7084.9;\
                57 2.78 6286.6; 56 4.39 14143.5; 56 3.47 6279.55; 52 0.19 12139.55;\
                52 1.33 1748.02; 51 0.28 5856.48; 49 0.49 1194.45; 41 5.37 8429.24;\
                41 2.4 19651.05; 39 6.17 10447.39; 37 6.04 10213.29; 37 2.57 1059.38;\
                36 1.71 2352.87; 36 1.78 6812.77; 33 0.59 17789.85; 30 0.44 83996.85;\
                30 2.74 1349.87; 25 3.16 4690.48')

L1_data = np.matrix('628331966747 0 0; 206059 2.678235 6283.07585; 4303 2.6351 12566.1517;\
                    425 1.59 3.523; 119 5.796 26.298; 109 2.966 1577.344; 93 2.59 18849.23;\
                    72 1.14 529.69; 68 1.87 398.15; 67 4.41 5507.55; 59 2.89 5223.69;\
                    56 2.17 155.42; 45 0.4 796.3; 36 0.47 775.52; 29 2.65 7.11; 21 5.34 0.98;\
                    19 1.85 5486.78; 19 4.97 213.3; 17 2.99 6275.96; 16 0.03 2544.31;\
                    16 1.43 2146.17; 15 1.21 10977.08; 12 2.83 1748.02; 12 3.26 5088.63;\
                    12 5.27 1194.45; 12 2.08 4694; 11 0.77 553.57; 10 1.3 6286.6; 10 4.24 1349.87;\
                    9 2.7 242.73; 9 5.64 951.72; 8 5.3 2352.87; 6 2.65 9437.76; 6 4.67 4690.48')

L2_data = np.matrix('52919 0 0; 8720 1.0721 6283.0758; 309 0.867 12566.152;\
                    27 0.05 3.52; 16 5.19 26.3; 16 3.68 155.42; 10 0.76 18849.23;\
                    9 2.06 77713.77; 7 0.83 775.52; 5 4.66 1577.34; 4 1.03 7.11;\
                    4 3.44 5573.14; 3 5.14 796.3; 3 6.05 5507.55; 3 1.19 242.73;\
                    3 6.12 529.69; 3 0.31 398.15; 3 2.28 553.57; 2 4.38 5223.69;\
                    2 3.75 0.98')

L3_data = np.matrix('289 5.844 6283.076; 35 0 0; 17 5.49 12566.15; 3 5.2 155.42;\
                    1 4.72 3.52; 1 5.3 18849.23; 1 5.97 242.73')

L4_data = np.matrix('114 3.142 0; 8 4.13 6283.08; 1 3.84 12566.15')

L5_data = np.matrix('1 3.14 0')

B0_data = np.matrix('280 3.199 84334.662; 102 5.422 5507.553; 80 3.88 5223.69;\
                    44 3.7 2352.87; 32 4 1577.34')

B1_data = np.matrix('9 3.9 5507.55; 6 1.73 5223.69')

R0_data = np.matrix('100013989 0 0; 1670700 3.0984635 6283.07585; 13956 3.05525 12566.1517;\
                    3084 5.1985 77713.7715; 1628 1.1739 5753.3849; 1576 2.8469 7860.4194; \
                    925 5.453 11506.77; 542 4.564 3930.21; 472 3.661 5884.927; 346 0.964 5507.553;\
                    329 5.9 5223.694; 307 0.299 5573.143; 243 4.273 11790.629; 212 5.847 1577.344;\
                    186 5.022 10977.079; 175 3.012 18849.228; 110 5.055 5486.778; 98 0.89 6069.78;\
                    86 5.69 15720.84; 86 1.27 161000.69; 65 0.27 17260.15; 63 0.92 529.69; 57 2.01 83996.85;\
                    56 5.24 71430.7; 49 3.25 2544.31; 47 2.58 775.52; 45 5.54 9437.76; 43 6.01 6275.96;\
                    39 5.36 4694; 38 2.39 8827.39; 37 0.83 19651.05; 37 4.9 12139.55; 36 1.67 12036.46;\
                    35 1.84 2942.46; 33 0.24 7084.9; 32 0.18 5088.63; 32 1.78 398.15; 28 1.21 6286.6;\
                    28 1.9 6279.55; 26 4.59 10447.39')

R1_data = np.matrix('103019 1.10749 6283.07585; 1721 1.0644 12566.1517; 702 3.142 0; 32 1.02 18849.23;\
                    31 2.84 5507.55; 25 1.32 5223.69; 18 1.42 1577.34; 10 5.91 10977.08; 9 1.42 6275.96;\
                    9 0.27 5486.78')

R2_data = np.matrix('4359 5.7846 6283.0758; 124 5.579 12566.152; 12 3.14 0; 9 3.63 77713.77; 6 1.87 5573.14;\
                    3 5.47 18849.23')

R3_data = np.matrix('145 4.273 6283.076; 7 3.92 12566.15')

R4_data = np.matrix('4 2.56 6283.08')

other_data = np.matrix('0 0 0 0 1 -171996 -174.2 92025 8.9;\
                       -2 0 0 2 2 -13187 -1.6 5736 -3.1;\
                       0 0 0 2 2 -2274 -0.2 977 -0.5;\
                       0 0 0 0 2 2062 0.2 -895 0.5;\
                       0 1 0 0 0 1426 -3.4 54 -0.1;\
                       0 0 1 0 0 712 0.1 -7 0;\
                       -2 1 0 2 2 -517 1.2 224 -0.6;\
                       0 0 0 2 1 -386 -0.4 200 0; \
                       0 0 1 2 2 -301 0 129 -0.1;\
                       -2 -1 0 2 2 217 -0.5 -95 0.3;\
                       -2 0 1 0 0 -158 0 0 0; \
                       -2 0 0 2 1 129 0.1 -70 0;\
                       0 0 -1 2 2 123 0 -53 0;\
                       2 0 0 0 0 63 0 0 0; \
                       0 0 1 0 1 63 0.1 -33 0;\
                       2 0 -1 2 2 -59 0 26 0;\
                       0 0 -1 0 1 -58 -0.1 32 0;\
                       0 0 1 2 1 -51 0 27 0;\
                       -2 0 2 0 0 48 0 0 0;\
                       0 0 -2 2 1 46 0 -24 0;\
                       2 0 0 2 2 -38 0 16 0;\
                       0 0 2 2 2 -31 0 13 0;\
                       0 0 2 0 0 29 0 0 0;\
                       -2 0 1 2 2 29 0 -12 0;\
                       0 0 0 2 0 26 0 0 0;\
                       -2 0 0 2 0 -22 0 0 0;\
                       0 0 -1 2 1 21 0 -10 0;\
                       0 2 0 0 0 17 -0.1 0 0;\
                       2 0 -1 0 1 16 0 -8 0;\
                       -2 2 0 2 2 -16 0.1 7 0;\
                       0 1 0 0 1 -15 0 9 0;\
                       -2 0 1 0 1 -13 0 7 0;\
                       0 -1 0 0 1 -12 0 6 0;\
                       0 0 2 -2 0 11 0 0 0;\
                       2 0 -1 2 1 -10 0 5 0;\
                       2 0 1 2 2 -8 0 3 0;\
                       0 1 0 2 2 7 0 -3 0;\
                       -2 1 1 0 0 -7 0 0 0;\
                       0 -1 0 2 2 -7 0 3 0;\
                       2 0 0 2 1 -7 0 3 0;\
                       2 0 1 0 0 6 0 0 0;\
                       -2 0 2 2 2 6 0 -3 0;\
                       -2 0 1 2 1 6 0 -3 0;\
                       2 0 -2 0 1 -6 0 3 0;\
                       2 0 0 0 1 -6 0 3 0;\
                       0 -1 1 0 0 5 0 0 0;\
                       -2 -1 0 2 1 -5 0 3 0;\
                       -2 0 0 0 1 -5 0 3 0;\
                       0 0 2 2 1 -5 0 3 0;\
                       -2 0 2 0 1 4 0 0 0;\
                       -2 1 0 2 1 4 0 0 0;\
                       0 0 1 -2 0 4 0 0 0;\
                       -1 0 1 0 0 -4 0 0 0;\
                       -2 1 0 0 0 -4 0 0 0;\
                       1 0 0 0 0 -4 0 0 0;\
                       0 0 1 2 0 3 0 0 0;\
                       0 0 -2 2 2 -3 0 0 0;\
                       -1 -1 1 0 0 -3 0 0 0;\
                       0 1 1 0 0 -3 0 0 0;\
                       0 -1 1 2 2 -3 0 0 0;\
                       2 -1 -1 2 2 -3 0 0 0;\
                       0 0 3 2 2 -3 0 0 0;\
                       2 -1 0 2 2 -3 0 0 0')


def LnBnR(data, JME):
    nrow = np.shape(data)[0]
    Ln = 0
    for i in range(0, nrow):
        Ln = Ln + data[i,0] * np.cos(data[i,1] + data[i,2] * JME)
    return(Ln)        


def SPA(Day, Month, Year, hour, TZ, lat, long, elev):
    # Month = int(input('Enter the month(Note: if month<3, then month = month + 12, year = year - 1): '))
    # Year = int(input('Enter the year (e.g. 2003): '))
    # Day = int(input('Enter the day of the month: '))
    # hour = int(input('Enter the local time hour: '))
    minute = 0  # int(input('Enter the minute: '))
    second = 0  # int(input('Enter the second: '))
    # TZ = int(input('Enter the time zone: '))
    DT = -20 + 32 * ((Year-1820)/100)**2  # float(input('Enter the time difference between the Earth rotation time and Terrestrial Time: ')) # extrapolated in seconds
    # lat = float(input('Enter the latitude in degrees: '))
    # long = float(input('Enter the longitude in degrees: '))
    P = 700  # float(input('Enter the pressure in mbar: '))
    T = 12 # float(input('Enter the temperature in °C: '))
    # elev = float(input('Enter the elevation in m: '))
    slope = 0  # float(input('Enter the surface slope in °: '))
    ra = 0  # float(input('Enter the surface Azimuth rotation in °: '))
    D = Day + ((hour-TZ)/24) + (minute/(24*60)) + (second/(24*3600)) # the "decimal" version of day
    A = (Year/100)//1
    B = 2 - A + (A/4)//1 
    JD = (365.25*(Year+4716))//1 + 30.6001*(Month+1)//1 + D + B - 1524.5 # computation of Julian Day
    JDE = JD + DT/86400  # Computation of Julian Ephemeris Day
    JC = (JD - 2451545)/36525 # computation of Julian Century
    JCE = (JDE - 2451545)/36525
    JME = JCE/10 # Computation of Julian Millenium
    L = (LnBnR(L0_data, JME) + LnBnR(L1_data, JME)*JME + LnBnR(L2_data, JME)*pow(JME, 2) + LnBnR(L3_data, JME)*pow(JME, 3) + \
    LnBnR(L4_data, JME)*pow(JME, 4) + LnBnR(L5_data, JME)*pow(JME, 5))/1e8
    L = (L*180/np.pi) % 360
    Theta = (L + 180) % 360
    B = (LnBnR(B0_data, JME) + LnBnR(B1_data, JME)*JME)/1e8
    B = B*180/np.pi
    beta = -B
    R = (LnBnR(R0_data, JME) + LnBnR(R1_data, JME)*JME + LnBnR(R2_data, JME)*pow(JME, 2) + LnBnR(R3_data, JME)*pow(JME, 3) + \
    LnBnR(R4_data, JME)*pow(JME, 4))/1e8
    X_0 = (297.85036 + 445267.111480*JCE - 0.0019142*pow(JCE, 2) + pow(JCE, 3)/189474)*np.pi/180
    X_1 = (357.52772 + 35999.050340*JCE - 0.0001603*pow(JCE, 2) - pow(JCE, 3)/300000)*np.pi/180
    X_2 = (134.96298 + 477198.867398*JCE + 0.0086972*pow(JCE, 2) + pow(JCE, 3)/56250)*np.pi/180
    X_3 = (93.27191 + 483202.017538*JCE - 0.0036825*pow(JCE, 2) + pow(JCE, 3)/327270)*np.pi/180
    X_4 = (125.04452 - 1934.136261*JCE + 0.0020708*pow(JCE, 2) + pow(JCE, 3)/450000)*np.pi/180
    dpsi = 0
    deps = 0
    for j in range(0, np.shape(other_data)[0]):
        dpsi = dpsi + (other_data[j, 5] + other_data[j, 6]*JCE)*np.sin(X_0*other_data[j, 0] + X_1*other_data[j, 1] + \
                    X_2*other_data[j, 2] + X_3*other_data[j, 3] + X_4*other_data[j, 4])
    dpsi = dpsi/36000000
    for k in range(0, np.shape(other_data)[0]):
        deps = deps + (other_data[k, 7] + other_data[k, 8]*JCE)*np.cos(X_0*other_data[k, 0] + X_1*other_data[k, 1] + \
                    X_2*other_data[k, 2] + X_3*other_data[k, 3] + X_4*other_data[k, 4])
    deps = deps/36000000
    U = JME/10
    eps_0 = 84381.448 - 4680.93*U - 1.55*pow(U, 2) + 1999.25*pow(U, 3) - 51.38*pow(U, 4) - 249.67*pow(U, 5) - 39.05*pow(U,6) + \
    7.12*pow(U, 7) + 27.87*pow(U, 8) + 5.79*pow(U, 9) + 2.45*pow(U, 10)
    eps = eps_0/3600 + deps
    dtau = -20.4898/(3600*R)
    Lambda = Theta + dpsi + dtau
    nu_0 = (280.46061837 + 360.98564736629*(JD - 2451545) + 0.000387933*pow(JC, 2) - pow(JC, 3)/38710000) % 360
    nu = nu_0 + dpsi*np.cos(eps*np.pi/180)
    alpha = np.arctan2((np.sin(Lambda*np.pi/180)*np.cos(eps*np.pi/180) - np.tan(beta*np.pi/180)*np.sin(eps*np.pi/180)), np.cos(Lambda*np.pi/180))
    alpha = (alpha*180/np.pi) % 360
    delta = (np.arcsin(np.sin(beta*np.pi/180)*np.cos(eps*np.pi/180) + np.cos(beta*np.pi/180)*np.sin(eps*np.pi/180)*np.sin(Lambda*np.pi/180)))*180/np.pi
    H = (nu + long - alpha) % 360
    xi = 8.794/(3600*R)
    u = np.arctan(0.99664719*np.tan(lat*np.pi/180))
    x = np.cos(u) + elev*np.cos(lat*np.pi/180)/6378140
    y = 0.99664719*np.sin(u) + elev*np.sin(lat*np.pi/180)/6378140
    dalpha = (np.arctan2(-x*np.sin(xi*np.pi/180)*np.sin(H*np.pi/180), (np.cos(delta*np.pi/180)-x*np.sin(xi*np.pi/180)*np.cos(H*np.pi/180))))*180/np.pi
    nalpha = alpha + dalpha
    ndelta = np.arctan2((np.sin(delta*np.pi/180) - y*np.sin(xi*np.pi/180))*np.cos(dalpha*np.pi/180), (np.cos(delta*np.pi/180) - x*np.sin(xi*np.pi/180)*np.cos(H*np.pi/180)))*180/np.pi
    nH = H - dalpha
    e_0 = np.arcsin(np.sin(lat*np.pi/180)*np.sin(ndelta*np.pi/180)+np.cos(lat*np.pi/180)*np.cos(ndelta*np.pi/180)*np.cos(nH*np.pi/180))
    e_0 = e_0*180/np.pi
    de = (P/1010)*(283/(273+T))*(1.02/(60*np.tan((e_0 + 10.3/(e_0+5.11)*np.pi/180))))
    e = e_0 + de
    theta = 90 - e
    Gamma = np.arctan2(np.sin(nH*np.pi/180), (np.cos(nH*np.pi/180)*np.sin(lat*np.pi/180)-np.tan(ndelta*np.pi/180)*np.cos(lat*np.pi/180)))
    Gamma = (Gamma*180/np.pi) % 360
    Phi = (Gamma + 180) % 360
    I = np.arccos(np.cos(theta*np.pi/180)*np.cos(slope*np.pi/180)+np.sin(slope*np.pi/180)*np.sin(theta*np.pi/180)*np.cos((Gamma-ra)*np.pi/180))*180/np.pi
    # print("The angle of incidence is " + str(I) + "°")
    return I


def SPA_day(Day,Month,Year,TZ,lat,long,elev):
    I = np.zeros(24)
    for i in range(24):
        I[i] = SPA(Day,Month,Year,i,TZ,lat,long,elev)
    return I


# Computes solar intensity
def solar_intensity(Day=20, Month=5, Year=2018, hour=0, TZ=2, lat=47, long=10, elev=3000):
    incident_angle = SPA(Day, Month, Year, hour, TZ, lat, long, elev)
    if incident_angle <= 90:
        return abs(incident_angle - 90) / 90
    else:
        return 0
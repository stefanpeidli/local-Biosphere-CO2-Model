import numpy as np
import model
import matplotlib.pyplot as plt

# bla
n = 23

# initial parameter values
p = 10
k = 1
c_e = 0.5

# Gradient descent parameters
gd_stepsize = 0.05  # gradient descent step size


# Measurement series
B = np.genfromtxt('data/2016_06_02_co2.csv', delimiter=',', names=True)
B = B['CO2']

iters = 20
losses = []
for j in range(iters):

    # Prediction series produced with current parameter values
    stepsize = 0.1  # Model integration stepsize
    [times, A] = model.evolve(B[0], 0, (2, 6, 2016), 24, stepsize, p, k, c_e, 0, 0)
    A = np.interp(np.arange(n), times, A)  # interpolate to match measurement times
    if j == 0:
        C = A  # save for plot
    # Compute losses (for visualizations and testing purposes)
    loss = 1/n * sum((A-B) ** 2)
    print('Loss in iteration ' + str(j) + " : " + str(loss))
    losses.append(loss)

    I = np.arange(n)  # Intensities for the series

    # Compute gradient of prediction
    gradA = np.zeros((n+1, 3))
    for i in range(n):
        gradf = np.array([-I[i] / (k + I[i]), p * I[i] / (k + I[i]) ** 2, 1])  # format: [dp, dk, dc_e]
        gradA[i+1] = gradA[i] + stepsize * gradf

    # Compute gradient on error surface
    gradL = [0, 0, 0]
    for i in range(n):
        gradL = gradL + 2 * (A[i]-B[i]) * gradA[i]
    gradL = gradL / n

    # Make gradient step
    [p, k, c_e] = [p, k, c_e] - gd_stepsize * gradL
    print('New parameters: p = ' + str(p) + ' , k = ' + str(k) + ' , c_e = ' + str(c_e))

plt.plot(np.arange(iters), losses)
plt.show()
plt.plot(np.arange(n), A)
plt.plot(np.arange(n), B)
plt.plot(np.arange(n), C)
plt.show()

import numpy as np
import model
import matplotlib.pyplot as plt

# defining a time-series piece
n = 24

# initial parameter values
p = 10
k = 1
c_e = 0.5

# Gradient descent parameters
gd_stepsize = 0.05  # gradient descent step size

# Load Measurement series (e.g. '2016_06_02_co2.csv')
data = np.genfromtxt('data/data_small_ordered.csv', delimiter=',', names=True)
# print(data)
Bs = data['CO2']
day_count = int(len(Bs)/24)
for i in range(len(Bs)):
    if Bs[i] != Bs[i]:
        Bs[i] = Bs[i-1] + (Bs[i-1] - Bs[i-2])  # some heuristics/ black magic / linear interpolation
Bs = np.reshape(Bs, newshape=(day_count, 24))


iters = 5
losses = []
As = np.zeros(shape=Bs.shape)
Cs = np.zeros(shape=Bs.shape)
for j in range(iters):
    loss = 0
    for day in range(len(Bs)):
        B = Bs[day]
        # Prediction series produced with current parameter values
        stepsize = 0.1  # Model integration stepsize
        [model_times, A] = model.evolve(B[0], 0, (2, 6, 2016), 24, stepsize, p, k, c_e, 0, 0)
        A = np.interp(np.arange(n), model_times, A)  # interpolate to match measurement times
        As[day] = A

        if j == 0:
            Cs[day] = A  # save for plot
        # Compute losses (for visualizations and testing purposes)
        loss = loss + 1/n * sum((A-B) ** 2)
        #losses.append(loss)

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
    losses.append(loss / day_count)
    print('Loss in iteration ' + str(j) + " : " + str(losses[j]))

# final (20iters, eta=0.05) New parameters: p = 6.08670749111 , k = 7.83966596653 , c_e = 0.333788943736
plt.semilogy(np.arange(iters), losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks(np.arange(iters))
plt.title('Total Loss by epoch')
plt.show()

plt.plot(np.arange(len(As[0])), As[0])
plt.plot(np.arange(len(Bs[0])), Bs[0])
plt.plot(np.arange(len(Cs[0])), Cs[0])
plt.xlabel('Time (hours)')
plt.ylabel('CO2 - Concentration')
plt.title('Results for single Day')
plt.legend(['Prediction final', 'Measurement', 'Prediction initial'])
plt.show()

"""
plt.plot(np.arange(day_count * n), np.reshape(As, day_count * n))
plt.plot(np.arange(day_count * n), np.reshape(Bs, day_count * n))
plt.plot(np.arange(day_count * n), np.reshape(Cs, day_count * n))
plt.xlabel('Time (hours)')
plt.ylabel('CO2 - Concentration')
plt.title('Parameter Optimization')
plt.legend(['Prediction final', 'Measurement', 'Prediction initial'])
for day in range(day_count-1):
    plt.axvline((day+1) * n, linestyle='--', c='r', alpha=0.6)
    plt.text(day * n + n/2, 470, 'day ' + str(day), horizontalalignment='center')
plt.show()
"""
ka = 1
for i in [2,8,9,10,12,13]:
    plt.subplot(2,3,ka)
    plt.plot(np.arange(n), np.reshape(As[i], n))
    plt.plot(np.arange(n), np.reshape(Bs[i], n))
    plt.plot(np.arange(n), np.reshape(Cs[i], n))
    plt.xlabel('Time (hours)')
    plt.ylabel('CO2 - Concentration')
    plt.suptitle('Parameter Optimization', fontsize=16)
    plt.legend(['Prediction final', 'Measurement', 'Prediction initial'])
    plt.text(24/2, 470, 'day ' + str(i+1), horizontalalignment='center')
    ka = ka + 1
plt.show()
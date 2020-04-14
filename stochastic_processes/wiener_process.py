from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def wiener_process_from_def(delta=0.025, n=100):
    '''
    W(0) = W_0=0
    W(t+dt) = W(t) + N(0, delta**2*dt)

    :param delta:
    :param n:
    :return:
    '''


    dt = 0.01
    w = np.zeros(n)
    w[0] = 0
    for i in range(1, n):
        w[i] = w[i-1] + norm.rvs(delta**2*dt)
    return w

def wiener_process_from_fourier_series(x=10.0):
    def fourier_series(t, n=100):
        '''
        W_t = zeta_0 * t + sqrt(2) * sum(zeta_n * sin(pi *n * t)/(pi*n), 1, inf)
        :param t:
        :param n:
        :return:
        '''
        s = 0
        for i in range(1, n):
            zeta = norm.rvs()
            s += zeta * np.sin(np.pi * i * t)/(np.pi * i)

        w_t = norm.rvs() * t + np.sqrt(2) * s
        return w_t

    dt = 0.01
    w = [fourier_series(t, n=100) for t in np.arange(0.0, x, dt)]
    return w


num_samples = 10
num_steps = 10000

for i in range(num_samples):
    w = wiener_process_from_def(n=10000, delta=1)
    #w = wiener_process_from_fourier_series()
    plt.plot(w)

#plt.plot(np.sqrt(np.arange(0, num_steps)), color=[0,0,0])
#plt.plot(-np.sqrt(np.arange(0, num_steps)), color=[0,0,0])
plt.show()
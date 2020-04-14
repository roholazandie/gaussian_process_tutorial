from scipy.stats import randint
import numpy as np
import matplotlib.pyplot as plt


def random_walk(n=1000):
    x = np.zeros(n)
    for i in range(1, n):
        toss = randint(0, 2).rvs()
        if toss:
            x[i] = x[i-1] - 1
        else:
            x[i] = x[i-1] + 1

    return x

num_samples = 10
num_steps = 100
for i in range(num_samples):
    x = random_walk(n=num_steps)
    plt.plot(x, lw = 1)

#plt.plot(np.sqrt(np.arange(0, num_steps)))
#plt.plot(-np.sqrt(np.arange(0, num_steps)))
plt.show()
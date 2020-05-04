from scipy.special import hermite, eval_hermite
import matplotlib.pyplot as plt
import numpy as np


for k in range(1, 10):
    x = np.linspace(-1, 1, 100)
    h = hermite(k)
    plt.plot(x, h(x), label=str(k))
plt.show()
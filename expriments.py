from GPy.kern import RBF, Brownian, Cosine
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA




kernel = RBF(input_dim=1, variance=2.0)
#kernel = Brownian(input_dim=1, variance=2.0)
#kernel = Cosine(input_dim=1, variance=2.0)


values = []
r = range(10, 300)
for n_dims in r:
    x = np.linspace(0, 10, n_dims)[:, np.newaxis]
    k = kernel.K(x, x)
    eigenvalues, eigenvectors = eig(k)
    first_eigenvalue = np.max(np.abs(eigenvalues))
    values.append(first_eigenvalue/n_dims)

plt.plot(list(r), values)
plt.show()

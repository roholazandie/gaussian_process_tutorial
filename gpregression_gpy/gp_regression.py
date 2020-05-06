from GPy.kern import RBF, Brownian, PeriodicExponential, Cosine, Exponential, Integral, Matern32
from GPy.models import GPRegression
import numpy as np
from from_scratch.kernels import kernel, rbf
import matplotlib.pyplot as plt

X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)
# Mean and covariance of the prior
mu = np.zeros(X_test.shape) # we set the mean to zero without loss of generality

cov = kernel(X_test, X_test, kernel_func=rbf)
# Draw three samples from the prior
Y_test = np.random.multivariate_normal(mu.ravel(), cov, 3)

noise = 0.1
####################
# Noise free training data
X_train = np.array([-4, -3, -2, -1, 3]).reshape(-1, 1)
Y_train = np.sin(X_train)


rbf = RBF(input_dim=1, variance=1.0, lengthscale=1.0)
brownian = Brownian(input_dim=1, variance=1.0)
periodic = PeriodicExponential(input_dim=1, variance=2.0, n_freq=100)
cosine = Cosine(input_dim=1, variance=2)
exponential = Exponential(input_dim=1, variance=2.0)
integral = Integral(input_dim=1, variances=2.0)
matern = Matern32(input_dim=1, variance=2.0)



gpr = GPRegression(X_train, Y_train, matern)

# Fix the noise variance to known value
gpr.Gaussian_noise.variance = noise**2
gpr.Gaussian_noise.variance.fix()

# Run optimization
ret = gpr.optimize()
print(ret)
# Display optimized parameter values
print(gpr)

# Obtain optimized kernel parameters
#l = gpr.rbf.lengthscale.values[0]
#sigma_f = np.sqrt(gpr.rbf.variance.values[0])


# Plot the results with the built-in plot function
gpr.plot()

plt.show()
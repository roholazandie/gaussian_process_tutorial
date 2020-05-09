import numpy as np
from GPy.kern import Matern32, GridRBF, RBF
from GPy.models import GPRegression
import pods
import matplotlib.pyplot as plt



# read data
from visualization import plot_gp

data = pods.datasets.olympic_marathon_men()
x_train = data["X"]
y_train = data["Y"]

# choose a kernel
#kernel = Matern32(input_dim=1, variance=2.0)
#kernel = GridRBF(input_dim=1)
#kernel = RBF(input_dim=1, variance=2.0)


# gp regression and optimize the paramters using logliklihood
gp_regression = GPRegression(x_train, y_train)

#gp_regression.kern.lengthscale = 500
#gp_regression.likelihood.variance = 0.001

print("loglikelihood: ", gp_regression.log_likelihood())

gp_regression.optimize()

print("loglikelihood: ", gp_regression.log_likelihood())


# predict new unseen samples
x_test = np.linspace(1870, 2030, 200)[:, np.newaxis]
yt_mean, yt_var = gp_regression.predict(x_test)
yt_sd = np.sqrt(yt_var)

# draw some samples from the posterior
samples = gp_regression.posterior_samples(x_test, size=1).squeeze(1)

# plot
plot_gp(yt_mean, yt_var, x_test, X_train=x_train, Y_train=y_train, samples=samples)

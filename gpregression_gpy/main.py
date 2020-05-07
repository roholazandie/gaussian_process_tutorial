import numpy as np
from GPy.kern import Matern32, GridRBF
from GPy.models import GPRegression
import pods
import matplotlib.pyplot as plt



# read data
from visualization import plot_gp

data = pods.datasets.olympic_marathon_men()
x = data["X"]
y = data["Y"]

# choose a kernel
kernel = Matern32(input_dim=1, variance=2.0)
#kernel = GridRBF(input_dim=1)


# gp regression and optimize the paramters using logliklihood
gp_regression = GPRegression(x, y, kernel=kernel)

#gp_regression.kern.lengthscale = 500
#gp_regression.likelihood.variance = 0.001

print("loglikelihood: ", gp_regression.log_likelihood())

gp_regression.optimize()

print("loglikelihood: ", gp_regression.log_likelihood())


# predict new unseen samples
xt = np.linspace(1870, 2030, 200)[:, np.newaxis]
yt_mean, yt_var = gp_regression.predict(xt)
yt_sd = np.sqrt(yt_var)

# draw some samples from the posterior
samples = gp_regression.posterior_samples(xt, size=1).squeeze(1)

# plot
plot_gp(yt_mean, yt_var, xt, X_train=x, Y_train=y, samples=samples)

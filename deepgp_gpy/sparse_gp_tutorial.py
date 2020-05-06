import numpy as np
import GPy
from GPy.kern import RBF
from GPy.models import GPRegression, SparseGPRegression
import matplotlib.pyplot as plt
from visualization import plot_gp, model_output

N = 50
noise_var = 0.01
X = np.zeros((N, 1))
x_half = int(N/2)
X[:x_half, :] = np.linspace(0, 2, x_half)[:, None] # First cluster of inputs/covariates
X[x_half:, :] = np.linspace(8, 10, x_half)[:, None] # Second cluster of inputs/covariates

rbf = RBF(input_dim=1)
mu = np.zeros(N)
cov = rbf.K(X) + np.eye(N)*np.sqrt(noise_var)
y = np.random.multivariate_normal(mu, cov).reshape(-1, 1)

# plt.scatter(X, y)
# plt.show()

gp_regression = GPRegression(X, y)
gp_regression.optimize(messages=True)
log_likelihood1 = gp_regression.log_likelihood()

model_output(gp_regression, title="GP Regression with loglikelihood: "+str(log_likelihood1))

#################################
# inducing variables, u. Each inducing variable has its own associated input index, Z, which lives in the same space as X.
Z = np.hstack(
        (np.linspace(2.5, 4., 3),
        np.linspace(7, 8.5, 3)))[:, None]

sparse_regression = SparseGPRegression(X, y, kernel=rbf, Z=Z)

sparse_regression.noise_var = noise_var
sparse_regression.inducing_inputs.constrain_fixed()

sparse_regression.optimize(messages=True)

log_likelihood2 = sparse_regression.log_likelihood()
#inducing variables fixed
model_output(sparse_regression, title="Inducing variables fixed parameters optimized with loglikelihood: "+str(log_likelihood2[0][0]))

#inducing variables optimized
sparse_regression.inducing_inputs.unconstrain()
sparse_regression.optimize()

log_likelihood3 = sparse_regression.log_likelihood()

model_output(sparse_regression, title="inducing variables and parameters optimized with loglikelihood: "+str(log_likelihood3[0][0]))
#change the number of inducing variables
n_inducing = 8
sparse_regression.num_inducing = n_inducing
sparse_regression.randomize()
sparse_regression.set_Z(np.random.rand(n_inducing, 1)*10) #multiply by 10 to speared on the x space
sparse_regression.optimize()
log_likelihood4 = sparse_regression.log_likelihood()
model_output(sparse_regression, title="eight inducing variables with loglikelihood: "+str(log_likelihood4[0][0]))




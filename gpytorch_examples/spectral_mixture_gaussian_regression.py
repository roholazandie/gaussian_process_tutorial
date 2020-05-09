import torch
from gpytorch import models
from gpytorch import means, kernels, distributions, likelihoods, mlls, settings
import numpy as np
import pods

from visualization import plot_gp

class SpectralMixtureGPModel(models.ExactGP):

    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean = means.ConstantMean()
        self.covariance = kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covariance.initialize_from_data(x_train, y_train)

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covariance(x)
        return distributions.MultivariateNormal(mean_x, covar_x)



x_train = torch.linspace(0, 1, 15)
y_train = torch.sin(x_train * (2 * np.pi))

likelihood = likelihoods.GaussianLikelihood()

model = SpectralMixtureGPModel(x_train, y_train, likelihood)

# find the optimal model hyperparamters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
marginal_loglikelihood = mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iterations = 2000
for i in range(n_iterations):
    optimizer.zero_grad()

    output = model(x_train)

    loss = -marginal_loglikelihood(output, y_train)

    loss.backward()

    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iterations, loss.item()))

    optimizer.step()


#The spectral mixture kernel is especially good at extrapolation.
# To that end, we'll see how well the model extrapolates past the interval [0, 1].


# Test points every 0.1 between 0 and 5
x_test = torch.linspace(0, 5, 51)

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

import matplotlib.pyplot as plt

with torch.no_grad(), settings.fast_pred_var():
    # Make predictions
    y_pred = likelihood(model(x_test))

    mean = y_pred.mean.numpy()
    var = y_pred.variance.numpy()*1e3
    plot_gp(mean, var, x_test.numpy(), X_train=x_train.numpy(), Y_train=y_train.numpy())


    # # Initialize plot
    # f, ax = plt.subplots(1, 1, figsize=(4, 3))
    #
    # # Get upper and lower confidence bounds
    # lower, upper = observed_pred.confidence_region()
    # # Plot training data as black stars
    # ax.plot(x_train.numpy(), y_train.numpy(), 'k*')
    # # Plot predictive means as blue line
    # ax.plot(x_test.numpy(), observed_pred.mean.numpy(), 'b')
    # # Shade between the lower and upper confidence bounds
    # ax.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    # ax.legend(['Observed Data', 'Mean', 'Confidence'])
    #
    # plt.show()









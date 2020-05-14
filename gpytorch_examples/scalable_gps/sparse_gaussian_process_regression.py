import numpy as np
import torch
import gpytorch
import pods
from gpytorch import models
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import settings

from visualization import plot_gp


class SparseGaussianProcessRegressionModel(models.ExactGP):

    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean = ConstantMean()
        base_kernel = ScaleKernel(RBFKernel())
        # self.covariance = base_kernel
        # here we chose inducing points very randomly just based on the first five
        # samples of training data but it can be much better or smarter
        self.covariance = InducingPointKernel(base_kernel,
                                              inducing_points=x_train[:5],
                                              likelihood=likelihood)

    def forward(self, x):
        mean = self.mean(x)
        covar = self.covariance(x)
        return MultivariateNormal(mean, covar)


data = pods.datasets.olympic_marathon_men()
x_train = torch.from_numpy(data["X"]).squeeze(-1)
y_train = torch.from_numpy(data["Y"]).squeeze(-1)

likelihood = GaussianLikelihood()
model = SparseGaussianProcessRegressionModel(x_train, y_train, likelihood)

if torch.cuda.is_available():
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    model = model.cuda()
    likelihood = likelihood.cuda()

training_iterations = 400

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)


# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    # Zero backprop gradients
    optimizer.zero_grad()
    # Get output from model
    output = model(x_train)
    # Calc loss and backprop derivatives
    loss = -mll(output, y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
    torch.cuda.empty_cache()

model.eval()
likelihood.eval()

x_test = torch.from_numpy(np.linspace(1870, 2030, 200)[:, np.newaxis])
x_test = x_test.cuda()

with settings.max_preconditioner_size(10), torch.no_grad():
    with settings.max_root_decomposition_size(30), settings.fast_pred_var():
        f_preds = model(x_test)
        y_pred = likelihood(f_preds)

# plot
with torch.no_grad():
    mean = y_pred.mean.cpu().numpy()
    var = y_pred.variance.cpu().numpy()
    samples = y_pred.sample().cpu().numpy()
    plot_gp(mean, var, x_test.cpu().numpy(), X_train=x_train.cpu().numpy(), Y_train=y_train.cpu().numpy(), samples=samples)

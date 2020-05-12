import torch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import fast_pred_var, max_root_decomposition_size

import numpy as np
import pods
import time
from visualization import plot_gp


class ExactGPModel(ExactGP):

    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


data = pods.datasets.olympic_marathon_men()
x_train = torch.from_numpy(data["X"]).squeeze(-1)
y_train = torch.from_numpy(data["Y"]).squeeze(-1)# + torch.randn(train_x.size()) * np.sqrt(0.04)

likelihood = GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood)

x_train = x_train.cuda()
y_train = y_train.cuda()
model = model.cuda()
likelihood = likelihood.cuda()


model.train()
likelihood.train()

optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)

##loss for gp
marginal_loglikelihood = ExactMarginalLogLikelihood(likelihood, model)

training_iter = 2000
for i in range(training_iter):
    optimizer.zero_grad()

    output = model(x_train)

    loss = -marginal_loglikelihood(output, y_train) # this gives the marginal loglikelihood  log(p(y|X))
    loss.backward()

    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()))

    optimizer.step()


model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
# LOVE: fast_pred_var is used for faster computation of predictive posterior
# https://arxiv.org/pdf/1803.06058.pdf
# This can be especially useful in settings like small-scale Bayesian optimization,
# where predictions need to be made at enormous numbers of candidate points,
# but there aren't enough training examples to necessarily warrant the use of sparse GP methods
# max_root_decomposition_size(35) affects the accuracy of the LOVE solves (larger is more accurate, but slower
t1 = time.time()
with torch.no_grad(), fast_pred_var(), max_root_decomposition_size(25):
    x_test = torch.from_numpy(np.linspace(1870, 2030, 200)[:, np.newaxis])
    x_test = x_test.cuda()
    f_preds = model(x_test) #f_preds gives us the mean and cov from a distribution that can be used inside liklihood
    y_pred = likelihood(f_preds)

t2 = time.time()
print(t2-t1)


# plot
with torch.no_grad():
    mean = y_pred.mean.cpu().numpy()
    var = y_pred.variance.cpu().numpy()
    samples = y_pred.sample().cpu().numpy()
    plot_gp(mean, var, x_test.cpu().numpy(), X_train=x_train.cpu().numpy(), Y_train=y_train.cpu().numpy(), samples=samples)

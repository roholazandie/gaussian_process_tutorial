import math
import torch
import gpytorch
from urllib import request
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.io import loadmat
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, MeanFieldVariationalDistribution, \
    DeltaVariationalDistribution
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
import matplotlib.pyplot as plt
from gpytorch_examples.data_utils import get_data


'''
variational_distribution (or q) is the approximation of p (the GP prior ) becaue we can't calculate p directly.
So we want a q* = argmin KL(q||p) but this is again directly hard to compute, then we can derive ELBO
ELBO = KL(q_z||p_z) - likelihood 
(in which z are inducing points) maximizing ELBO is like minimizing the KL(q||p). and
likelihood =  E_q(log(P(Y|f_d))
the only thing that remains is that to maximize this (or minimize the negative of ELBO) by taking gradients using torch
just like any other loss function we ever had!

inducing points (Z's) are not new paramters they are like psudo data and choosing them in the right way
help us to save a lot of computation. That's why using inducing points are called SparseGP
In the most naive scenario we choose Z's randomly but a big portion of researches in GP is around
finding the best way for Z's.
'''

class ApproxiateGaussianProcessModel(ApproximateGP):

    def __init__(self, inducting_points):
        '''
        As a default, we'll use the default VariationalStrategy class with a CholeskyVariationalDistribution.
        The CholeskyVariationalDistribution class allows S to be on any positive semidefinite matrix.
        This is the most general/expressive option for approximate GPs
        '''
        variational_distribution = CholeskyVariationalDistribution(inducting_points.size(-2))
        variational_strategy = VariationalStrategy(self, inducting_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean = ConstantMean()
        self.covar = ScaleKernel(RBFKernel())

    def forward(self, x):
        x_mean = self.mean(x)
        x_covar = self.covar(x)
        return MultivariateNormal(x_mean, x_covar)


class MeanFieldApproximateGaussianProcessModel(ApproximateGP):
    def __init__(self, inducting_points):
        '''
        One way to reduce the number of parameters is to restrict that $\mathbf S$ is only diagonal.
        This is less expressive, but the number of parameters is now linear in $m$ instead of quadratic.
        All we have to do is take the previous example, and change CholeskyVariationalDistribution S
         to MeanFieldVariationalDistribution S.
        '''
        variational_distribution = MeanFieldVariationalDistribution(inducting_points.size(-2))
        variational_strategy = VariationalStrategy(self, inducting_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean = ConstantMean()
        self.covar = ScaleKernel(RBFKernel())

    def forward(self, x):
        x_mean = self.mean(x)
        x_covar = self.covar(x)
        return MultivariateNormal(x_mean, x_covar)


class MAPApproximateGaussianProcessModel(ApproximateGP):
    def __init__(self, inducting_points):
        '''
        A more extreme method of reducing parameters is to get rid of S entirely.
        This corresponds to learning a delta distribution u=m rather than a multivariate Normal distribution
        for u. In other words, this corresponds to performing MAP estimation rather than variational inference.
        '''
        variational_distribution = DeltaVariationalDistribution(inducting_points.size(-2))
        variational_strategy = VariationalStrategy(self, inducting_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean = ConstantMean()
        self.covar = ScaleKernel(RBFKernel())

    def forward(self, x):
        x_mean = self.mean(x)
        x_covar = self.covar(x)
        return MultivariateNormal(x_mean, x_covar)


'''
Approximate Gaussian processes learn an approximate posterior distribution p(f(X)|y) by an easy to compute q(f): 
q(u) is usually a gaussian with N(m, S) that we should approximate m and S
'''

train_loader, test_loader, x_train, x_test, y_test = get_data()
n_train = x_train.shape[0]

# method = "variationalelbo"
method = "predictiveloglikelihood"
if method == "variationalelbo":
    '''
    The variational evidence lower bound - or ELBO - is the most common objective function. 
    It can be derived as an lower bound on the likelihood p(y|X)
    
    \mathcal{L}_\text{ELBO} &=
  \mathbb{E}_{p_\text{data}( y, \mathbf x )} \left[
    \log \mathbb{E}_{q(\mathbf u)} \left[  p( y \! \mid \! \mathbf u) \right]
  \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
  \\
  &\approx \sum_{i=1}^N \mathbb{E}_{q( \mathbf u)} \left[
    \log \int p( y_i \! \mid \! f_i) p(f_i \! \mid \! \mathbf u) \: d \mathbf f_i
  \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
    
    '''
    objective_function_cls = VariationalELBO
else:
    objective_function_cls = PredictiveLogLikelihood

# inducing_points = torch.linspace(0, 1)
inducing_points = torch.randn(128, x_train.size(-1), dtype=x_train.dtype, device=x_train.device)

# model_name = "approximate_gaussian"
# model_name = "meanfield"
model_name = "map_approximate"

if model_name.lower() == "approximate_gaussian":
    model = ApproxiateGaussianProcessModel(inducing_points)
elif model_name.lower() == "meanfield":
    model = MeanFieldApproximateGaussianProcessModel(inducing_points)
elif model_name.lower() == "map_approximate":
    model = MAPApproximateGaussianProcessModel(inducing_points)
else:
    raise ValueError(f"Unknown model_name {model_name}")

likelihood = GaussianLikelihood()
objective_function = objective_function_cls(likelihood, model, n_train)
optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)

model = model.cuda()
likelihood = likelihood.cuda()

model.train()
likelihood.train()

n_epochs = 10
for i in range(n_epochs):

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -objective_function(output, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Iter {i} - Loss: {loss.item()}')

# Test
model.eval()
likelihood.eval()
means = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])

means = means[1:]
error = torch.mean(torch.abs(means - y_test.cpu()))
print(f" MAE: {error.item()}")

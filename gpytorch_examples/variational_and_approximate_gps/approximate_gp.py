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


class ApproxiateGaussianProcessModel(ApproximateGP):

    def __init__(self, inducting_points):
        '''
        As a default, we'll use the default VariationalStrategy class with a CholeskyVariationalDistribution.
        The CholeskyVariationalDistribution class allows S to be on any positive semidefinite matrix.
        This is the most general/expressive option for approximate GPs
        '''
        variational_distribution = CholeskyVariationalDistribution(inducting_points.size(-2))
        variational_strategy = VariationalStrategy(self, inducting_points, variational_distribution, learn_inducing_locations=True)
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
        As a default, we'll use the default VariationalStrategy class with a CholeskyVariationalDistribution.
        The CholeskyVariationalDistribution class allows S to be on any positive semidefinite matrix.
        This is the most general/expressive option for approximate GPs
        '''
        variational_distribution = MeanFieldVariationalDistribution(inducting_points.size(-2))
        variational_strategy = VariationalStrategy(self, inducting_points, variational_distribution, learn_inducing_locations=True)
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
        variational_strategy = VariationalStrategy(self, inducting_points, variational_distribution, learn_inducing_locations=True)
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

# x_train = torch.linspace(0, 1, 100)
# y_train = torch.cos(x_train * 2 * math.pi) + torch.randn(100).mul(x_train.pow(3) * 1.)
#
# fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.scatter(x_train, y_train, c='k', marker='.', label="Data")
# ax.set(xlabel="x", ylabel="y")
# plt.show()

def get_data():
    if not os.path.isfile("elevators.mat"):
        request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', 'elevators.mat')

    data = torch.Tensor(loadmat('elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]

    n_train = int(np.floor(0.8 * len(X)))
    x_train = X[:n_train, :].contiguous()
    y_train = y[:n_train].contiguous()

    x_test = X[n_train:, :].contiguous()
    y_test = y[n_train:].contiguous()

    # if torch.cuda.is_available():
    #     x_train, y_train, x_test, y_test = x_train.cuda(), y_train.cuda(), x_test.cuda(), y_test.cuda()


    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    return train_loader, test_loader, x_train, y_test, y_train.numel()



train_loader, test_loader, x_train, y_test, n_train = get_data()


#method = "variationalelbo"
method = "predictiveloglikelihood"
if method == "variationalelbo":
    '''
    The variational evidence lower bound - or ELBO - is the most common objective function. 
    It can be derived as an lower bound on the likelihood p(y|X)
    '''
    objective_function_cls = VariationalELBO
else:
    objective_function_cls = PredictiveLogLikelihood


#inducing_points = torch.linspace(0, 1)
inducing_points = torch.randn(128, x_train.size(-1), dtype=x_train.dtype, device=x_train.device)

#model_name = "approximate_gaussian"
#model_name = "meanfield"
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

# Plot model
# fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# line, = ax.plot(x_train, mean, "blue")
# ax.fill_between(x_train, f_lower, f_upper, color=line.get_color(), alpha=0.3, label="q(f)")
# ax.fill_between(x_train, y_lower, y_upper, color=line.get_color(), alpha=0.1, label="p(y)")
# ax.scatter(x_train, y_train, c='k', marker='.', label="Data")
# ax.legend(loc="best")
# ax.set(xlabel="x", ylabel="y")
# plt.show()
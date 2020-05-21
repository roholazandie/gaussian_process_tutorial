import math

from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import use_toeplitz, fast_pred_var
from tqdm import tqdm
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import GridInterpolationKernel, ScaleKernel, RBFKernel
from matplotlib import pyplot as plt

from gpytorch_examples.data_utils import get_elevators_data


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.linear1 = torch.nn.Linear(data_dim, 1000)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1000, 500)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(500, 50)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(50, 2)

    def forward(self, x):
        x1 = self.relu1(self.linear1(x))
        x2 = self.relu2(self.linear2(x1))
        x3 = self.relu3(self.linear3(x2))
        return self.linear4(x3)


class GPRegressionModel(ExactGP):
    def __init__(self, x_train, y_train, likelihood, feature_extractor):
        super(GPRegressionModel, self).__init__(x_train, y_train, likelihood)
        self.mean = ConstantMean()
        self.covar = GridInterpolationKernel(ScaleKernel(RBFKernel(ard_num_dims=2)),
                                             num_dims=2, grid_size=100
                                             )
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        x_mean = self.mean(projected_x)
        x_covar = self.covar(projected_x)
        return MultivariateNormal(x_mean, x_covar)


train_loader, test_loader, x_train, x_test, y_test = get_elevators_data()
y_train = train_loader.dataset.tensors[1]

likelihood = GaussianLikelihood()
feature_extractor = LargeFeatureExtractor(data_dim=x_train.size(-1))
model = GPRegressionModel(x_train, y_train, likelihood, feature_extractor)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar.parameters()},
    {'params': model.mean.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 60


def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(x_train)
        # Calc loss and backprop derivatives
        loss = -mll(output, y_train)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

train()

model.eval()
likelihood.eval()
with torch.no_grad(), use_toeplitz(False), fast_pred_var():
    preds = model(x_test)
    print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - y_test))))
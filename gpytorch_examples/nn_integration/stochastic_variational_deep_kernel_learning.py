from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.settings import use_toeplitz, num_likelihood_samples
from gpytorch.utils.grid import scale_to_bounds
from gpytorch.variational import CholeskyVariationalDistribution, MultitaskVariationalStrategy, \
    GridInterpolationVariationalStrategy
from gpytorch import Module
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import math
from tqdm import tqdm
import numpy as np

from gpytorch_examples.data_utils import get_vision_data
from gpytorch_examples.nn_integration.densenet import DenseNet


class DenseNetFeatureExtractor(DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        return out


class GaussianProcessLayer(ApproximateGP):

    def __init__(self, num_dim, grid_bounds=(-10.0, 10.0), grid_size=64):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=grid_size,
                                                                   batch_shape=torch.Size([num_dim]))

        base_strategy = GridInterpolationVariationalStrategy(self,
                                                             grid_size=grid_size,
                                                             grid_bounds=[grid_bounds],
                                                             variational_distribution=variational_distribution)

        variational_strategy = MultitaskVariationalStrategy(base_strategy, num_tasks=num_dim)
        super().__init__(variational_strategy)

        self.covar = ScaleKernel(
            RBFKernel(lengthscale_prior=SmoothedBoxPrior(math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)))
        self.mean = ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean(x)
        covar = self.covar(x)
        return MultivariateNormal(mean, covar)


class DeepKernelLearningModel(Module):

    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10.0, 10.0)):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim, grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = scale_to_bounds(features, lower_bound=self.grid_bounds[0], upper_bound=self.grid_bounds[1])
        features = features.transpose(-1, -2).unsqueeze(-1)
        result = self.gp_layer(features)
        return result


train_loader, test_loader, num_classes = get_vision_data()

feature_extractor = DenseNetFeatureExtractor(block_config=(6, 6, 6), num_classes=num_classes)
num_features = feature_extractor.classifier.in_features

model = DeepKernelLearningModel(feature_extractor, num_dim=num_features)
likelihood = SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

n_epochs = 5
lr = 0.1

optimizer = SGD([{'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
                 {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
                 {'params': model.variational_parameters()},
                 {'params': likelihood.parameters()}],
                lr=lr, momentum=0.9, nesterov=True, weight_decay=0)

scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

mll = VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))


def train(epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())


def test():
    model.eval()
    likelihood.eval()

    correct = 0
    with torch.no_grad(), num_likelihood_samples(16):
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))


for epoch in range(1, n_epochs + 1):
    with use_toeplitz(False):
        train(epoch)
        test()
    scheduler.step()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')

import math
import torch
import gpytorch
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, UnwhitenedVariationalStrategy, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.settings import num_likelihood_samples
from gpytorch_examples.data_utils import get_data
from matplotlib import pyplot as plt


class DeepGaussianProcessHiddenLayer(DeepGPLayer):

    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type="constant"):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing,
                                                                   batch_shape=batch_shape)

        variational_strategy = VariationalStrategy(self,
                                                   inducing_points,
                                                   variational_distribution,
                                                   learn_inducing_locations=True)

        super().__init__(variational_strategy, input_dims=input_dims, output_dims=output_dims)

        if mean_type == "constant":
            self.mean = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean = LinearMean(input_dims)

        self.covar = ScaleKernel(RBFKernel(ard_num_dims=input_dims, batch_shape=batch_shape),
                                 batch_shape=batch_shape,
                                 ard_num_dims=None)

    def forward(self, x):
        x_mean = self.mean(x)
        x_covar = self.covar(x)
        return MultivariateNormal(x_mean, x_covar)

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


num_output_dims = 10


class DeepGaussianProcess(DeepGP):

    def __init__(self, x_train_shape):
        hidden_layer = DeepGaussianProcessHiddenLayer(input_dims=x_train_shape[-1],
                                                      output_dims=num_output_dims,
                                                      mean_type="linear")

        output_layer = DeepGaussianProcessHiddenLayer(input_dims=hidden_layer.output_dims,
                                                      output_dims=None,
                                                      mean_type="constant")

        super().__init__()

        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        h1 = self.hidden_layer(x)
        output = self.output_layer(h1)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            loglikelihoods = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self.forward(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                loglikelihoods.append(self.likelihood.log_marginal(y_batch, self.forward(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(loglikelihoods, dim=-1)


train_loader, test_loader, x_train, x_test, y_test = get_data()

model = DeepGaussianProcess(x_train_shape=x_train.shape)
if torch.cuda.is_available():
    model = model.cuda()

# Because deep GPs use some amounts of internal sampling (even in the stochastic variational setting),
# we need to handle the objective function (e.g. the ELBO) in a slightly different way.
num_samples = 10

optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

'''
DeepApproximateMLL only adds the elbo losses of each layer!
'''

marginal_loglikelihood = DeepApproximateMLL(VariationalELBO(model.likelihood, model, x_train.shape[-2]))

n_epochs = 100
for i in range(n_epochs):

    for x_batch, y_batch in train_loader:
        with num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -marginal_loglikelihood(output, y_batch)
            loss.backward()
            optimizer.step()

    print(f"epochs {i}, loss {loss.item()}")

## test and evaluate the model

model.eval()
predictive_means, predictive_variances, test_loglikelihoods = model.predict(test_loader)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - y_test, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_loglikelihoods.mean().item()}")

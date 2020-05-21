from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood, SoftmaxLikelihood
import torch
import numpy as np
from gpytorch.settings import num_likelihood_samples
from torch.distributions import MultivariateNormal

#likelihood = GaussianLikelihood()
#likelihood = BernoulliLikelihood()
likelihood = SoftmaxLikelihood(num_classes=5, num_features=2)


observations = torch.from_numpy(np.array([1.0, 1.0])).type(torch.float32)
mean = torch.from_numpy(np.array([[1.0], [2.0]])).type(torch.float32)
covar = torch.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]])).type(torch.float32)
multivariate_normal = MultivariateNormal(mean, covar)
with num_likelihood_samples(8000):
    explog = likelihood.expected_log_prob(observations, multivariate_normal)

print(explog)
import torch
from gpytorch import models
from gpytorch import means, kernels, distributions, likelihoods, mlls, settings
import numpy as np
import pods

from visualization import plot_gp


class ExactGPModel(models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)


data = pods.datasets.olympic_marathon_men()
train_x = torch.from_numpy(data["X"]).squeeze(-1)
train_y = torch.from_numpy(data["Y"]).squeeze(-1)# + torch.randn(train_x.size()) * np.sqrt(0.04)

likelihood = likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)

##loss for gp
mll = mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 700
for i in range(training_iter):
    optimizer.zero_grad()

    output = model(train_x)

    loss = -mll(output, train_y) # this gives the marginal loglikelihood  log(p(y|X))
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
with torch.no_grad(), settings.fast_pred_var():
    test_x = torch.from_numpy(np.linspace(1870, 2030, 200)[:, np.newaxis])
    f_preds = model(test_x)
    y_pred = likelihood(f_preds)



# plot
with torch.no_grad():
    mean = y_pred.mean.numpy()
    var = y_pred.variance.numpy()
    samples = y_pred.sample().numpy()
    plot_gp(mean, var, test_x.numpy(), X_train=train_x.numpy(), Y_train=train_y.numpy(), samples=samples)

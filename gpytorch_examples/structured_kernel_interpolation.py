import torch
from gpytorch import models
from gpytorch import means, kernels, distributions, likelihoods, mlls, settings
from gpytorch.utils.grid import choose_grid_size
import numpy as np
import pods
import time
from visualization import plot_gp


#SKI (or KISS-GP) is a great way to scale a GP up to very large datasets (100,000+ data points).
# Kernel interpolation for scalable structured Gaussian processes (KISS-GP)
# was introduced in this paper: http://proceedings.mlr.press/v37/wilson15.pdf

# SKI is asymptotically very fast (nearly linear), very precise (error decays cubically)

class StructredKernelGaussianProcess(models.ExactGP):

    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        # SKI requires a grid size hyperparameter. This util can help with that.
        # Here we are using a grid that has the same number of points as the training data (a ratio of 1.0).
        # Performance can be sensitive to this parameter, so you may want to adjust it for your own problem on a validation set
        grid_size = choose_grid_size(x_train, ratio=1.0)
        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(kernels.GridInterpolationKernel(kernels.RBFKernel(),
                                                                                grid_size=grid_size,
                                                                                num_dims=1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)


data = pods.datasets.olympic_marathon_men()
x_train = torch.from_numpy(data["X"]).squeeze(-1).type(torch.float32)
y_train = torch.from_numpy(data["Y"]).squeeze(-1).type(torch.float32)# + torch.randn(train_x.size()) * np.sqrt(0.04)


likelihood = likelihoods.GaussianLikelihood()
model = StructredKernelGaussianProcess(x_train, y_train, likelihood)

x_train = x_train.cuda()
y_train = y_train.cuda()
model = model.cuda()
likelihood = likelihood.cuda()


model.train()
likelihood.train()


optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)

##loss for gp
marginal_loglikelihood = mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 1000
for i in range(training_iter):
    optimizer.zero_grad()

    output = model(x_train)

    loss = -marginal_loglikelihood(output, y_train) # this gives the marginal loglikelihood  log(p(y|X))
    loss.backward()

    print(f'Iter {i + 1} - Loss: {loss.item()}   noise: {model.likelihood.noise.item()}')

    optimizer.step()


model.eval()
likelihood.eval()


with torch.no_grad(), settings.fast_pred_var(), settings.max_root_decomposition_size(25):
    x_test = torch.from_numpy(np.linspace(1870, 2030, 200)[:, np.newaxis]).type(torch.float32)
    x_test = x_test.cuda()
    f_preds = model(x_test)
    y_pred = likelihood(f_preds)

# plot
with torch.no_grad():
    mean = y_pred.mean.cpu().numpy()
    var = y_pred.variance.cpu().numpy()
    samples = y_pred.sample().cpu().numpy()
    plot_gp(mean, var, x_test.cpu().numpy(), X_train=x_train.cpu().numpy(), Y_train=y_train.cpu().numpy(), samples=samples)



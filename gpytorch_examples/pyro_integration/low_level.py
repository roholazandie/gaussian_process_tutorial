from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
import torch
import pyro
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from tqdm import tqdm

from gpytorch_examples.data_utils import get_pyro_data


class PyroGaussianProcessRegressionModel(ApproximateGP):

    def __init__(self, num_inducing_points=64, name_prefix="mixture_gp"):
        self.name_prefix = name_prefix

        inducing_points = torch.linspace(0, 1, num_inducing_points)
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution)

        super().__init__(variational_strategy)

        self.mean = ConstantMean()
        self.covar = ScaleKernel(RBFKernel())

    def forward(self, x):
        # which computes the prior GP mean and covariance at the supplied times.
        x_mean = self.mean(x)
        x_covar = self.covar(x)
        return MultivariateNormal(x_mean, x_covar)

    def guide(self, x, y):
        # which defines the approximate GP posterior.
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # draw independent samples from q
            pyro.sample(self.name_prefix + "f(x)", function_dist)

    def model(self, x, y):
        '''
        Computes the GP prior at x
        Converts GP function samples into scale function samples, using the link function defined above.
        Sample from the observed distribution p(y | f). (This takes the place of a gpytorch Likelihood that we would've used in the high-level interface).
        '''
        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # draw independent samples from p
            samples = pyro.sample(self.name_prefix + "f(x)", function_dist)

            # Use the link function to convert GP samples into scale samples
            scale_samples = samples.exp()

            # sample from observed distribution
            return pyro.sample(self.name_prefix + ".y",
                               pyro.distributions.Exponential(scale_samples.reciprocal()),
                               obs=y)


x_train, y_train = get_pyro_data()
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()

model = PyroGaussianProcessRegressionModel()

num_iter = 200
num_particles = 256


def train():
    optimizer = pyro.optim.Adam({"lr": 0.1})
    elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True)
    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()
    iterator = tqdm(range(num_iter))
    for i in iterator:
        model.zero_grad()
        loss = svi.step(x_train, y_train)
        iterator.set_postfix(loss=loss, lengthscale=model.covar.base_kernel.lengthscale.item())

train()
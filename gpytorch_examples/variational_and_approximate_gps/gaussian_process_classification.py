import math
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, UnwhitenedVariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from matplotlib import pyplot as plt


class GaussianProcessClassification(ApproximateGP):
    def __init__(self, x_train):
        '''
        Since exact inference is intractable for GP classification, GPyTorch approximates the classification posterior
        using variational inference. We believe that variational inference is ideal for a number of reasons.
        Firstly, variational inference commonly relies on gradient descent techniques, which take full advantage
        of PyTorch's autograd. This reduces the amount of code needed to develop complex variational models.
        Additionally, variational inference can be performed with stochastic gradient decent,
        which can be extremely scalable for large datasets.
        '''
        variational_distribution = CholeskyVariationalDistribution(x_train.size(0))
        # we're using an UnwhitenedVariationalStrategy because we are using the training data as inducing points
        variational_strategy = UnwhitenedVariationalStrategy(self,
                                                             inducing_points=x_train,
                                                             variational_distribution=variational_distribution,
                                                             learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean = ConstantMean()
        self.covar = ScaleKernel(RBFKernel())

    def forward(self, x):
        x_mean = self.mean(x)
        x_covar = self.covar(x)
        return MultivariateNormal(x_mean, x_covar)


x_train = torch.linspace(0, 1, 10)
y_train = torch.sign(torch.cos(x_train * (4 * math.pi))).add(1).div(2)

# Initialize model and likelihood
model = GaussianProcessClassification(x_train)
likelihood = BernoulliLikelihood()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the number of training datapoints
mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())

n_iterations = 100
for i in range(n_iterations):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(x_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    print(f"Iter {i} - Loss: {loss.item()}")
    optimizer.step()


# Go into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():
    # Test x are regularly spaced by 0.01 0,1 inclusive
    test_x = torch.linspace(0, 1, 101)
    # Get classification predictions
    observed_pred = likelihood(model(test_x))

    # Initialize fig and axes for plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(x_train.numpy(), y_train.numpy(), 'k*')
    # Get the predicted labels (probabilites of belonging to the positive class)
    # Transform these probabilities to be 0/1 labels
    pred_labels = observed_pred.mean.ge(0.5).float()
    ax.plot(test_x.numpy(), pred_labels.numpy(), 'b')
    ax.set_ylim([-1, 2])
    ax.legend(['Observed Data', 'Mean'])
    plt.show()

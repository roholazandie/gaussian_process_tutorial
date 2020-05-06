import numpy as np
from GPy.core import parameterization


def posterior_sample(model, X, **kwargs):
    """Give a sample from the posterior of the deep GP."""
    Z = X
    for i, layer in enumerate(reversed(model.layers)):
        Z = layer.posterior_samples(Z, size=1, **kwargs).squeeze(1)

    return Z


def initialize(model, noise_factor=0.01, linear_factor=1):
    """Helper function for deep model initialization."""
    model.obslayer.likelihood.variance = model.Y.var() * noise_factor
    for layer in model.layers:
        if type(layer.X) is parameterization.variational.NormalPosterior:
            if layer.kern.ARD:
                var = layer.X.mean.var(0)
            else:
                var = layer.X.mean.var()
        else:
            if layer.kern.ARD:
                var = layer.X.var(0)
            else:
                var = layer.X.var()

        # Average 0.5 upcrossings in four standard deviations.
        layer.kern.lengthscale = linear_factor * np.sqrt(layer.kern.input_dim) * 2 * 4 * np.sqrt(var) / (2 * np.pi)

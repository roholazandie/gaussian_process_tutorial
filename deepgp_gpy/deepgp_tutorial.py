from GPy.models import GPRegression
from GPy.kern import RBF
import numpy as np
import pods
from deepgp_gpy.utils import posterior_sample
from deepgp import DeepGP
from visualization import plot_gp, model_output, pred_range, visualize_pinball

data = pods.datasets.olympic_marathon_men()
x = data['X']
y = data['Y']

offset = np.mean(y)
scale = np.sqrt(np.var(y))

xlim = (1875, 2030)
ylim = (2.5, 6.5)
yhat = (y - offset) / scale

gp_regression = GPRegression(x, yhat)
gp_regression.optimize()

model_output(gp_regression)
###################deep gp

hidden = 1
dgp = DeepGP([y.shape[1], hidden, x.shape[1]], Y=yhat, X=x, inits=['PCA', 'PCA'],
             kernels=[RBF(hidden, ARD=True),
                      RBF(x.shape[1], ARD=True)],  # the kernels for each layer
             num_inducing=50, back_constraint=False)

dgp.optimize(messages=True, max_iters=100)

# plot the layer1 observables to hidden layer
xt = pred_range(x)
yt_mean, yt_var = dgp.layers[1].predict(xt)

samples = dgp.layers[1].posterior_samples(xt, size=1).squeeze(1)

plot_gp(yt_mean, yt_var, xt, dgp.layers[1].X.flatten(), dgp.layers[1].Y.mean, samples=samples,
        title="layer1 observables to hidden layer")

# plot the layer0 from hidden to targets
x = dgp.layers[1].Y.mean
xt = pred_range(x)
yt_mean, yt_var = dgp.layers[0].predict(xt)

samples = dgp.layers[0].posterior_samples(xt, size=1).squeeze(1)

plot_gp(yt_mean, yt_var, xt, dgp.layers[0].X.mean, dgp.layers[0].Y, samples=samples,
        title="layer0 from hidden to targets")

# dgp.obslayer.kern.variance.constrain_positive(warning=False)

# for layer in dgp.layers:
#     layer.likelihood.variance.constrain_positive()


xt = pred_range(data['X'])
samples = posterior_sample(dgp, xt).flatten()
model_output(dgp, samples)

visualize_pinball(dgp, fig_name="pinball.svg")

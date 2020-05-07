from sklearn.datasets import load_digits
import numpy as np
from deepgp import DeepGP
from GPy.kern import RBF
import matplotlib.pyplot as plt

def staged_initialization(dgp):
    '''
    Just like deep neural networks, there are some tricks to intitializing these models.
    The tricks we use here include some early training of the model with model parameters constrained.
    This gives the variational inducing parameters some scope to tighten the bound for the case where
    the noise variance is small and the variances of the Gaussian processes are around 1
    '''

    dgp.obslayer.likelihood.variance[:] = Y.var() * 0.01
    for layer in dgp.layers:
        layer.kern.variance.fix(warning=False)
        layer.likelihood.variance.fix(warning=False)

    dgp.optimize(messages=True, max_iters=10)

    for layer in dgp.layers:
        layer.kern.variance.constrain_positive(warning=False)

    dgp.optimize(messages=True, max_iters=10)

    for layer in dgp.layers:
        layer.likelihood.variance.constrain_positive(warning=False)

    dgp.optimize(messages=True, max_iters=10)

    return dgp


mnist = load_digits()

# Sub-sample the dataset to make the training faster.
np.random.seed(0)
digits = [0, 1, 2, 3, 4]
N_per_digit = 100
Y = []
labels = []
for d in digits:
    imgs = mnist['data'][mnist['target'] == d]
    Y.append(imgs[np.random.permutation(imgs.shape[0])][:N_per_digit])
    labels.append(np.ones(N_per_digit)*d)
Y = np.vstack(Y).astype(np.float64)
labels = np.hstack(labels)
Y /= 255.

num_latent = 2
num_hidden_2 = 5
dgp = DeepGP([Y.shape[1], num_hidden_2, num_latent],
             Y,
             kernels=[RBF(num_hidden_2, ARD=True), #automatic relevance determination (ARD) is true for the hidden layer
                      RBF(num_latent, ARD=False)], #but we want it to be fixed for latent layer because ARD finds the appropriate size of latent space
             num_inducing=50, back_constraint=False,
             encoder_dims=[[200], [200]])


dgp = staged_initialization(dgp)

def plot_hidden_states():
    # the top layer with latent variable
    fig, ax = plt.subplots(figsize=(16, 9))
    for d in digits:
        ax.plot(dgp.layers[1].X.mean[labels == d, 0], dgp.layers[1].X.mean[labels == d, 1], '.', label=str(d))
    _ = plt.legend()

    plt.show()

    # layers[0] is the observation layer
    fig, ax = plt.subplots(figsize=(16, 9))
    for i in range(5):
        for j in range(i):
            dims = [i, j]
            ax.cla()
            for d in digits:
                plt.plot(dgp.layers[0].X.mean[labels == d, dims[0]].values,
                         dgp.layers[0].X.mean[labels == d, dims[1]].values, '.', label=str(d))
            plt.legend()
            plt.xlabel('dimension ' + str(dims[0]))
            plt.ylabel('dimension ' + str(dims[1]))

            plt.show()


#plot_hidden_states()

## Generate from model
# we can take a look at a sample from the model, by drawing a Gaussian
# random sample in the latent space and propagating it through the model
# this is very similar to restricted boltzman machines

rows = 10
cols = 20
t = np.linspace(-1, 1, rows*cols)[:, None]
kern = RBF(1, lengthscale=0.05)
cov = kern.K(t, t)
x = np.random.multivariate_normal(np.zeros(rows*cols), cov, num_latent).T


yt, _ = dgp.predict(x)
fig, axs = plt.subplots(rows, cols, figsize=(10,6))
for i in range(rows):
    for j in range(cols):
        #v = np.random.normal(loc=yt[0][i*cols+j, :], scale=np.sqrt(yt[1][i*cols+j, :]))
        v = yt[i*cols+j, :]
        axs[i, j].imshow(v.reshape(8, 8),
                        cmap='gray', interpolation='none',
                        aspect='equal')
        axs[i, j].set_axis_off()

plt.show()
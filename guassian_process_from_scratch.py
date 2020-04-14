import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import multivariate_normal

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


def rbf_kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes
    a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


def kernel(X1, X2, *args, **kwargs):
  return rbf_kernel(X1, X2, *args, **kwargs)


def posterior_predictive(X_test, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.

    mu_s = mean(x_test) + K_s^T x K^-1 x (f-mean(x_train))
    cov_s = K_ss - K_s^T x K^-1 x K_s

    Args:
        X_test: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_test, l, sigma_f)
    K_ss = kernel(X_test, X_test, l, sigma_f) + 1e-8 * np.eye(len(X_test))
    K_inv = inv(K)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    return mu_s, cov_s


################Prior########################
# Finite number of points
X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)

# Mean and covariance of the prior
mu = np.zeros(X_test.shape) # we set the mean to zero without loss of generality

cov = kernel(X_test, X_test)

# Draw three samples from the prior
Y_test = np.random.multivariate_normal(mu.ravel(), cov, 3)
#print(samples)


# Plot GP mean, confidence interval and samples
plot_gp(mu, cov, X_test, samples=Y_test)
plt.show()

#################Posterior#####################
# Noise free training data
X_train = np.array([-4, -3, -2, -1, 3]).reshape(-1, 1)
Y_train = np.sin(X_train)

# Compute mean and covariance of the posterior predictive distribution
mu_s, cov_s = posterior_predictive(X_test, X_train, Y_train)
# Y_test is not unique! we draw samples, we have a distribution of Y_test's (f's) In other words there is no maximum likelihood hypothesis,
# we have all hypotheis with different probs of representing the truth

#Y_test = np.random.multivariate_normal(mu_s.ravel(), cov_s, 4)
Y_test = multivariate_normal.rvs(mu_s.ravel(), cov_s, 5)
entropy = multivariate_normal.entropy(mu_s.ravel(), cov_s)
print(entropy)
#likelihood = multivariate_normal.logpdf(Y_test, mu_s.ravel(), cov_s)
#likelihood/=sum(likelihood)
plot_gp(mu_s, cov_s, X_test, X_train=X_train, Y_train=Y_train, samples=Y_test)
plt.show()




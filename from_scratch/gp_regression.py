import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from visualization import plot_gp_2D
from from_scratch.gpr_best_hyperparameters import best_hyperparamters
from from_scratch.kernels import kernel, rbf

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
    K = kernel(X_train, X_train, kernel_func=rbf, l=l, sigma_f=sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_test, kernel_func=rbf, l=l, sigma_f=sigma_f)
    K_ss = kernel(X_test, X_test, kernel_func=rbf, l=l, sigma_f=sigma_f) + 1e-8 * np.eye(len(X_test))
    K_inv = inv(K)
    mu_s = np.transpose(K_s).dot(K_inv).dot(Y_train)
    cov_s = K_ss - np.transpose(K_s).dot(K_inv).dot(K_s)
    return mu_s, cov_s


################Prior########################


# Finite number of points
X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)

# Mean and covariance of the prior
mu = np.zeros(X_test.shape) # we set the mean to zero without loss of generality

cov = kernel(X_test, X_test, kernel_func=rbf)

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

##############best hyperparamters using minimization of NLL ############
best_l, best_sigma_f = best_hyperparamters(X_train, Y_train)
print("best l", best_l)
print("best sigma_f", best_sigma_f)
mu_s, cov_s = posterior_predictive(X_test, X_train, Y_train, l=best_l, sigma_f=best_sigma_f)
plot_gp(mu_s, cov_s, X_test, X_train=X_train, Y_train=Y_train, samples=Y_test)
plt.show()



##########################3D gaussian regression ############
noise_2D = 0.1

rx, ry = np.arange(-5, 5, 0.3), np.arange(-5, 5, 0.3)
gx, gy = np.meshgrid(rx, rx)

X_2D = np.c_[gx.ravel(), gy.ravel()]

X_2D_train = np.random.uniform(-4, 4, (100, 2))
Y_2D_train = np.sin(0.5 * np.linalg.norm(X_2D_train, axis=1)) + \
             noise_2D * np.random.randn(len(X_2D_train))

mu_s, _ = posterior_predictive(X_2D, X_2D_train, Y_2D_train, sigma_y=noise_2D)

plot_gp_2D(gx, gy, mu_s, X_2D_train, Y_2D_train, f'Before parameter optimization: l={1.00} sigma_f={1.00}', 1)

best_l, best_sigma_f = best_hyperparamters(X_2D_train, Y_2D_train, noise_2D)

mu_s, _ = posterior_predictive(X_2D, X_2D_train, Y_2D_train, l=best_l, sigma_f=best_sigma_f, sigma_y=noise_2D)
plot_gp_2D(gx, gy, mu_s, X_2D_train, Y_2D_train, f'After parameter optimization: l={best_l:.2f} sigma_f={best_sigma_f:.2f}', 2)
plt.show()
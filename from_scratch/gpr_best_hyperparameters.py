import numpy as np
from numpy.linalg import cholesky, inv
from from_scratch.kernels import kernel, rbf
from scipy.optimize import minimize


def best_hyperparamters(X_train, Y_train, noise=0):

    def nll(X_train, Y_train, noise=0):
        '''
        compute the nll for a given theta
        '''
        def step(theta):
            '''
            We have
            \log\det(\Sigma) = 2 \sum_i \log [ diag(L)_i ]
            to calculate the det easier
            '''
            K = kernel(X_train, X_train, kernel_func=rbf, l=theta[0], sigma_f=theta[1]) + noise**2 * np.eye(len(X_train))
            N = len(Y_train)
            nll_value = 0.5 * np.transpose(Y_train).dot(inv(K).dot(Y_train)) + np.sum(np.log(np.diagonal(cholesky(K)))) + 0.5*N*np.log(2*np.pi)
            return nll_value

        return step

    # we start at [l, sigma]=[1, 1]
    result = minimize(nll(X_train, Y_train, noise), [1, 1], bounds=((1e-5, None), (1e-5, None)), method='L-BFGS-B')
    l = result.x[0]
    sigma_f = result.x[1]
    return l, sigma_f

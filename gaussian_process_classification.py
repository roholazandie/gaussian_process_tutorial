from kernels import rbf_kernel
import numpy as np
from scipy.stats import norm
from scipy.special import erf
from scipy.linalg import solve
from numpy.linalg import cholesky
from scipy.optimize import minimize

best_nlZ = np.inf

def cholesky_solve(a, b):
    return solve(a, solve(a.T, b))


def cum_gaussian(x):
    return norm.cdf()


def cum_gauss_derivative(y, f):
    yf = y * f
    p = (1 + erf(yf / np.sqrt(2))) / 2

    lp = np.zeros(len(f))
    b = 0.158482605320942
    c = -1.785873318175113
    ok = yf > -6
    lp[ok] = np.log(p[ok])
    lp[~ok] = -yf[~ok] ** 2 / 2 + b * yf[~ok] + c

    return p, lp


def cum_gauss_hessian(y, f):

    yf = y * f
    p, lp = cum_gauss_derivative(y, f)
    out1 = np.sum(lp)

    n_p = np.zeros(len(f))
    ok = yf > -5
    n_p[ok] = (np.exp(-yf[ok] ** 2 / 2) / (np.sqrt(2 * np.pi)) / p[ok])

    bd = yf < -6
    n_p[bd] = np.sqrt(yf[bd] ** 2 / 4 + 1) - yf[bd] / 2

    interp = np.logical_and(np.logical_not(ok), np.logical_not(bd))
    tmp = yf[interp]
    lam = -5 - yf[interp]
    n_p[interp] = (1 - lam) * (np.exp(-tmp ** 2 / 2) / np.sqrt(2 * np.pi)) / p[interp] + lam * (
                np.sqrt(tmp ** 2 / 4 + 1) - tmp / 2)

    out2 = y * n_p

    out3 = -n_p ** 2 - yf * n_p

    out4 = 2*y*n_p**3 + 3*f*n_p**2 + y*(f**2 - 1)* n_p

    return out1, out2, out3, out4


def binary_laplace_gpc(X_train, y, l, sigma_f):

    def step(best_alpha, best_nlZ=np.inf):

        tol = 10e-6
        N = len(X_train)

        #best_alpha = np.random.randn(N, 1)
        #best_alpha = np.zeros((N, 1))
        #best_nlZ = np.inf

        K = rbf_kernel(X_train, X_train, l=l, sigma_f=sigma_f)
        alpha = best_alpha

        f = K.dot(alpha).flatten()
        lp, dlp, d2lp, d3lp = cum_gauss_hessian(y, f)
        W = -d2lp
        psi_new = -alpha.T.dot(f / 2) + lp

        if psi_new < -N * np.log(2):
            f = np.zeros((N, 1)).flatten()
            alpha = f
            lp, dlp, d2lp, d3lp = cum_gauss_hessian(y, f)
            W = -d2lp
            psi_new = -alpha.T.dot(f / 2) + lp

        # newtons iteration
        psi_old = -np.inf
        while psi_new - psi_old > tol:
            psi_old = psi_new
            alpha_old = alpha

            sW = np.sqrt(W)
            L = cholesky(np.eye(N) + sW.dot(sW.T) * K)
            b = W * f + dlp
            alpha = b - sW * cholesky_solve(L, sW * (K.dot(b)))
            lp, dlp, d2lp, d3lp = cum_gauss_hessian(y, f)
            W = -d2lp

            psi_new = -alpha.T.dot(f / 2) + lp
            i = 0
            while i < 10 and psi_new < psi_old:  # if objective didn't increase reduce step size by half
                alpha = (alpha + alpha_old) / 2
                f = K.dot(alpha).flatten()
                lp, dlp, d2lp, d3lp = cum_gauss_hessian(y, f)
                W = -d2lp
                psi_new = -alpha.T.dot(f / 2)[0] + lp
                i += 1
        sW = np.sqrt(W)
        L = cholesky(np.eye(N) + sW.dot(sW.T) * K)
        nlZ = alpha.T.dot(f / 2) - lp + np.sum(np.log(np.diag(L)))

        if nlZ < best_nlZ:
            best_nlZ = nlZ
            best_alpha = alpha
            print(best_nlZ)

        return best_nlZ
    # dnlZ = np.zeros((2, 1))
    # Z = np.tile(sW, [1, N]) * cholesky_solve(L, np.diag(sW))
    # C = solve(L.T, np.tile(sW, [1, N]) * K)
    # s2 = 0.5 * (np.diag(K) - np.sum(C**2, 1).T) * d3lp
    # for j in range(2):

    return step


# X_train = np.array([[1, 2], [3, 4], [5, 6]])
# y = np.array([1, 1, -1])
# binary_laplace_gpc(X_train, y, l=1, sigma_f=1)


## make data
n1 = 80
n2 = 40
S1 = np.eye(2)
S2 = np.array([[1, 0.95], [0.95, 1]])
m1 = np.array([0.75, 0])
m2 = np.array([-0.75, 0])

x1 = cholesky(S1).dot(np.random.randn(2, n1)) + np.tile(m1, [n1, 1]).T
x2 = cholesky(S2).dot(np.random.randn(2, n2)) + np.tile(m2,[n2, 1]).T
x = np.hstack([x1, x2]).T
y = np.hstack([np.tile(-1, [1, n1]), np.tile(1, [1, n2])]).T
#t1, t2 = np.meshgrid(np.arange(-4, 4, 0.1),np.arange(-4, 4, 0.1))

# N = len(x)
# l = binary_laplace_gpc(x, y.flatten(), 1, 1)
# print(l(np.random.randn(N, 1), np.inf))

N = len(x)
result = minimize(binary_laplace_gpc(x, y.flatten(), 1, 1), [np.random.randn(N, 1)], method='L-BFGS-B')
print(result.x)

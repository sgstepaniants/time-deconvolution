import numpy as np


def inverse_cdf(alpha, beta, x):
    n_pieces = len(alpha)
    beta_cumsum = np.insert(np.cumsum(beta), 0, 0)
    Finv = np.zeros(len(x))
    for k in range(n_pieces):
        if k < n_pieces-1:
            inds = np.logical_and(beta_cumsum[k] <= x, x < beta_cumsum[k+1])
        elif k == n_pieces-1:
            inds = beta_cumsum[k] <= x
        Finv[inds] = alpha[k]
    return Finv

def wasserstein_dist_invcdf(Finv1, Finv2, x, p=2):
    dx = x[1] - x[0]
    return np.sum((np.sum((Finv1 - Finv2)**p)*dx)**(1/p))

def wasserstein_dist(alpha1, beta1, alpha2, beta2, x, p=2):
    Finv1 = inverse_cdf(alpha1, beta1, x)
    Finv2 = inverse_cdf(alpha2, beta2, x)
    return wasserstein_dist_invcdf(Finv1, Finv2, x, p=p)
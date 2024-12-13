import math
import numpy as np
import scipy
from pykeops.numpy import LazyTensor

def mobius(theta):
    return np.real(1j*(1-np.exp(1j*theta))/(1+np.exp(1j*theta)))

def mobius_inv(x):
    z = (1j - x)/(1j + x)
    return np.arctan2(np.imag(z), np.real(z))

# c1 and b are assumed to be positive
# computing the inverse of dynamical system c1s - c0 - \sum_i b_i/(s + a_i)
def B_R_discrete(c0, c1, a, b):
    n = len(a)
    assert(len(b) == n)
    assert(np.all(b >= 0))
    assert(c1 >= 0)
    
    sorted_inds = np.argsort(a)
    a = a[sorted_inds]
    b = b[sorted_inds]
    
    alphas = np.zeros(n-1)
    diffs = np.diff(a)
    for i in range(n-1):
        if diffs[i] < 1e-10:
            alphas[i] = a[i]
            continue
        
        def f(x, i):
            if x == 0:
                return np.infty
            elif x == 1:
                return -np.infty
            else:
                y = diffs[i]*x + a[i]
                return np.sum(b/(y-a)) - (c1*y + c0)
        root = scipy.optimize.brentq(f, 0, 1, args=i)
        alphas[i] = diffs[i]*root + a[i]
    
    if c0 < 0 or c1 > 0:
        def f(x):
            if x == a[0]:
                return -np.inf
            return np.sum(b/(x - a)) - (c1*x + c0)
        root_found = False
        k = 1
        while not root_found:
            x_left = a[0] - 10**k
            root_found = f(x_left) > 0
            k += 1
        try:
            left_alpha = scipy.optimize.brentq(f, x_left, a[0])
            alphas = np.insert(alphas, 0, left_alpha)
        except:
            print("no left root")
            pass
    if c0 > 0 or c1 > 0:
        def f(x):
            if x == a[-1]:
                return np.inf
            return np.sum(b/(x - a)) - (c1*x + c0)
        root_found = False
        k = 1
        while not root_found:
            x_right = a[-1] + 10**k
            root_found = f(x_right) < 0
            k += 1
        try:
            right_alpha = scipy.optimize.brentq(f, a[-1], x_right)
            alphas = np.append(alphas, right_alpha)
        except:
            print("no right root")
            pass
    
    if c1 > 0:
        zeta0 = 0
        zeta1 = 0
    elif c0 != 0:
        zeta0 = -np.pi**2/c0
        zeta1 = 0
    else:
        m0_lmbda = np.sum(b)
        m1_lmbda = np.sum(b*a)
        zeta0 = -np.pi**2*m1_lmbda/m0_lmbda**2
        zeta1 = np.pi**2/m0_lmbda
    
    betas = np.pi**2/(c1 + np.sum(b[:, None]/(a[:, None] - alphas[None, :])**2, axis=0))
    return zeta0, zeta1, alphas, betas

def hilbert_transform(f, X, dX, Y=None, fc=None, Xc=None, eps=1e-7, lazy=True):
    if X.ndim == 1:
        X = X[None, :]
        if Y is not None:
            Y = Y[None, :]
    d = X.shape[0]
    
    if d > 1 and len(dX) == 1:
        dX = d*[dX[0]]
    
    Hf = np.zeros(X.shape)
    
    if Y is None:
        Y = X
    
    if lazy:
        Yi = LazyTensor(Y.T[:, None, :])
        Xj = LazyTensor(X.T[None, :, :])
        fj = LazyTensor(f[None, :, None])

        D = Yi - Xj
        Dsign = -2*(D.sign() == -1).sum(2).mod(2) + 1
        Dabs = D.abs()
        Dprod = Dabs.log().sum(2).exp() * Dsign
        K = np.prod(dX) * fj/Dprod
        K = (Dprod.abs() - eps).ifelse(K, 0)
        Hf = np.squeeze(K.sum(2).sum(1))/(math.pi**d)
    else:
        D = np.prod(Y[:, :, None] - X[:, None, :], axis=0)
        K = np.prod(dX)/D
        K[np.abs(D) <= eps] = 0
        Hf = K@f/(math.pi**d)
    
    if fc is not None and xc is not None:
        for i in range(len(fc)):
            d = y - xc[i]
            k = fc[i]/d
            k[np.abs(d) <= eps] = 0
            Hf += k/math.pi
    return Hf

def exp_kernel(alphas, betas, t, quadrature=False):
    squeeze = len(alphas.shape) == 1
    if squeeze:
        alphas = alphas[None, :]
        betas = betas[None, :]
    dy = np.ones(alphas.shape[0])
    if quadrature:
        dy = alphas[:, 1] - alphas[:, 0]
    kernels = np.sum(betas[:, :, None] * np.exp(-alphas[:, :, None] * t[None, None, :]) * dy[:, None, None], axis=1)
    if squeeze:
        kernels = np.squeeze(kernels, axis=0)
    return kernels
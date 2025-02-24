import math
import numpy as np
import scipy

from aaa_algorithms import *

# Find root of func(x) in interval [a, b]
def rescale_rootfind(func, interval, root_tol=1e-15):
    a, b = interval
    assert(a <= b)
    if a == b:
        return a if np.abs(func(a)) < root_tol else None
    
    #if np.sign(func(a)) == np.sign(func(b)):
    #    return None
    
    if np.isfinite(b - a):
        def rescaled_func(x):
            if x == 0:
                return np.infty
            elif x == 1:
                return -np.infty
            return func((b-a)*x + a)
        root, result = scipy.optimize.brentq(rescaled_func, 0, 1, full_output=True, disp=False)
        return (b-a)*root + a if result.converged else None
    elif np.isinf(b) and np.isfinite(a):
        def rescaled_func(x):
            if x == 0:
                return np.infty
            elif x == 1:
                return -1e-15
            return func(x/(1-x) + a)
        root, result = scipy.optimize.brentq(rescaled_func, 0, 1, full_output=True, disp=False)
        return root/(1-root) + a if result.converged and root != 1 else None
    elif np.isinf(a) and np.isfinite(b):
        def rescaled_func(x):
            if x == -1:
                return 1e-15
            elif x == 0:
                return -np.infty
            return func(x/(1+x) + b)
        root, result = scipy.optimize.brentq(rescaled_func, -1, 0, full_output=True, disp=False)
        return root/(1+root) + b if result.converged and root != -1 else None
    else:
        def rescaled_func(x):
            if x == -1:
                return -1e-15
            if x == 1:
                return 1e-15
            return func(x/(1-np.abs(x)))
        root, result = scipy.optimize.brentq(rescaled_func, -1, 1, full_output=True, disp=False)
        return root/(1-np.abs(root)) if result.converged and root != -1 and root != 1 else None

# Accepts a function u, performs AAA rational approximation,
# and returns the Hilbert transform of u on the real line
def continuous_hilbert_transform(us, zs, aaa_iters=100):
    pol, _, _, _, _, _ = aaa(us, zs, max_terms=aaa_iters)
    
    pol = pol[np.imag(pol) < 0]
    d = np.min(np.abs(zs - pol[:, None]), axis=1)
    
    A = d / (zs[:, None] - pol)
    A = np.concatenate([np.real(A), -np.imag(A)], axis=1)
    c = np.reshape(np.linalg.lstsq(A, us, rcond=None)[0], (-1, 2), order='F') @ np.array([1, 1j])

    f = lambda x: np.sum((c*d) / (x[:, None] - pol), axis=1)
    Hu = lambda x: np.imag(f(x))
    return Hu

# Accepts a function u, performs AAA rational approximation,
# and returns the Hilbert transform of u with periodic extension away from its domain of definition
def continuous_periodic_hilbert_transform(us, xs, aaa_iters=100, periodic_domain=(-np.pi, np.pi)):
    zs = np.exp(1j*(xs - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)
    pol, _, _, _, _, _ = aaa(us, zs, max_terms=aaa_iters)
    
    pol = pol[np.abs(pol) < 1]
    d = np.min(np.abs(zs - pol[:, None]), axis=1)
    
    A = d / (zs[:, None] - pol)
    A = np.concatenate([np.real(A), -np.imag(A)], axis=1)
    c = np.reshape(np.linalg.lstsq(A, us, rcond=None)[0], (-1, 2), order='F') @ np.array([1, 1j])

    f = lambda x: np.sum((c*d) / (x[:, None] - pol), axis=1)
    Hu = lambda x: -np.imag(f(np.exp(1j*(x - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)))
    return Hu

'''
def continuous_periodic_hilbert_transform(us, xs, aaa_iters=100, periodic_domain=(-np.pi, np.pi)):
    zs = np.exp(1j*(xs - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)
    pol, _, _, _, _, _ = aaa(us, zs, max_terms=aaa_iters)
    
    pol = pol[np.abs(pol) < 1]
    d = np.min(np.abs(zs - pol[:, None]), axis=1)
    
    A = d / (zs[:, None] - pol)
    A = np.concatenate([np.real(A), -np.imag(A)], axis=1)
    c = np.reshape(np.linalg.lstsq(A, us, rcond=None)[0], (-1, 2), order='F') @ np.array([1, 1j])

    f = lambda x: np.sum((c*d) / (x[:, None] - pol), axis=1)
    Hu = lambda x: -np.imag(f(np.exp(1j*(x - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)))
    return Hu

def continuous_periodic_hilbert_transform_v2(us, xs, aaa_iters=100, periodic_domain=(-np.pi, np.pi)):
    zs = np.exp(1j*(xs - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)
    pol, res, _, _, _, _ = aaa(us, zs, max_terms=aaa_iters)
    
    in_circle = np.abs(pol) < 1

    f = lambda x: np.sum(res[in_circle] / (x[:, None] - pol[in_circle]), axis=1) - np.sum(res[~in_circle] / (x[:, None] - pol[~in_circle]), axis=1)
    Hu = lambda x: -np.imag(f(np.exp(1j*(x - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)))
    return Hu

def continuous_periodic_hilbert_transform_v3(us, xs, aaa_iters=100, periodic_domain=(-np.pi, np.pi)):
    zs = np.exp(1j*(xs - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)
    pol, res, _, _, _, _ = aaa(us, zs, max_terms=aaa_iters)
    
    in_circle = np.abs(pol) < 1

    f = lambda x: np.sum(res[in_circle] / (x[:, None] - pol[in_circle]), axis=1)
    Hu = lambda x: -2*np.imag(f(np.exp(1j*(x - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)))
    return Hu

def continuous_periodic_hilbert_transform_v4(us, xs, aaa_iters=100, periodic_domain=(-np.pi, np.pi)):
    zs = np.exp(1j*(xs - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)
    pol, res, _, _, _, _ = aaa(us, zs, max_terms=aaa_iters)
    
    in_circle = np.abs(pol) > 1

    f = lambda x: np.sum(res[in_circle] / (x[:, None] - pol[in_circle]), axis=1)
    Hu = lambda x: 2*np.imag(f(np.exp(1j*(x - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)))
    return Hu

def continuous_periodic_hilbert_transform_v5(us, xs, aaa_iters=100, periodic_domain=(-np.pi, np.pi)):
    zs = np.exp(1j*(xs - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)
    pol, res, _, _, _, _ = aaa(us, zs, max_terms=aaa_iters)
    
    in_circle = np.abs(pol) < 1

    f = lambda x: np.sum(res[in_circle] / (x[:, None] - pol[in_circle]), axis=1)
    Hu = lambda x: -2*np.real(f(np.exp(-1j*(x - periodic_domain[0])/(periodic_domain[1] - periodic_domain[0])*2*np.pi)))
    return Hu
'''

def fourier_hilbert_transform(u, x):
    n = len(u)
    per = n/(n-1) * (x[-1] - x[0])
    thetas = 2*np.pi/per*np.fft.rfftfreq(n)*n
    wts = np.fft.rfft(u)/n
    Hu = lambda x: 2*np.imag(np.exp(1j*(x[:, None]-x[0])*thetas) @ wts)
    return Hu

class HilbertTransform:
    def __init__(self, lmbda, aaa_samples=None, aaa_iters=200):
        self.lmbda = lmbda
        
        if aaa_samples is not None:
            self.aaa_samples = aaa_samples
        elif self.lmbda.density is not None:
            self.aaa_samples = self.lmbda.quad_pts
        else:
            self.aaa_samples = np.array([])
        
        self.aaa_iters = aaa_iters
        
        # Discrete part of Hilbert transform (of discrete part of lambda)
        self.H_disc = lambda x: np.zeros_like(x)
        if self.lmbda.num_atoms > 0:
            if self.lmbda.periodic_domain is None:
                self.H_disc = lambda x: 1/math.pi*np.sum(self.lmbda.atom_wts[None, :] / (x[:, None] - self.lmbda.atoms[None, :]), axis=1)
            else:
                self.H_disc = lambda x: np.sum(self.lmbda.atom_wts[None, :] / np.tan((x[:, None] - self.lmbda.atoms[None, :])/2), axis=1)
        
        # Continuous part of Hilbert transform (of continuous density of lambda)
        self.H_cont = lambda x: np.zeros_like(x)
        if self.lmbda.density is not None:
            if self.lmbda.periodic_domain is None:
                # Use AAA rational approximation if domain is reals
                self.H_cont = continuous_hilbert_transform(self.lmbda.density(self.aaa_samples), self.aaa_samples, self.aaa_iters)
            else:
                # Use Fourier transform if domain is periodic
                #dx = self.lmbda.quad_pts[1] - self.lmbda.quad_pts[0]
                #assert(np.allclose(np.diff(self.lmbda.quad_pts), dx))
                #assert(self.lmbda.quad_pts[0] == self.lmbda.periodic_domain[0])
                #assert(np.isclose(self.lmbda.periodic_domain[1]-self.lmbda.quad_pts[-1], dx))
                #self.H_cont = fourier_hilbert_transform(self.lmbda.density_vals, self.lmbda.quad_pts)
                
                # Use AAA rational approximation if domain is periodic
                self.H_cont = continuous_periodic_hilbert_transform(self.lmbda.density(self.aaa_samples), self.aaa_samples, self.aaa_iters, self.lmbda.periodic_domain)
    
    def __call__(self, x):
        scalar = False
        if np.isscalar(x):
            scalar = True
            x = np.array([x])
        
        res = self.H_disc(x) + self.H_cont(x)
        
        if scalar:
            res = res[0]
        return res
    
    # Find roots of Hilbert transform H(x) when intersected with mx + c
    # in the zero sets of lambda
    def roots(self, m=0, c=0):
        if self.lmbda.periodic_domain is not None:
            assert(m == 0)
        
        def func(x):
            return self(x) - m*x - c
        
        if self.lmbda.zero_sets is None:
            return np.array([])
        
        # each lambda zero set should have at most one root since the Hilbert transform is monotonic there
        roots = np.array([])
        nzs = self.lmbda.zero_sets.shape[0]
        for i in range(nzs):
            root = None
            if self.lmbda.zero_sets[i, 0] == self.lmbda.zero_sets[i, 1]:
                root = self.lmbda.zero_sets[i, 0]
            elif (self.lmbda.periodic_domain is None) or i < nzs - 1:
                root = rescale_rootfind(func, self.lmbda.zero_sets[i, :])
            else:
                # Edge case if lmbda is defined on a periodic domain
                a = self.lmbda.zero_sets[i, 0]
                b = self.lmbda.zero_sets[i, 1]
                per_l = self.lmbda.periodic_domain[0]
                per_r = self.lmbda.periodic_domain[1]
                assert(b >= per_r)
                per = per_r - per_l
                func_per = lambda x: func(((x + a - per_l) % per) + per_l)
                interval = np.array([0, b-a])
                root = rescale_rootfind(func_per, interval)
                root = ((root + a - per_l) % per) + per_l
            
            if root is not None:
                roots = np.append(roots, root)
    
        return roots
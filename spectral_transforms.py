import math
import numpy as np
import scipy
import torch
from torch import nn
import torch.optim as optim
from pykeops.numpy import LazyTensor

from hilbert_transform import HilbertTransform
from probability import Distribution

def mobius(theta):
    return np.real(1j*(1-np.exp(1j*theta))/(1+np.exp(1j*theta)))

def mobius_inv(x):
    z = (1j - x)/(1j + x)
    return np.arctan2(np.imag(z), np.real(z))

# alphas is a grid, b_grid indicates the continuous density part of lambda on this grid
# a, b are the atoms and weights of the discrete part of lambda
# the density of lambda does not need to be compactly supported but needs to decay at infinity
def B_real(lmbda, c0, c1, H=None):
    assert(np.all(lmbda.quad_wts >= 0))
    assert(np.all(lmbda.atom_wts >= 0))
    
    assert(c1 >= 0)
    
    c0_re = np.real(c0)
    c0_im = np.imag(c0)
    assert(c0_im >= 0)
    
    alpha_disc = np.array([])
    beta_disc = np.array([])
    
    # If Hilbert transform of lmbda is not provided as a function, compute it
    if H is None:
        H = HilbertTransform(lmbda)
    
    lmbda_density = lambda s: 0
    if lmbda.density is not None:
        lmbda_density = lmbda.density
    
    if c0_im > 0:
        # Compute continuous part of mu, no discrete part
        if c1 == 0:
            mu_density = lambda s: (lmbda_density(s) + c0_im/math.pi)/((lmbda_density(s) + c0_im/math.pi)**2 + (H(s) - (c1*s + c0_re)/math.pi)**2) - math.pi*c0_im/(c0_re**2 + c0_im**2)
        else:
            mu_density = lambda s: (lmbda_density(s) + c0_im/math.pi)/((lmbda_density(s) + c0_im/math.pi)**2 + (H(s) - (c1*s + c0_re)/math.pi)**2)
        
        # Compute coefficients zeta0, zeta1
        zeta0 = 0
        zeta1 = 0
        if c1 == 0:
            zeta0 = -math.pi**2/c0
    else:
        # Compute continuous part of mu
        mu_density = None
        if lmbda.density is not None:
            mu_density = lambda s: lmbda_density(s)/(lmbda_density(s)**2 + (H(s) - (c1*s + c0_re)/math.pi)**2)
        
        # Compute discrete part of mu
        alpha_disc = H.roots(c1/math.pi, c0_re/math.pi)
        beta_disc = np.pi**2/(c1 + lmbda.offset_moments(alpha_disc, -2))
        
        # If lambda is purely atomic, verify all roots have been found
        if lmbda.density is None and lmbda.num_atoms > 0:
            assert(lmbda.zero_sets.shape[0] == lmbda.num_atoms+1)
            assert(lmbda.zero_sets[0, 0] == -np.inf)
            assert(lmbda.zero_sets[-1, 1] == np.inf)
            
            # there is exactly one root between every atom of lmbda
            if c1 > 0:
                # there is one root to the left of the first atom and to the right of the last atom
                print("left and right roots found")
                assert(np.logical_and(alpha_disc >= lmbda.zero_sets[:, 0], alpha_disc <= lmbda.zero_sets[:, 1]).all())
            elif c0_re < 0 and c1 == 0:
                # there is one root to the left of the first atom
                print("left root found")
                assert(np.logical_and(alpha_disc >= lmbda.zero_sets[:-1, 0], alpha_disc <= lmbda.zero_sets[:-1, 1]).all())
            elif c0_re > 0 and c1 == 0:
                # there is one root to the right of the last atom
                print("right root found")
                assert(np.logical_and(alpha_disc >= lmbda.zero_sets[1:, 0], alpha_disc <= lmbda.zero_sets[1:, 1]).all())
            elif c0_re == 0 and c1 == 0:
                # there are no roots to the left of the first atom and to the right of the last atom
                print("no roots to left or right")
                assert(np.logical_and(alpha_disc >= lmbda.zero_sets[1:-1, 0], alpha_disc <= lmbda.zero_sets[1:-1, 1]).all())
        
        # Compute coefficients zeta0, zeta1
        zeta0 = 0
        zeta1 = 0
        if c0_re != 0 and c1 == 0:
            zeta0 = -np.pi**2/c0_re
        elif c0_re == 0 and c1 == 0:
            m0_lmbda = lmbda.moment(0)
            m1_lmbda = lmbda.moment(1)
            zeta0 = -np.pi**2*m1_lmbda/m0_lmbda**2
            zeta1 = np.pi**2/m0_lmbda
    
    # Save continuous and discrete parts of mu into a distribution
    mu = Distribution(mu_density, alpha_disc, beta_disc, lmbda.quad_pts, lmbda.quad_wts)
    return mu, zeta0, zeta1

def exp_kernel(lmbda, t):
    kernel = np.zeros(len(t))
    if lmbda.num_atoms > 0:
        kernel += np.sum(lmbda.atom_wts*np.exp(-lmbda.atoms*t[:, None]), axis=1)
    if lmbda.density is not None:
        kernel += np.sum(lmbda.quad_wts*lmbda.density_vals*np.exp(-lmbda.quad_pts*t[:, None]), axis=1)
    return kernel

def complex_exp_kernel(lmbda, t):
    kernel = np.zeros(len(t), dtype=np.complex128)
    if lmbda.num_atoms > 0:
        kernel += np.sum(lmbda.atom_wts*np.exp(-1j*lmbda.atoms*t[:, None]), axis=1)
    if lmbda.density is not None:
        kernel += np.sum(lmbda.quad_wts*lmbda.density_vals*np.exp(-1j*lmbda.quad_pts*t[:, None]), axis=1)
    return kernel

def invert_volterra_cm(lmbda, c0, c1, H=None):
    mu, zeta0, zeta1 = B_real(lmbda, c0, c1, H=H)
    mu.rescale(-1/math.pi**2)
    return mu, -zeta0/math.pi**2, -zeta1/math.pi**2

def invert_volterra_pd(lmbda, c0, c1, H=None):
    mu, zeta0, zeta1 = B_real(lmbda, c0, c1, H=H)
    mu.rescale(1/math.pi**2)
    return mu, zeta0/math.pi**2, zeta1/math.pi**2

def fit_cosine_sum(x, t, n, omegas_init=None, betas_init=None, opt_iter=1000, omega_min=0, omega_max=np.inf, omega_init_scale=1):
    if omegas_init is None:
        omegas = torch.DoubleTensor(n).uniform_(omega_min, min(omega_max, omega_init_scale))
        omegas.requires_grad = True
    else:
        omegas = torch.tensor(omegas_init, requires_grad=True, dtype=torch.float64)
    
    if betas_init is None:
        betas = torch.randn(n, requires_grad=True, dtype=torch.float64)
    else:
        betas = torch.tensor(betas_init, requires_grad=True, dtype=torch.float64)
    
    optimizer = optim.Adam([omegas, betas], lr=1e-2)#, amsgrad=True)

    x_torch = torch.DoubleTensor(x)
    t_torch = torch.DoubleTensor(t)

    for _ in range(opt_iter):
        optimizer.zero_grad()
        x_pred = torch.sum(betas[:, None] * torch.cos(omegas[:, None]*t_torch[None, :]), dim=0)
        loss = torch.sum((x_torch - x_pred)**2)/torch.sum(x_torch**2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            omegas[:] = torch.clamp(omegas, omega_min, omega_max)
            betas[:] = torch.clamp(betas, 0, None)

    omegas = omegas.detach().numpy()
    betas = betas.detach().numpy()
    sorted_inds = np.argsort(omegas)
    omegas = omegas[sorted_inds]
    betas = betas[sorted_inds]
    return omegas, betas
import math
import numpy as np
import scipy
import torch
from torch import nn
import torch.optim as optim
from pykeops.numpy import LazyTensor

from aaa_algorithms import HilbertTransform, aaa_exp_sum
from probability import Distribution, symmetrize_dist, remove_small_masses

def mobius(theta):
    return np.real(1j*(1-np.exp(1j*theta))/(1+np.exp(1j*theta)))

def mobius_inv(x):
    z = (1j - x)/(1j + x)
    return np.arctan2(np.imag(z), np.real(z))

# alphas is a grid, b_grid indicates the continuous density part of lambda on this grid
# a, b are the atoms and weights of the discrete part of lambda
# the density of lambda does not need to be compactly supported but needs to decay at infinity
def B_real(lmbda, c0, c1, H=None, compute_mu=True, thresh=1e-15):
    assert(np.all(lmbda.quad_wts >= 0))
    assert(np.all(lmbda.atom_wts >= 0))
    
    assert(c1 >= 0)
    
    c0_re = np.real(c0)
    c0_im = np.imag(c0)
    assert(c0_im >= 0)
    
    lmbda = remove_small_masses(lmbda, thresh)
    
    mu_density = None
    alpha_disc = np.array([])
    beta_disc = np.array([])
    
    # If Hilbert transform of lmbda is not provided as a function, compute it
    if H is None and compute_mu:
        H = HilbertTransform(lmbda)
    
    lmbda_density = lambda s: 0
    if lmbda.density is not None:
        lmbda_density = lmbda.density
    
    if c0_im > 0:
        if compute_mu:
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
        if compute_mu:
            # Compute continuous part of mu
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
    
    if compute_mu:
        # Save continuous and discrete parts of mu into a distribution
        mu = Distribution(mu_density, alpha_disc, beta_disc, lmbda.quad_pts, lmbda.quad_wts)
        return mu, zeta0, zeta1
    return zeta0, zeta1

def exp_kernel(lmbda, t, deriv_order=0):
    kernel = np.zeros(len(t))
    if lmbda.num_atoms > 0:
        inds = lmbda.atom_wts != 0
        kernel += np.sum(lmbda.atom_wts[inds]*(-lmbda.atoms[inds])**deriv_order*np.exp(-lmbda.atoms[inds]*t[:, None]), axis=1)
    if lmbda.density is not None:
        inds = lmbda.density_vals != 0
        kernel += np.sum(lmbda.quad_wts[inds]*(-lmbda.quad_pts[inds])**deriv_order*lmbda.density_vals[inds]*np.exp(-lmbda.quad_pts[inds]*t[:, None]), axis=1)
    return kernel

def complex_exp_kernel(lmbda, t, deriv_order=0):
    kernel = np.zeros(len(t), dtype=np.complex128)
    if lmbda.num_atoms > 0:
        kernel += np.sum(lmbda.atom_wts*(-1j*lmbda.atoms)**deriv_order*np.exp(-1j*lmbda.atoms*t[:, None]), axis=1)
    if lmbda.density is not None:
        kernel += np.sum(lmbda.quad_wts*(-1j*lmbda.quad_pts)**deriv_order*lmbda.density_vals*np.exp(-1j*lmbda.quad_pts*t[:, None]), axis=1)
    return kernel

# Invert the Volterra equation y = c1*xdot - c0*x - K*x
# into x = zeta1*ydot - zeta0*y - J*y where K and J are bilateral Laplace transforms of lmbda and mu respectively
#def volterra_cm_spectral_inversion(lmbda, c0, c1, H=None, compute_mu=True):
#    assert(np.isreal(c0))
#    if compute_mu:
#        mu, zeta0, zeta1 = B_real(lmbda, c0, c1, H, compute_mu)
#        mu.rescale(-1/math.pi**2)
#        return mu, -zeta0/math.pi**2, -zeta1/math.pi**2
#    else:
#        zeta0, zeta1 = B_real(lmbda, c0, c1, H, compute_mu)
#        return -zeta0/math.pi**2, -zeta1/math.pi**2

# Invert the Volterra equation y = c1*xdot - c0*x + K*x
# into x = zeta1*ydot - zeta0*y + J*y where K and J are Fourier transforms of lmbda and mu respectively
#def volterra_pd_spectral_inversion(lmbda, c0, c1, H=None, compute_mu=True):
#    if compute_mu:
#        mu, zeta0, zeta1 = B_real(lmbda, c0, c1, H, compute_mu)
#        mu.rescale(1/math.pi**2)
#        return mu, zeta0/math.pi**2, zeta1/math.pi**2
#    else:
#        zeta0, zeta1 = B_real(lmbda, c0, c1, H, compute_mu)
#        return zeta0/math.pi**2, zeta1/math.pi**2

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

def fit_exp_sum(x, t, n, init="random", opt_iter=1000, lr=1e-2):
    if init == "random":
        omegas = np.tan(np.random.uniform(0, np.pi/2, n))
        betas = np.random.rand(n)
    elif init == "lsqfit":
        omegas = np.tan(np.linspace(0, np.pi/2, n))
        A = np.exp(-t[:, None]*omegas[None, :])
        betas = np.linalg.lstsq(A, x, rcond=None)[0]
        betas[betas < 0] = 0
    elif init == "aaa":
        s = np.logspace(0, 2, 1000)
        _, pol, res = aaa_exp_sum(x, t, s, aaa_iters=n, real_symm=False, max_exp=0.0)
        #assert(np.all(np.abs(np.imag(pol)) < 1e-10))
        #assert(np.all(np.abs(np.imag(res)) < 1e-10))
        omegas = np.tan(np.linspace(0, np.pi/2, n))
        betas = np.zeros(max(len(pol), n))
        omegas[:len(pol)] = np.real(pol)
        betas[:len(pol)] = np.real(res)
        betas[betas < 0] = 0
    else:
        raise ValueError("Initialization method")
    
    sorted_inds = np.argsort(omegas)
    omegas = omegas[sorted_inds]
    betas = betas[sorted_inds]
    omegas = torch.tensor(omegas, requires_grad=True, dtype=torch.float64)
    betas = torch.tensor(betas, requires_grad=True, dtype=torch.float64)
    
    optimizer = optim.Adam([omegas, betas], lr=lr)#, amsgrad=True)
    
    x_torch = torch.DoubleTensor(x)
    t_torch = torch.DoubleTensor(t)
    
    for _ in range(opt_iter):
        optimizer.zero_grad()
        x_pred = torch.sum(betas[:, None] * torch.exp(-omegas[:, None]*t_torch[None, :]), dim=0)
        loss = torch.sum(torch.abs(x_torch - x_pred)**2)/torch.sum(torch.abs(x_torch)**2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            betas[:] = torch.clamp(betas, 0, None)

    omegas = omegas.detach().numpy()
    betas = betas.detach().numpy()
    sorted_inds = np.argsort(omegas)
    omegas = omegas[sorted_inds]
    betas = betas[sorted_inds]
    return omegas, betas

def fit_cos_sum(x, t, n, init="dct", omega_init_min=None, omega_init_max=None, opt_iter=1000, lr=1e-2):
    dt = t[1] - t[0]
    tmin = np.min(t)
    tmax = np.max(t)
    
    if omega_init_min is None:
        omega_init_min = 0
    if omega_init_max is None:
        omega_init_max = math.pi/dt
    
    if init == "random":
        omegas = np.random.uniform(omega_init_min, omega_init_max, n)
        betas = np.random.rand(n)
    elif init == "lsqfit":
        omegas = np.linspace(omega_init_min, omega_init_max, n)
        A = np.cos(t[:, None]*omegas[None, :])
        betas = np.linalg.lstsq(A, x, rcond=None)[0]
        betas[betas < 0] = 0
    elif init == "dct":
        if n > len(t):
            print("DCT with n > len(t) will cause aliasing")
        if n == 1:
            omegas = np.array([2*math.pi/(tmax - tmin)])
            betas = np.array([1])
        else:
            t_new = np.linspace(tmin, tmax, n)
            dt_new = t_new[1] - t_new[0]
            x_interp = np.interp(t_new, t, x)
            omegas = math.pi*(2*np.arange(n)+1)/(2*n*dt_new)
            betas = (1/n)*scipy.fft.dct(x_interp, type=3, norm=None, orthogonalize=False)
            betas[betas < 0] = 0
    else:
        raise ValueError("Initialization method")
    
    sorted_inds = np.argsort(omegas)
    omegas = omegas[sorted_inds]
    betas = betas[sorted_inds]
    omegas = torch.tensor(omegas, requires_grad=True, dtype=torch.float64)
    betas = torch.tensor(betas, requires_grad=True, dtype=torch.float64)
    
    x_torch = torch.DoubleTensor(x)
    t_torch = torch.DoubleTensor(t)
    
    optimizer = optim.Adam([omegas, betas], lr=lr)#, amsgrad=True)

    for _ in range(opt_iter):
        optimizer.zero_grad()
        x_pred = torch.sum(betas[:, None] * torch.cos(omegas[:, None]*t_torch[None, :]), dim=0)
        loss = torch.sum((x_torch - x_pred)**2)/torch.sum(x_torch**2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            omegas[:] = torch.clamp(omegas, 0, None)
            betas[:] = torch.clamp(betas, 0, None)

    omegas = omegas.detach().numpy()
    betas = betas.detach().numpy()
    sorted_inds = np.argsort(omegas)
    omegas = omegas[sorted_inds]
    betas = betas[sorted_inds]
    return omegas, betas

def fit_complex_exp_sum(x, t, n, init="dft", omega_init_min=None, omega_init_max=None, omega_min=None, omega_max=None, opt_iter=1000, lr=1e-2):
    dt = t[1] - t[0]
    tmin = np.min(t)
    tmax = np.max(t)
    
    if omega_init_min is None:
        omega_init_min = 0
    if omega_init_max is None:
        omega_init_max = math.pi/dt
    
    if init == "random":
        omegas = np.random.uniform(omega_init_min, omega_init_max, n)
        betas = np.random.rand(n)
    elif init == "lsqfit":
        omegas = np.linspace(omega_init_min, omega_init_max, n)
        A = np.exp(-1j*t[:, None]*omegas[None, :])
        betas = np.real(np.linalg.lstsq(A, x, rcond=None)[0])
        betas[betas < 0] = 0
    elif init == "dft":
        x_interp = np.interp(np.linspace(tmin, tmax, n), t, x)
        omegas = -np.fft.fftfreq(n, d=dt/(2*math.pi))
        betas = (1/n)*np.real(np.fft.fft(x_interp, norm=None))
        betas[betas < 0] = 0
    else:
        raise ValueError("Initialization method")
    
    sorted_inds = np.argsort(omegas)
    omegas = omegas[sorted_inds]
    betas = betas[sorted_inds]
    omegas = torch.tensor(omegas, requires_grad=True, dtype=torch.float64)
    betas = torch.tensor(betas, requires_grad=True, dtype=torch.float64)
    
    x_torch = torch.DoubleTensor(x)
    t_torch = torch.DoubleTensor(t)
    
    optimizer = optim.Adam([omegas, betas], lr=lr)#, amsgrad=True)

    for _ in range(opt_iter):
        optimizer.zero_grad()
        x_pred = torch.sum(betas[:, None] * torch.exp(-1j*omegas[:, None]*t_torch[None, :]), dim=0)
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
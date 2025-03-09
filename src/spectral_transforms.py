import math
import numpy as np
import scipy
import torch
#from torch import nn
import torch.optim as optim
#from pykeops.numpy import LazyTensor

from .aaa_algorithms import aaa_exp_sum
from .hilbert_transform import HilbertTransform
from .probability import Distribution, remove_small_masses, sigma_int, xi_frac
from .numerical import trap_quad, trap_quad_gen, fourier_quad

# phi map that takes circle angle theta in [-pi, pi] to real line x in (-infty, infty)
def mobius(theta):
    return np.real(1j*(1-np.exp(1j*theta))/(1+np.exp(1j*theta)))

# phi_inv map that takes real line x in (-infty, infty) to circle angle theta in [-pi, pi]
def mobius_inv(x):
    z = (1j - x)/(1j + x)
    return np.arctan2(np.imag(z), np.real(z))

# pts are assumed to be on real line
def psi_real_to_circle(lmbda_real, c0_real, c1_real, pts=None, regularized=False):
    assert(lmbda_real.periodic_domain is None)
    
    atoms_circle = mobius_inv(lmbda_real.atoms)
    atom_wts_circle = lmbda_real.atom_wts / (math.pi * (lmbda_real.atoms**2 + 1))
    if c1_real != 0:
        atoms_circle = np.insert(atoms_circle, 0, -np.pi)
        atom_wts_circle = np.insert(atom_wts_circle, 0, c1_real/np.pi)
    
    density_circle = None if lmbda_real.density is None else lambda x: lmbda_real.density(mobius(x))
    if pts is None:
        pts = len(lmbda_real.quad_pts)
    if np.isscalar(pts):
        # if only number of quadrature points passed in, use equispaced points on the circle with equal weights
        quad_pts, quad_wts = fourier_quad(-np.pi, np.pi, pts)
    else:
        # if position of quadrature points passed in, use inverse Mobius transform of these points on the circle with trapezoid weighting
        quad_pts, quad_wts = trap_quad_gen(mobius_inv(pts))
    
    c0_circle = -c0_real/np.pi
    if not regularized:
        c0_circle += sigma_int(lmbda_real)
    
    lmbda_circle = Distribution(analytic_density=density_circle, atoms=atoms_circle, atom_wts=atom_wts_circle, quad_pts=quad_pts, quad_wts=quad_wts, zero_sets=None, full_support=lmbda_real.full_support, periodic_domain=(-np.pi, np.pi))
    return lmbda_circle, c0_circle

# pts are assumed to be on real line
def psi_inv_circle_to_real(lmbda_circle, c0_circle, pts=None, regularized=False):
    assert(lmbda_circle.periodic_domain == (-np.pi, np.pi))
    
    delta_at_minus_pi = lmbda_circle.num_atoms > 0 and np.isclose(lmbda_circle.atoms[0], -np.pi)
    delta_at_plus_pi = lmbda_circle.num_atoms > 0 and np.isclose(lmbda_circle.atoms[-1], np.pi)
    assert(~(delta_at_minus_pi*delta_at_plus_pi))
    
    atoms_circle = np.copy(lmbda_circle.atoms)
    atom_wts_real = np.copy(lmbda_circle.atom_wts)
    if delta_at_minus_pi:
        atoms_circle = np.copy(lmbda_circle.atoms[1:])
        atom_wts_real = np.copy(lmbda_circle.atom_wts[1:])
    if delta_at_plus_pi:
        atoms_circle = np.copy(lmbda_circle.atoms[:-1])
        atom_wts_real = np.copy(lmbda_circle.atom_wts[:-1])
    
    atoms_real = mobius(atoms_circle)
    atom_wts_real *= math.pi * (atoms_real**2 + 1)
    density_real = None if lmbda_circle.density is None else lambda x: lmbda_circle.density(mobius_inv(x))
    
    if pts is None:
        pts = mobius(lmbda_circle.quad_pts)
    quad_pts, quad_wts = trap_quad_gen(pts)
    lmbda_real = Distribution(analytic_density=density_real, atoms=atoms_real, atom_wts=atom_wts_real, quad_pts=quad_pts, quad_wts=quad_wts, zero_sets=None, full_support=lmbda_circle.full_support, periodic_domain=None)
    
    c0_real = -np.pi*c0_circle
    if not regularized:
        c0_real += np.pi*sigma_int(lmbda_real)
    
    c1_real = 0
    if delta_at_minus_pi:
        c1_real = np.pi*lmbda_circle.atom_wts[0]
    if delta_at_plus_pi:
        c1_real = np.pi*lmbda_circle.atom_wts[-1]
    return lmbda_real, c0_real, c1_real

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
    
    lmbda_density = lambda _: 0
    if lmbda.density is not None:
        lmbda_density = lmbda.density
    
    if c0_im > 0:
        if compute_mu:
            # Compute continuous part of mu, no discrete part
            if c1 == 0:
                mu_density = lambda s: (lmbda_density(s) + c0_im/math.pi)/((lmbda_density(s) + c0_im/math.pi)**2 + (H(s) - c0_re/math.pi)**2) - math.pi*c0_im/(c0_re**2 + c0_im**2)
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
                    assert(len(alpha_disc) == lmbda.num_atoms+1)
                    assert(np.logical_and(alpha_disc >= lmbda.zero_sets[:, 0], alpha_disc <= lmbda.zero_sets[:, 1]).all())
                    print("left and right roots found")
                elif c0_re < 0 and c1 == 0:
                    # there is one root to the left of the first atom
                    assert(len(alpha_disc) == lmbda.num_atoms)
                    assert(np.logical_and(alpha_disc >= lmbda.zero_sets[:-1, 0], alpha_disc <= lmbda.zero_sets[:-1, 1]).all())
                    print("left root found")
                elif c0_re > 0 and c1 == 0:
                    # there is one root to the right of the last atom
                    assert(len(alpha_disc) == lmbda.num_atoms)
                    assert(np.logical_and(alpha_disc >= lmbda.zero_sets[1:, 0], alpha_disc <= lmbda.zero_sets[1:, 1]).all())
                    print("right root found")
                elif c0_re == 0 and c1 == 0:
                    # there are no roots to the left of the first atom and to the right of the last atom
                    assert(len(alpha_disc) == lmbda.num_atoms-1)
                    assert(np.logical_and(alpha_disc >= lmbda.zero_sets[1:-1, 0], alpha_disc <= lmbda.zero_sets[1:-1, 1]).all())
                    print("no roots to left or right")
        
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
        mu = remove_small_masses(mu, thresh)
        return mu, zeta0, zeta1
    return zeta0, zeta1

def B_circle(lmbda, c0, H=None, compute_mu=True, thresh=1e-15):
    assert(np.isreal(c0))
    assert(np.all(lmbda.quad_wts >= 0))
    assert(np.all(lmbda.atom_wts >= 0))
    assert(lmbda.periodic_domain[0] == -np.pi)
    assert(lmbda.periodic_domain[1] == np.pi)
    
    assert(c0 != 0 or lmbda.num_atoms > 0 or (lmbda.density is not None))
    
    lmbda = remove_small_masses(lmbda, thresh)
    
    mu_density = None
    alpha_disc = np.array([])
    beta_disc = np.array([])
    
    # If Hilbert transform of lmbda is not provided as a function, compute it
    if H is None and compute_mu:
        H = HilbertTransform(lmbda)
    
    lmbda_density = lambda _: 0
    if lmbda.density is not None:
        lmbda_density = lmbda.density
    
    if compute_mu:
        # Compute continuous part of mu
        if lmbda.density is not None:
            mu_density = lambda s: lmbda_density(s)/(lmbda_density(s)**2 + (H(s) + c0)**2)
        
        # Compute discrete part of mu
        alpha_disc = H.roots(0, -c0)
        beta_disc = 1/lmbda.offset_moments(alpha_disc, -2)
        keep_inds = np.isfinite(beta_disc)
        alpha_disc = alpha_disc[keep_inds]
        beta_disc = beta_disc[keep_inds]
    
    # Compute coefficients zeta0
    zeta0 = 0
    if c0 != 0:
        m0_lmbda = lmbda.moment(0)
        zeta0 = np.imag(1/(m0_lmbda + 1j*c0))
    
    if compute_mu:
        # Save continuous and discrete parts of mu into a distribution
        mu = Distribution(mu_density, alpha_disc, beta_disc, lmbda.quad_pts, lmbda.quad_wts, periodic_domain=(-np.pi, np.pi))
        mu = remove_small_masses(mu, thresh)
        return mu, zeta0
    return zeta0

def B_reg(lmbda, c0, c1, thresh=1e-15):
    lmbda_circle, c0_circle = psi_real_to_circle(lmbda, c0, c1, lmbda.quad_pts, regularized=True)
    mu_circle, zeta0_circle = B_circle(lmbda_circle, c0_circle)
    mu, zeta0, zeta1 = psi_inv_circle_to_real(mu_circle, zeta0_circle, lmbda.quad_pts, regularized=True)
    return mu, zeta0, zeta1
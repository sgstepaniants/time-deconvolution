import numpy as np

from .aaa_algorithms import aaa, barycentric_representation

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

def symmetrize_dist(a, b, halve_center=False):
    assert(len(a) == len(b))
    if len(a) == 0:
        return a, b
    
    symm_a = np.abs(a)
    sorted_inds = np.argsort(symm_a)
    symm_a = symm_a[sorted_inds]
    symm_b = b[sorted_inds]
    
    if symm_a[0] == 0:
        symm_a = np.concatenate((-np.flip(symm_a[1:]), symm_a))
        center = b[0]/2 if halve_center else b[0]
        symm_b = np.concatenate((np.flip(symm_b[1:])/2, [center], symm_b[1:]/2))
    else:
        symm_a = np.concatenate((-np.flip(symm_a), symm_a))
        symm_b = np.concatenate((np.flip(symm_b), symm_b))/2
    return symm_a, symm_b

# allow users to pass in an analytic Hilbert transform of lambda
# allow users to depend on the aaa reconstruction of the density (with the density thresholded below some thresh)
class Distribution:
    def __init__(self, analytic_density=None, atoms=np.array([]), atom_wts=np.array([]), quad_pts=np.array([]), quad_wts=np.array([]), zero_sets=None, full_support=False, periodic_domain=None, aaa_pts=None, aaa_iters=100, use_aaa=False):
        assert(len(quad_pts) == len(quad_wts))
        assert(len(atoms) == len(atom_wts))
        
        sorted_inds = np.argsort(atoms)
        sorted_inds = sorted_inds[atom_wts[sorted_inds] != 0]
        self.atoms = atoms[sorted_inds].astype(np.float64)
        self.num_atoms = len(self.atoms)
        self.atom_wts = atom_wts[sorted_inds]
        
        sorted_inds = np.argsort(quad_pts)
        sorted_inds = sorted_inds[quad_wts[sorted_inds] != 0]
        self.quad_pts = quad_pts[sorted_inds].astype(np.float64)
        self.quad_wts = quad_wts[sorted_inds]
        assert(np.all(np.diff(self.quad_pts) > 0))
        
        self.periodic_domain = None
        if periodic_domain is not None:
            assert(np.all(self.atoms >= periodic_domain[0]))
            assert(np.all(self.atoms < periodic_domain[1]))
            assert(np.all(self.quad_pts >= periodic_domain[0]))
            assert(np.all(self.quad_pts < periodic_domain[1]))
            assert(~np.logical_xor(*np.isfinite(periodic_domain)))
            self.periodic_domain = periodic_domain
        
        self.analytic_density = analytic_density
        self.density_vals = np.array([]) if analytic_density is None else self.analytic_density(self.quad_pts)
        
        if self.analytic_density is not None and len(self.quad_pts) == 0:
            print("Density constructed, quadrature not set")
        
        if full_support and analytic_density is None:
            raise ValueError("Atomic distribution cannot have full support")
        self.full_support = full_support
        
        self.use_aaa = use_aaa and analytic_density is not None
        self.aaa_iters = aaa_iters
        self.aaa_pts = self.quad_pts if aaa_pts is None else aaa_pts
        self.aaa_params = None
        if self.use_aaa:
            # (pol, res, zer, zj, fj, wj)
            self.aaa_params = aaa(self.density, self.aaa_pts, max_terms=aaa_iters)
        
        # Define density function
        self.density = None
        if self.analytic_density is not None:
            def func(x):
                if self.use_aaa:
                    _, _, _, zj, fj, wj = self.aaa_params
                    return barycentric_representation(x, zj, fj, wj)
                return None if self.analytic_density is None else self.analytic_density(x)
            self.density = func
        
        # Compute zero sets of distribution
        self.update_zero_sets(zero_sets)
    
    #def rescale(self, scale):
    #    self.atom_wts *= scale
    #    if self.density is not None:
    #        original_density = self.density
    #        self.density = lambda x: original_density(x)*scale
    #        self.density_vals *= scale
    
    def update_zero_sets(self, zero_sets=None):
        # If full support, then lmbda is zero nowhere
        if self.full_support:
            self.zero_sets = np.zeros((0, 2))
            return
        
        # If zero_sets is specified, then define it as specified
        if zero_sets is not None:
            self.zero_sets = zero_sets
            return
        
        # If lambda is a discrete measure define zero sets of lambda to be the intervals between its atoms
        #if self.density is None and self.num_atoms > 0:
        #    if self.periodic_domain is None:
        #        zero_sets = np.zeros((self.num_atoms+1, 2))
        #        zero_sets[0, :] = [-np.inf, self.atoms[0]]
        #        zero_sets[-1, :] = [self.atoms[-1], np.inf]
        #        
        #        if self.num_atoms > 1:
        #            for i in range(self.num_atoms-1):
        #                zero_sets[i+1, :] = [self.atoms[i], self.atoms[i+1]]
        #        self.zero_sets = zero_sets
            
        #elif self.density is not None:
        
        # Define zero sets where density is zero and where the measure has no atoms
        # Assume that outside of quadrature points, density is zero
        if self.density is None:
            zero_sets = np.array([[-np.inf, np.inf]])
        else:
            pts = np.concatenate(([-np.inf, -np.inf], self.quad_pts, [np.inf, np.inf]))
            vals = np.concatenate(([1, 0], self.density_vals, [0, 1]))
            start_inds = np.where(np.diff((vals == 0).astype(int)) == 1)[0]
            end_inds = np.where(np.diff((vals == 0).astype(int)) == -1)[0]+1
            zero_sets = np.row_stack([pts[start_inds], pts[end_inds]]).T
        
        # Divide up these preliminary zero sets further based on where lmbda has atoms
        new_zero_sets = np.zeros((0, 2))
        for i in range(zero_sets.shape[0]):
            atoms_i = self.atoms[np.logical_and(self.atoms > zero_sets[i, 0], self.atoms < zero_sets[i, 1])]
            atoms_i = np.repeat(atoms_i, 2)
            atoms_i = np.insert(atoms_i, 0, zero_sets[i, 0])
            atoms_i = np.append(atoms_i, zero_sets[i, 1])
            new_zero_sets = np.concatenate((new_zero_sets, np.reshape(atoms_i, (-1, 2))))
        
        # If lmbda is periodic, wrap the last zero set around
        if self.periodic_domain is not None:
            if new_zero_sets.shape[0] == 1:
                new_zero_sets = np.array([[self.periodic_domain[0], self.periodic_domain[1]]])
            else:
                first_pt = new_zero_sets[0, 1]
                last_pt = new_zero_sets[-1, 0]
                if first_pt != self.periodic_domain[0] or last_pt != self.periodic_domain[1]:
                    per = self.periodic_domain[1] - self.periodic_domain[0]
                    new_zero_sets = np.concatenate((new_zero_sets[1:-1, :], [[last_pt, first_pt + per]]))
        self.zero_sets = new_zero_sets
    
    def update_quadrature(self, quad_pts, quad_wts, update_zero_sets=True):
        self.quad_pts = quad_pts
        self.quad_wts = quad_wts
        if self.density is not None:
            self.density_vals = self.density(self.quad_pts)
        if update_zero_sets:
            self.update_zero_sets()
    
    #def update_density(self, density, update_zero_sets=True):
    #    if density is not None:
    #        self.density = density
    #        self.density_vals = self.density(self.quad_pts)
    #    else:
    #        self.density = None
    #        self.density_vals = np.array([])
    #    if update_zero_sets:
    #        self.update_zero_sets()
    
    # change this to use AAA if given
    def moment(self, k):
        if self.use_aaa:
            print("hello")
            
        res = 0
        if self.density is not None:
            if self.periodic_domain is None:
                res += np.sum(self.quad_wts*self.density_vals*(self.quad_pts**k))
            else:
                per = self.periodic_domain[1] - self.periodic_domain[0]
                res += np.sum(self.quad_wts/per*self.density_vals*np.exp(1j*k*self.quad_pts))
        if self.num_atoms > 0:
            if self.periodic_domain is None:
                res += np.sum(self.atom_wts*(self.atoms**k))
            else:
                res += np.sum(self.atom_wts*np.exp(1j*k*self.atoms))
        return res
    
    # change this to use AAA if given
    def offset_moments(self, offset, k):
        if self.periodic_domain is not None:
            assert(k < 0)
        
        if self.use_aaa:
            print("hello")
        
        res = np.zeros_like(offset)
        if self.density is not None:
            if self.periodic_domain is None:
                res += np.sum(self.quad_wts[:, None]*self.density_vals[:, None]*((self.quad_pts[:, None] - offset[None, :])**k), axis=0)
            else:
                per = self.periodic_domain[1] - self.periodic_domain[0]
                z_pts = np.exp(1j*self.quad_pts)
                z_offset = np.exp(1j*offset)
                res += np.real(np.sum(self.quad_wts[:, None]/per*self.density_vals[:, None]*(-4*z_pts[:, None]*z_offset[None, :]*(z_pts[:, None] - z_offset[None, :])**k), axis=0))
        if self.num_atoms > 0:
            if self.periodic_domain is None:
                res += np.sum(self.atom_wts[:, None]*((self.atoms[:, None] - offset[None, :])**k), axis=0)
            else:
                z_pts = np.exp(1j*self.atoms)
                z_offset = np.exp(1j*offset)
                res += np.real(np.sum(self.atom_wts[:, None]*(-4*z_pts[:, None]*z_offset[None, :]*(z_pts[:, None] - z_offset[None, :])**k), axis=0))
        return res

def remove_small_masses(lmbda, thresh):
    density_thresh = None
    if lmbda.density is not None:
        def density_thresh(x):
            y = lmbda.density(x)
            return y * (np.abs(y) >= thresh)
    
    atoms_thresh = lmbda.atoms[lmbda.atom_wts >= thresh]
    atom_wts_thresh = lmbda.atom_wts[lmbda.atom_wts >= thresh]
    
    lmbda_thresh = Distribution(density_thresh,
                                atoms=atoms_thresh,
                                atom_wts=atom_wts_thresh,
                                quad_pts=lmbda.quad_pts,
                                quad_wts=lmbda.quad_wts,
                                zero_sets=None,
                                full_support=lmbda.full_support,
                                periodic_domain=lmbda.periodic_domain,
                                aaa_pts=lmbda.aaa_pts,
                                aaa_iters=lmbda.aaa_iters,
                                use_aaa=lmbda.use_aaa)
    return lmbda_thresh

def sigma_int(lmbda):
    res = 0
    if lmbda.periodic_domain is None:
        if lmbda.density is not None:
            res -= np.sum(lmbda.quad_wts*lmbda.density_vals*lmbda.quad_pts/(1+lmbda.quad_pts**2))/np.pi
        if lmbda.num_atoms > 0:
            res -= np.sum(lmbda.atom_wts*lmbda.atoms/(1+lmbda.atoms**2))/np.pi
        return res
    
    if lmbda.density is not None:
        res -= np.sum(lmbda.quad_wts*lmbda.density_vals / np.tan((-np.pi - lmbda.quad_pts)/2))
    if lmbda.num_atoms > 0:
        res -= np.sum(lmbda.atom_wts / np.tan((-np.pi - lmbda.atoms)/2))
    return res

def xi_frac(lmbda):
    assert(lmbda.periodic_domain is None)
    res = 0
    if lmbda.density is not None:
        res += np.nansum(lmbda.quad_wts*lmbda.density_vals/(lmbda.quad_pts*(1+lmbda.quad_pts**2)))/np.pi
    if lmbda.num_atoms > 0:
        res += np.nansum(lmbda.atom_wts*1/(lmbda.atoms*(1+lmbda.atoms**2)))/np.pi
    return res
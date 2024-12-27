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

class Distribution:
    def __init__(self, density=None, atoms=np.array([]), atom_wts=np.array([]), quad_pts=np.array([]), quad_wts=np.array([]), zero_sets=None):
        assert(len(quad_pts) == len(quad_wts))
        assert(len(atoms) == len(atom_wts))
        
        sorted_inds = np.argsort(atoms)
        self.atoms = atoms[sorted_inds]
        self.num_atoms = len(atoms)
        self.atom_wts = atom_wts[sorted_inds]
        
        assert(np.all(np.diff(quad_pts) > 0))
        self.quad_pts = quad_pts
        self.quad_wts = quad_wts
        
        self.density = density
        self.density_vals = np.array([]) if density is None else self.density(self.quad_pts)
        
        if self.density is not None and len(self.quad_pts) == 0:
            print("Density constructed, quadrature not set")
        
        self.update_zero_sets(zero_sets)
    
    def rescale(self, scale):
        self.atom_wts *= scale
        if self.density is not None:
            original_density = self.density
            self.density = lambda x: original_density(x)*scale
            self.density_vals *= scale
    
    def update_zero_sets(self, zero_sets=None):
        if zero_sets is not None:
            self.zero_sets = zero_sets
            return
        
        '''
        # If lambda is a discrete measure define zero sets of lambda to be the intervals between its atoms
        if self.density is None and self.num_atoms > 0:
            zero_sets = np.zeros((self.num_atoms+1, 2))
            zero_sets[0, :] = [-np.inf, self.atoms[0]]
            zero_sets[-1, :] = [self.atoms[-1], np.inf]
            if self.num_atoms > 1:
                for i in range(self.num_atoms-1):
                    zero_sets[i+1, :] = [self.atoms[i], self.atoms[i+1]]
        elif self.density is not None:
        '''
        # Define zero sets where density is zero and where the measure has no atoms
        # Assume that outside of quadrature points, density is zero
        pts = np.array([]) if self.density is None else self.quad_pts
        vals = np.array([]) if self.density is None else self.density_vals
        pts = np.concatenate([pts, self.atoms[self.atom_wts != 0], [-np.inf, np.inf]])
        vals = np.concatenate([self.density_vals, np.ones(np.sum(self.atom_wts != 0)), [0, 0]])
        
        sorted_inds = np.argsort(pts)
        pts = pts[sorted_inds]
        vals = vals[sorted_inds]
        
        start_inds = np.where(np.diff((vals == 0).astype(int)) > 0)[0]
        end_inds = np.where(np.diff((vals == 0).astype(int)) < 0)[0]+1
        
        start_inds = np.insert(start_inds, 0, 0)
        end_inds = np.append(end_inds, len(pts)-1)
        self.zero_sets = np.row_stack([pts[start_inds], pts[end_inds]]).T
    
    def update_quadrature(self, quad_pts, quad_wts, update_zero_sets=True):
        self.quad_pts = quad_pts
        self.quad_wts = quad_wts
        if self.density is not None:
            self.density_vals = self.density(self.quad_pts)
        if update_zero_sets:
            self.update_zero_sets()
    
    def update_density(self, density, update_zero_sets=True):
        if density is not None:
            self.density = density
            self.density_vals = self.density(self.quad_pts)
        else:
            self.density = None
            self.density_vals = np.array([])
        if update_zero_sets:
            self.update_zero_sets()
    
    def moment(self, k):
        res = 0
        if self.density is not None:
            res += np.sum(self.quad_wts*self.density_vals*(self.quad_pts**k))
        if self.num_atoms > 0:
            res += np.sum(self.atom_wts*(self.atoms**k))
        return res
    
    def offset_moments(self, offset, k):
        res = np.zeros_like(offset)
        if self.density is not None:
            res += np.sum(self.quad_wts[:, None]*self.density_vals[:, None]*((self.quad_pts[:, None] - offset[None, :])**k), axis=0)
        if self.num_atoms > 0:
            res += np.sum(self.atom_wts[:, None]*((self.atoms[:, None] - offset[None, :])**k), axis=0)
        return res

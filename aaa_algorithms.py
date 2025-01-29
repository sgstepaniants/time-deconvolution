import math
import numpy as np
import scipy

# Find root of func(x) in interval [a, b]
def rescale_rootfind(func, interval, root_tol=1e-15):
    a, b = interval
    assert(a <= b)
    if a == b:
        return a if np.abs(func(a)) < root_tol else None
    elif np.isfinite(b - a):
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

# AAA rational approximation, taken from Scipy documentation and Chebfun
def aaa(f, z, rtol=None, max_terms=99):
    M = np.size(z)
    mask = np.ones(M, dtype=np.bool_)
    dtype = np.result_type(z, f, 1.0)
    rtol = np.finfo(dtype).eps**0.75 if rtol is None else rtol
    atol = rtol * np.linalg.norm(f, ord=np.inf)
    zj = np.empty(max_terms, dtype=dtype)
    fj = np.empty(max_terms, dtype=dtype)
    # Cauchy matrix
    C = np.empty((M, max_terms), dtype=dtype)
    # Loewner matrix
    A = np.empty((M, max_terms), dtype=dtype)
    errors = np.empty(max_terms, dtype=A.real.dtype)
    R = np.repeat(np.mean(f), M)

    # AAA iteration
    for m in range(max_terms):
        # Introduce next support point
        # Select next support point
        jj = np.argmax(np.abs(f[mask] - R[mask]))
        # Update support points
        zj[m] = z[mask][jj]
        # Update data values
        fj[m] = f[mask][jj]
        # Next column of Cauchy matrix
        # Ignore errors as we manually interpolate at support points
        with np.errstate(divide="ignore", invalid="ignore"):
            C[:, m] = 1 / (z - z[mask][jj])
        # Update mask
        mask[np.nonzero(mask)[0][jj]] = False
        # Update Loewner matrix
        # Ignore errors as inf values will be masked out in SVD call
        with np.errstate(invalid="ignore"):
            A[:, m] = (f - fj[m]) * C[:, m]

        # Compute weights
        rows = mask.sum()
        if rows >= m + 1:
            # The usual tall-skinny case
            _, s, V = scipy.linalg.svd(
                A[mask, : m + 1], full_matrices=False, check_finite=False,
            )
            # Treat case of multiple min singular values
            mm = s == np.min(s)
            # Aim for non-sparse weight vector
            wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
        else:
            # Fewer rows than columns
            V = scipy.linalg.null_space(A[mask, : m + 1])
            nm = V.shape[-1]
            # Aim for non-sparse wt vector
            wj = V.sum(axis=-1) / np.sqrt(nm)

        # Compute rational approximant
        # Omit columns with `wj == 0`
        i0 = wj != 0
        # Ignore errors as we manually interpolate at support points
        with np.errstate(invalid="ignore"):
            # Numerator
            N = C[:, : m + 1][:, i0] @ (wj[i0] * fj[: m + 1][i0])
            # Denominator
            D = C[:, : m + 1][:, i0] @ wj[i0]
        # Interpolate at support points with `wj !=0`
        D_inf = np.isinf(D) | np.isnan(D)
        D[D_inf] = 1
        N[D_inf] = f[D_inf]
        R = N / D

        # Check if converged
        max_error = np.linalg.norm(f - R, ord=np.inf)
        errors[m] = max_error
        if max_error <= atol:
            break

    if m == max_terms - 1:
        print (f"AAA failed to converge within {max_terms} iterations.")

    # Trim off unused array allocation
    zj = zj[:m+1]
    fj = fj[:m+1]

    # Remove support points with zero weight
    i_non_zero = wj != 0
    zj = zj[i_non_zero]
    fj = fj[i_non_zero]
    wj = wj[i_non_zero]
    
    # Compute poles
    B = np.eye(len(wj) + 1, dtype=dtype)
    B[0, 0] = 0

    E = np.zeros_like(B, dtype=dtype)
    E[0, 1:] = wj
    E[1:, 0] = 1
    np.fill_diagonal(E[1:, 1:], zj)

    pol = scipy.linalg.eigvals(E, B)
    pol = pol[np.isfinite(pol)]
    
    # Compute residues
    N = (1/(np.subtract.outer(pol, zj))) @ (fj * wj)
    Ddiff = -((1/np.subtract.outer(pol, zj))**2) @ wj
    res = N / Ddiff
    
    # Compute zeros
    E = np.zeros_like(B, dtype=dtype)
    E[0, 1:] = wj*fj
    E[1:, 0] = 1
    np.fill_diagonal(E[1:, 1:], zj)

    zer = scipy.linalg.eigvals(E, B)
    zer = zer[np.isfinite(zer)]
    return pol, res, zer, zj, fj, wj

# Accepts a function u, performs AAA rational approximation on the interval [a, b] with N points,
# and extends u to the entire complex plane
def analytic_continuation(us, zs, aaa_iters=100):
    pol, res, _, _, _, _ = aaa(us, zs, max_terms=aaa_iters)
    f = lambda x: np.sum(res/(x[:, None] - pol), axis=1)
    return f, pol, res

def expE1(x):
    y = np.exp(x) * scipy.special.exp1(x)
    
    a1 = 8.5733287401
    a2 = 18.0590169730
    a3 = 8.6347608925
    a4 = 0.2677737343
    b1 = 9.5733223454
    b2 = 25.6329561486
    b3 = 21.0996530827
    b4 = 3.9584969228
    xf = x[~np.isfinite(y)]
    y[~np.isfinite(y)] = (xf**4 + a1*xf**3 + a2*xf**2 + a3*xf + a4)/(xf**5 + b1*xf**4 + b2*xf**3 + b3*xf**2 + b4*xf)
    return y

# us is a function u(z) sampled at the points zs
# real_symm encodes whether u is a real function sampled on the real axis
def laplace_transform(us, zs, aaa_iters=100, real_symm=True, max_exp=0):
    pol, res, _, _, _, _ = aaa(us*np.exp(-zs*max_exp), zs, max_terms=aaa_iters)
    if real_symm:
        def Lu(s):
            arr = expE1(-(s-max_exp)[:, None]*pol) @ res
            return (arr + arr.conj())/2
    else:
        Lu = lambda s: expE1(-(s-max_exp)[:, None]*pol) @ res
    return Lu

# Lus is a Laplace transform of u(z) sampled at the points zs
# real_symm encodes whether u is a real function sampled on the real axis
def inverse_laplace_transform(Lus, zs, aaa_iters=100, real_symm=True, max_exp=0):
    pol, res, _, _, _, _ = aaa(Lus, zs, max_terms=aaa_iters)
    pol = -pol
    res = res[np.real(pol) >= -max_exp]
    pol = pol[np.real(pol) >= -max_exp]
    if real_symm:
        res = np.append(res, res.conj())/2
        pol = np.append(pol, pol.conj())
    u = lambda t: np.exp(-t[:, None]*pol) @ res
    return u, pol, res

# us is a function u(z) sampled at the points zs
# real_symm encodes whether u is a real function sampled on the real axis
# Returns a complex exponential sum that approximates u(z)
def aaa_exp_sum(us, zs, s, aaa_iters=100, real_symm=True, max_exp=0):
    Lu = laplace_transform(us, zs, aaa_iters=aaa_iters, real_symm=real_symm, max_exp=max_exp)
    return inverse_laplace_transform(Lu(s), s, aaa_iters=aaa_iters, real_symm=real_symm, max_exp=max_exp)

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
            self.H_disc = lambda x: 1/math.pi*np.sum(self.lmbda.atom_wts[None, :] / (x[:, None] - self.lmbda.atoms[None, :]), axis=1)
        
        # Continuous part of Hilbert transform (of continuous density of lambda), using AAA rational approximation
        self.H_cont = lambda x: np.zeros_like(x)
        if self.lmbda.density is not None:
            self.H_cont = continuous_hilbert_transform(self.lmbda.density(self.aaa_samples), self.aaa_samples, self.aaa_iters)
    
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
        def func(x):
            return self(x) - m*x - c
        
        if self.lmbda.zero_sets is None:
            return np.array([])
        # each lambda zero set should have at most one root since the Hilbert transform is monotonic there
        roots = np.array([])
        for i in range(self.lmbda.zero_sets.shape[0]):
           root = rescale_rootfind(func, self.lmbda.zero_sets[i, :])
           if root is None and self.lmbda.zero_sets[i, 0] == self.lmbda.zero_sets[i, 1]:
               root = self.lmbda.zero_sets[i, 0]
           if root is not None:
               roots = np.append(roots, root)
        return roots
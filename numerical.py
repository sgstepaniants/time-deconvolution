import math
import numpy as np
import scipy
from scipy.integrate import solve_ivp
#from idesolver import IDESolver
#from symengine import Symbol, Function
#from jitcdde import jitcdde, y as y_sym, t as t_sym, quadrature

import spectral_transforms
from kernels import exp_kernel, complex_exp_kernel

def trap_quad(a, b, n):
    assert(n > 1)
    assert(a < b)
    quad_pts = np.linspace(a, b, n)
    quad_wts = (b-a)/(n-1)*np.ones(n)
    quad_wts[[0, -1]] /= 2
    return quad_pts, quad_wts

def fourier_quad(a, b, n):
    assert(n > 1)
    assert(a < b)
    quad_pts = np.linspace(a, a+(b-a)*(n-1)/n, n)
    quad_wts = np.ones(n) * (b-a)/n
    return quad_pts, quad_wts

def leggauss_quad(a, b, n):
    assert(n > 1)
    assert(a < b)
    quad_pts, quad_wts = np.polynomial.legendre.leggauss(n)
    quad_pts = (b-a)/2*(quad_pts+1) + a
    quad_wts *= (b-a)/2
    return quad_pts, quad_wts

# time convolution with trapezoid rule, assuming output starts at 0
def conv_trap(K, x, t):
    Nt = len(t)
    dt = t[1] - t[0]
    y = scipy.signal.convolve(np.pad(K, (Nt-1, 0)), x, mode='valid')*dt
    y -= (K[0]*x/2 + x[0]*K/2)*dt
    return y

# Solve y = c0x + K*x for x, expects that y(0) is close to or exactly 0
# Reproduced from https://inteq.readthedocs.io/en/latest/volterra/index.html
def solve_volterra(K, c0, y, t, method="righthand"):
    assert(t[0] == 0)
    Nt = len(t)
    dt = t[1] - t[0]
    
    Kmat = scipy.linalg.toeplitz(K, np.zeros(Nt))
    if method == "lefthand":
        np.fill_diagonal(Kmat, 0)
        if c0 != 0:
            x = scipy.linalg.solve_triangular(c0*np.eye(Nt) + Kmat*dt, y, lower=True, check_finite=False)
        else:
            # As built, Kmat is singular, take everything past first row since we expect that y(0) = 0
            Kmat = Kmat[1:, :-1]
            x = scipy.linalg.solve_triangular(c0*np.eye(Nt-1) + Kmat*dt, y[1:], lower=True, check_finite=False)
            x = np.append(x, x[-1])
    elif method == "righthand":
        Kmat[:, 0] = 0
        if c0 != 0:
            x = scipy.linalg.solve_triangular(c0*np.eye(Nt) + Kmat*dt, y, lower=True, check_finite=False)
        else:
            # As built, Kmat is singular, take everything past first column and row since we expect that x(0) = y(0) = 0
            Kmat = Kmat[1:, 1:]
            x = scipy.linalg.solve_triangular(c0*np.eye(Nt-1) + Kmat*dt, y[1:], lower=True, check_finite=False)
            x = np.insert(x, 0, x[0])
    elif method == "trapezoid":
        inds = np.arange(1, Nt)
        Kmat[inds, inds] /= 2
        Kmat[1:, 0] /= 2
        if c0 != 0:
            x = scipy.linalg.solve_triangular(c0*np.eye(Nt) + Kmat*dt, y, lower=True, check_finite=False)
        else:
            # As built, Kmat is not singular, but this numerical trick improves stability of the trapezoid rule by enforcing x(0) = x(1) instead of x(0) = 0
            Kmat = Kmat[1:, 1:]
            Kmat[:, 0] += K[1:]/2
            x = scipy.linalg.solve_triangular(c0*np.eye(Nt-1) + Kmat*dt, y[1:], lower=True, check_finite=False)
            x = np.insert(x, 0, x[0])
    else:
        raise ValueError("Method must be lefthand, righthand, or trapezoid rule.")
    
    return x

'''
# Solve y = c1xdot + c0x + K*x, x(0) = x0 for x
# K and y must be functions of one variable, c0, c1, and x0 are scalars
def solve_volterra_integrodiff(K, c0, c1, y, x0, t, method="iterative", maxiters=100, nsteps=20):
    assert(c1 != 0)
    
    if method == "iterative":
        solver = IDESolver(
            x = t,
            y_0 = x0+0j,
            c = lambda x, z: (y(x)-c0*z)/c1,
            d = lambda x: -1/c1,
            k = lambda x, s: K(np.array([x-s])),
            f = lambda y: y,
            lower_bound = lambda x: 0,
            upper_bound = lambda x: x,
            max_iterations = maxiters,
        )
        solver.solve()
        x = solver.y
    elif method == "integration":
        τ = Symbol("τ")
        
        k_re = Function("k_re")
        k_re_callback = lambda _, t: np.real(K(np.array([t]))/c1)
        
        k_im = Function("k_im")
        k_im_callback = lambda _, t: np.imag(K(np.array([t]))/c1)

        f_re = Function("f_re")
        f_re_callback = lambda _, t: np.real(y(t)/c1)
        f_im = Function("f_im")
        f_im_callback = lambda _, t: np.imag(y(t)/c1)
        
        c01_re = np.real(c0/c1)
        c01_im = np.imag(c0/c1)
        
        equation = [
            f_re(t_sym) - c01_re*y_sym(0, t_sym) + c01_im*y_sym(1, t_sym) - quadrature(k_re(t_sym-τ)*y_sym(0,τ)-k_im(t_sym-τ)*y_sym(1,τ), τ, 0, t_sym, nsteps=nsteps, method='gauss'),
            f_im(t_sym) - c01_im*y_sym(0, t_sym) - c01_re*y_sym(1, t_sym) - quadrature(k_re(t_sym-τ)*y_sym(1,τ)+k_im(t_sym-τ)*y_sym(0,τ), τ, 0, t_sym, nsteps=nsteps, method='gauss')]
        dde = jitcdde(equation, max_delay=t[-1], callback_functions = [(k_re, k_re_callback, 1), (k_im, k_im_callback, 1), (f_re, f_re_callback, 1), (f_im, f_im_callback, 1)])
        dde.constant_past([np.real(x0), np.imag(x0)])
        
        x = np.zeros((2, len(t)))
        for i in range(len(t)):
            x[:, i] = dde.integrate(t[i])
        x = x[0, :] + 1j*x[1, :]
    else:
        raise ValueError("Method must be iterative or integration.")
    
    return x
'''

# Solve y = c1xdot + c0x + K*x, x(0) = x0 for x
# K and y must be functions of one variable, c0, c1, and x0 are scalars
def solve_volterra_integrodiff(K, c0, c1, y, x0, t, Tmax = 10, max_step=1e-2, gauss_nodes=20):
    if not callable(K):
        Kf = lambda tmpt: np.interp(tmpt, t, K)
    else:
        Kf = K
    if not callable(y):
        #print(t.shape)
        #print(t.dtype)
        #print(y.shape)
        #print(y.dtype)
        yf = lambda tmpt: np.interp(tmpt, t, y)
    else:
        yf = y
    
    def integral_equation(t, x, t_past, x_past, c0, c1, yf):
        t_past.append(t)
        x_past.append(x[0])
        t_past = np.array(t_past)
        x_past = np.array(x_past)
        
        conv = 0
        if len(t_past) > 1:
            tmpts = t_past[t_past >= t-Tmax]
            kernel = Kf(t - tmpts)
            past = x_past[t_past >= t-Tmax]
            
            func = lambda p: np.interp(p, tmpts, past*kernel)
            conv = scipy.integrate.fixed_quad(func, tmpts[0], tmpts[-1], n=min(len(tmpts), gauss_nodes))[0]
        dydt = (-c0*x - conv + yf(t))/c1

        return dydt

    # Initialize a list to store past values
    t_past = []
    x_past = []
    
    if np.isscalar(x0):
        x0 = np.array([x0])

    sol = solve_ivp(integral_equation, [t[0], t[-1]], x0, method='RK45', t_eval=t, args=(t_past, x_past, c0, c1, yf), max_step=max_step)
    return sol.y[0]

# Invert the Volterra equation y = c1*xdot - c0*x - K*x
# into x = zeta1*ydot - zeta0*y - J*y where K and J are bilateral Laplace transforms of lmbda and mu respectively
def volterra_cm_numerical_inversion(lmbda, c0, c1, t, method=None, max_step=1e-2, gauss_nodes=20):
    method = "trapezoid" if c1 == 0 and method is None else "gauss integration"
    
    print(f"Solving numerically with {method} method")
    
    zeta0, zeta1 = spectral_transforms.B_real(lmbda, c0, c1, H=None, compute_mu=False)
    if c1 > 0:
        # Solve integrodifferential Volterra equation c1*Jdot = c0*J + K*J, J(0) = pi^2/c1
        K = lambda t: -exp_kernel(lmbda, t)
        y = lambda t: 0
        J = solve_volterra_integrodiff(K, -c0, c1, y, math.pi**2/c1, t, max_step=max_step, gauss_nodes=gauss_nodes)
    elif c0 != 0:
        K = exp_kernel(lmbda, t)
        # Solve Volterra equation of second type -zeta0*K = c0*J + K*J
        J = solve_volterra(K, c0, -zeta0*K, t, method)
    else:
        K = exp_kernel(lmbda, t)
        Kdot = exp_kernel(lmbda, t, 1)
        # Solve Volterra equation of first type zeta1*Kdot - zeta0*K = K*J
        J = solve_volterra(K, 0, zeta1*Kdot-zeta0*K, t, method)
    return J, zeta0, zeta1

def volterra_cm_resolvent_eq_error(K, c0, c1, J, zeta0, zeta1, t):
    Kdot = np.gradient(K, t)
    Jdot = np.gradient(J, t)
    KJconv = conv_trap(K, J, t)
    if c1 > 0:
        return np.linalg.norm(c1*Jdot - c0*J - KJconv) / np.linalg.norm(c1*Jdot)
    elif c0 != 0:
        return np.linalg.norm(zeta0*K + c0*J + KJconv) / np.linalg.norm(zeta0*K)
    else:
        return np.linalg.norm(zeta1*Kdot - zeta0*K - KJconv) / np.linalg.norm(zeta1*Kdot - zeta0*K)

# Invert the Volterra equation y = c1*xdot - c0*x + K*x
# into x = zeta1*ydot - zeta0*y + J*y where K and J are Fourier transforms of lmbda and mu respectively
def volterra_pd_numerical_inversion(lmbda, c0, c1, t, method=None, max_step=1e-2, gauss_nodes=20):
    if method is None:
        method = "trapezoid" if c1 == 0 and method is None else "gauss integration"
    
    zeta0, zeta1 = spectral_transforms.B_real(lmbda, c0, c1, H=None, compute_mu=False)
    if c1 > 0:
        # Solve integrodifferential Volterra equation c1*Jdot = ic0*J - K*J, J(0) = pi^2/c1
        K = lambda t: complex_exp_kernel(lmbda, t)
        y = lambda t: 0
        J = solve_volterra_integrodiff(K, -1j*c0, c1, y, math.pi**2/c1, t, max_step=max_step, gauss_nodes=gauss_nodes)
    elif c0 != 0:
        # Solve Volterra equation of second type izeta0*K = -ic0*J + K*J
        K = complex_exp_kernel(lmbda, t)
        J = solve_volterra(K, -1j*c0, 1j*zeta0*K, t, method)
    else:
        # Solve Volterra equation of first type izeta0*K - zeta1*Kdot = K*J
        K = complex_exp_kernel(lmbda, t)
        Kdot = complex_exp_kernel(lmbda, t, 1)
        J = solve_volterra(K, 0, 1j*zeta0*K-zeta1*Kdot, t, method)
    return J, zeta0, zeta1

def volterra_pd_resolvent_eq_error(K, c0, c1, J, zeta0, zeta1, t):
    Kdot = np.gradient(K, t)
    Jdot = np.gradient(J, t)
    KJconv = conv_trap(K, J, t)
    if c1 > 0:
        return np.linalg.norm(c1*Jdot - 1j*c0*J + KJconv) / np.linalg.norm(c1*Jdot)
    elif c0 != 0:
        return np.linalg.norm(1j*zeta0*K + 1j*c0*J - KJconv) / np.linalg.norm(1j*zeta0*K)
    else:
        return np.linalg.norm(1j*zeta0*K - zeta1*Kdot - KJconv) / np.linalg.norm(1j*zeta0*K - zeta1*Kdot)
import numpy as np

def trap_quad(a, b, n):
    assert(n > 1)
    assert(a < b)
    quad_pts = np.linspace(a, b, n)
    quad_wts = (b-a)/(n-1)*np.ones(n)
    quad_wts[[0, -1]] /= 2
    return quad_pts, quad_wts

def leggauss_quad(a, b, n):
    assert(n > 1)
    assert(a < b)
    quad_pts, quad_wts = np.polynomial.legendre.leggauss(n)
    quad_pts = (b-a)/2*(quad_pts+1) + a
    quad_wts *= (b-a)/2
    return quad_pts, quad_wts
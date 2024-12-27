import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

from quadrature import *
from probability import *

quad_pts, quad_wts = trap_quad(-20, 20, 2000)

density_1 = None
atoms_1 = np.array([-7.5, -math.pi, 1, 2.5, 5.32, 7.2])
atom_wts_1 = np.array([0.1, 2.3, 0.5, 0.7, 0.36, 0.4])
lmbda_1 = Distribution(None, atoms_1, atom_wts_1, quad_pts, quad_wts)

density_2 = lambda x: np.exp(-np.abs(x))
atoms_2 = np.array([])
atom_wts_2 = np.array([])
lmbda_2 = Distribution(density_2, atoms_2, atom_wts_2, quad_pts, quad_wts)

def density_3(x):
    y1 = 1 - (x - 3)**2
    y1 = y1 * (y1 > 0)
    y2 = 1 - (x + 3)**2
    y2 = y2 * (y2 > 0)
    return y1 + y2
lmbda_3 = Distribution(density_3, np.array([]), np.array([]), quad_pts, quad_wts)

def density_4(x):
    y1 = np.cos(x - 3)**2
    y1 = y1 * (y1 > 0)
    y2 = np.cos(x + 3)**2
    y2 = y2 * (y2 > 0)
    return y1 + y2
atoms_4 = np.array([-2, -1, 0, 1, 2])
atom_wts_4 = np.array([0.5, 1.2, 0.2, 1.32, 0.85])
lmbda_4 = Distribution(density_4, atoms_4, atom_wts_4, quad_pts, quad_wts)

quad_pts, quad_wts = trap_quad(-30, 30, 1000)
def density_6(x):
    y = np.exp(-(x+10)**2/10) + np.exp(-(x - 10)**2/10)
    y *= y > 0.1
    return y
atoms_6 = np.array([-2, 2])
atom_wts_6 = np.array([1, 1])
lmbda = Distribution(density_6, atoms_6, atom_wts_6, quad_pts, quad_wts)

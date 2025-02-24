import math
import numpy as np
from matplotlib import pyplot as plt

def plot_real_distribution(ax, lmbda, color="blue"):
    if lmbda.density is not None:
        ax.plot(lmbda.quad_pts, lmbda.density_vals, color=color, linewidth=1)
    ax.scatter(lmbda.atoms, lmbda.atom_wts, marker='o', color=color, s=12)
    ax.vlines(lmbda.atoms, ymin=0, ymax=lmbda.atom_wts, color=color, linewidth=1)

def plot_circle_distribution(ax, lmbda, width = 2*np.pi, atom_scale = 1, density_scale = 1, color="blue"):
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    if lmbda.density is not None:
        ax.plot((density_scale*lmbda.density_vals+1)*np.cos(lmbda.quad_pts), (density_scale*lmbda.density_vals+1)*np.sin(lmbda.quad_pts), color=color, linewidth=1)
    for i in range(lmbda.num_atoms):
        ax.plot(np.array([1, atom_scale*lmbda.atom_wts[i]+1])*np.cos(lmbda.atoms[i]), np.array([1, atom_scale*lmbda.atom_wts[i]+1])*np.sin(lmbda.atoms[i]), color=color, linewidth=1)
        ax.scatter((atom_scale*lmbda.atom_wts[i]+1)*np.cos(lmbda.atoms[i]), (atom_scale*lmbda.atom_wts[i]+1)*np.sin(lmbda.atoms[i]), color=color, s=12)
    ax.add_patch(circle)
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.set_aspect('equal')
import math
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

def plot_real_distribution(ax, lmbda, atom_scale = lambda x: x, density_scale = lambda x: x, color="blue", linewidth=1, s=12, density_label=None, atomic_label=None):
    if lmbda.density is not None:
        ax.plot(lmbda.quad_pts, density_scale(lmbda.density_vals), color=color, linewidth=linewidth, label=density_label)
    ax.scatter(lmbda.atoms, atom_scale(lmbda.atom_wts), marker='o', color=color, s=s, label=atomic_label)
    ax.vlines(lmbda.atoms, ymin=0, ymax=atom_scale(lmbda.atom_wts), color=color, linewidth=linewidth)

def plot_circle_distribution(ax, lmbda, width = 2*np.pi, atom_scale = lambda x: x, density_scale = lambda x: x, color="blue", linewidth=1, s=12):
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    if lmbda.density is not None:
        x = (density_scale(lmbda.density_vals)+1)*np.cos(lmbda.quad_pts)
        y = (density_scale(lmbda.density_vals)+1)*np.sin(lmbda.quad_pts)
        x = np.insert(x, 0, -1)
        y = np.insert(y, 0, 0)
        ax.plot(x, y, color=color, linewidth=linewidth)
    for i in range(lmbda.num_atoms):
        ax.plot(np.array([1, atom_scale(lmbda.atom_wts[i])+1])*np.cos(lmbda.atoms[i]), np.array([1, atom_scale(lmbda.atom_wts[i])+1])*np.sin(lmbda.atoms[i]), color=color, linewidth=linewidth)
        ax.scatter((atom_scale(lmbda.atom_wts[i])+1)*np.cos(lmbda.atoms[i]), (atom_scale(lmbda.atom_wts[i])+1)*np.sin(lmbda.atoms[i]), color=color, s=s)
    ax.add_patch(circle)
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.set_aspect('equal')

def plot_circle_distribution_3D(ax, lmbda, width = 2*np.pi, atom_scale = lambda x: x, density_scale = lambda x: x, color="blue", frac_fill=0.1, linewidth=1, s=12):
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    if lmbda.density is not None:
        x = np.cos(lmbda.quad_pts)
        y = np.sin(lmbda.quad_pts)
        z = density_scale(lmbda.density_vals)
        x = np.insert(x, 0, -1)
        y = np.insert(y, 0, 0)
        z = np.insert(z, 0, 0)
        ax.plot(x, y, z, color=color, linewidth=linewidth)
        
        dtheta = np.diff(lmbda.quad_pts)[0]
        for i in range(0, len(lmbda.quad_pts), int(frac_fill/dtheta)):
            ax.plot(np.array([1, 1])*x[i], np.array([1, 1])*y[i], np.array([0, z[i]]), color=color, linewidth=linewidth, alpha=0.3)
    for i in range(lmbda.num_atoms):
        ax.plot(np.array([1, 1])*np.cos(lmbda.atoms[i]), np.array([1, 1])*np.sin(lmbda.atoms[i]), np.array([0, atom_scale(lmbda.atom_wts[i])]), color=color, linewidth=linewidth)
        ax.scatter(np.cos(lmbda.atoms[i]), np.sin(lmbda.atoms[i]), atom_scale(lmbda.atom_wts[i]), color=color, s=s)
    p = ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.set_aspect('equal')
    
def plot_discrete(ax, t, x, color="blue", linewidth=1, linestyle='-'):
    for i in range(len(t)):
        ax.plot([t[i]-0.5, t[i]+0.5], [x[i], x[i]], color=color, linewidth=linewidth, linestyle=linestyle)
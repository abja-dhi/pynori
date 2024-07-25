import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xarray as xr
import mikeio
import importlib.resources as pkg_resources
import pynori.data
from scipy.interpolate import griddata
from pyplume.plotting.matplotlib_shell import subplots
from scipy.spatial import KDTree



def smooth1D(Y, lam):
    m, _ = Y.shape
    E = np.eye(m)
    D1 = np.diff(E, axis=0)
    D2 = np.diff(D1, axis=0)
    P = lam**2 * D2.T @ D2 + 2 * lam * D1.T @ D1
    Z = np.linalg.solve(E + P, Y)
    return Z



def scatter(X, Y, X_bins, Y_bins, density_power=0.5, lam=0, fig=None, ax=None):
    bin = (X_bins[1] - X_bins[0]) / 10
    nbin = (X_bins[-1] - X_bins[0]) / bin
    edges1 = np.interp(np.arange(0, len(X_bins)-1+0.01, 0.1), np.arange(len(X_bins)), X_bins)
    edges2 = np.interp(np.arange(0, len(Y_bins)-1+0.01, 0.1), np.arange(len(Y_bins)), Y_bins)
    ctrs1 = edges1[0:-1] + 0.5 * np.diff(edges1)
    ctrs2 = edges2[0:-1] + 0.5 * np.diff(edges2)


    F, xedges, yedges = np.histogram2d(X, Y, bins=[edges1, edges2])
    F = np.power(F, density_power)
    F = smooth1D(F, lam)
    
    with pkg_resources.open_text(pynori.data, 'DHI_ColorMap.csv') as f:
        colormap_matrix = np.loadtxt(f, delimiter=',')
    colormap = ListedColormap(colormap_matrix, name="DHI")
    [x, y] = np.meshgrid(ctrs1, ctrs2)
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    F_flatten = F.transpose().flatten()
    XY = np.column_stack((X, Y))
    C = griddata((x_flatten, y_flatten), F_flatten, XY, method='cubic', fill_value=1)
    ind = np.fix((C - np.min(C)) / (np.max(C) - np.min(C)) * (colormap_matrix.shape[0] - 1))
    if fig is None or ax is None:
        fig, ax = subplots()
    for i in range(len(colormap_matrix)):
        ax.scatter(X[ind == i], Y[ind == i], color=colormap_matrix[i, :], s=25, edgecolors='none', marker='o')
    ax.set_xlim([X_bins[0], X_bins[-1]])
    ax.set_ylim([Y_bins[0], Y_bins[-1]])
    pairs = {3000: 1, 30000: 10, 300000: 100, 3000000: 1000, 30000000: 10000}
    for i in range(len(pairs)):
        if len(X) < list(pairs.keys())[i]:
            d = list(pairs.values())[i]
    if i != 0:
        d = list(pairs.values())[i-1]
    C = np.power(C, 1/density_power)
    # Add colorbar
    d_scale = np.max(np.round((np.max(C) - np.max(np.min(C)))/d)*d/10)
    #cbar = fig.colorbar(ax.collections[0])
    #cbar.set_label('Density')

    return fig, ax
    


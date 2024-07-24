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


def scatter(X, Y, X_bins, Y_bins, density_power=0.5, lam=0):
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
    plt.register_cmap(cmap=colormap, name="DHI")
    plt.set_cmap("DHI")
    [x, y] = np.meshgrid(ctrs1, ctrs2)
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    F_flatten = F.transpose().flatten()
    XY = np.column_stack((X, Y))
    C = griddata((x_flatten, y_flatten), F_flatten, XY, method='cubic', fill_value=1)
    ind = np.fix((C - np.min(C)) / (np.max(C) - np.min(C)) * (colormap_matrix.shape[0] - 1))
    fig, ax = subplots()
    for i in range(len(colormap_matrix)):
        ax.scatter(X[ind == i], Y[ind == i], c=colormap_matrix[i, :], s=5, edgecolors='none', marker='o')
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
    print(d_scale)
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label('Density')

    return fig, ax
    

def find_elements_within_radius(array, point, radius):
    distances_squared = np.sum((array - point) ** 2, axis=1)
    within_radius_indices = np.where(distances_squared <= radius ** 2)[0]
    return within_radius_indices

def find_elements_within_ellipsoid(elements, center, horizontal_radius, vertical_to_horizontal_resolution):
    cx, cy, cz = center
    vertical_radius = horizontal_radius * vertical_to_horizontal_resolution
    
    # Vectorized calculation of normalized distances
    normalized_distances = np.sqrt(
        ((elements[:, 0] - cx) ** 2 / horizontal_radius ** 2) +
        ((elements[:, 1] - cy) ** 2 / horizontal_radius ** 2) +
        ((elements[:, 2] - cz) ** 2 / vertical_radius ** 2)
    )
    
    found_elements = np.where(normalized_distances <= 1)[0]
    return found_elements

def find_n_nearest_points(observ, model, row_index, n):
    # Extract the row from the observ array
    point = observ[row_index]

    # Create a KDTree for the model array
    tree = KDTree(model)

    # Query the tree for the n nearest neighbors to the point
    distances, indices = tree.query(point, k=n)

    return indices

def timeseries_calibration(model_dfsu,
                           model_item,
                           measured_df,
                           measured_item,
                           measured_coordinate_columns,
                           radius,
                           model_depth_correction=0,
                           change_depth_sign=False,
                           model_time_correction=0,
                           area=None,
                           area_column=None,
                           observation_unit_conversion_coefficient=1,
                           search_method="ellipsoid",
                           funcs=[np.nanmin, np.nanmax, np.nanmean, np.nanpercentile],
                           percentiles=[5, 95],
                           n_nearest_points=5,
                           ):
    
    dfsu = mikeio.read(model_dfsu, items=model_item)
    model_times = dfsu.time.copy() + pd.Timedelta(seconds=model_time_correction)
    geometry = dfsu.geometry
    element_coordinates = geometry.element_coordinates.copy()
    element_coordinates[:, 2] = element_coordinates[:, 2] + model_depth_correction
    if change_depth_sign:
        element_coordinates[:, 2] = -element_coordinates[:, 2]

    mask = measured_df[area_column].str.lower() == area
    measured_df= measured_df[mask]
    observation_x = measured_df[measured_coordinate_columns[0]].to_numpy()
    observation_y = measured_df[measured_coordinate_columns[1]].to_numpy()
    observation_z = measured_df[measured_coordinate_columns[2]].to_numpy()
    observation_coordinates = np.vstack([observation_x, observation_y, observation_z]).transpose()

    time_intersection = model_times.intersection(measured_df.index)
    measured_df= measured_df.loc[time_intersection]

    keys = {}
    for func in funcs:
        if func == np.nanmin:
            keys["Min"] = np.nanmin
        elif func == np.nanmax:
            keys["Max"] = np.nanmax
        elif func == np.nanmean:
            keys["Mean"] = np.nanmean
        elif func == np.nanpercentile:
            for elem in percentiles:
                keys[f"P{elem}"] = lambda arr, elem=elem: np.nanpercentile(arr, elem)
                
    outputs = {key: [] for key in keys.keys()}   
    for i in range(len(measured_df.index)):
        if search_method == "ellipsoid":
            model_indices = find_elements_within_ellipsoid(element_coordinates, observation_coordinates[i], radius, 1)
        elif search_method == "radius":
            model_indices = find_elements_within_radius(element_coordinates, observation_coordinates[i], radius)
        elif search_method == "nearest":
            model_indices = find_n_nearest_points(observation_coordinates, element_coordinates, i, n_nearest_points)
        model_values = dfsu[0].values[i, model_indices]
        for key in keys.keys():
            outputs[key].append(keys[key](model_values))
        print(i, np.mean(model_values), measured_df.iloc[i][measured_item] * observation_unit_conversion_coefficient)

    output_df = pd.DataFrame(data=outputs, index=time_intersection)
    return output_df
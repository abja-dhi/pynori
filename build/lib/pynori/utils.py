import mikeio
import os
import numpy as np
import pandas as pd
import mikeio

def split_dfsu_to_items(filename, path=None):
    """
    Split a dfsu file into separate dfsu files for each item.

    Parameters
    ----------
    filename : str
        Full path and file name to the dfsu file.
    path : str, optional
        Path to save the new files. By default: Current folder of the input filename.

    Returns
    -------
    list
        List of full paths to the new files.
    """
    if path is None:
        path = os.path.dirname(filename)
    actual_filename = os.path.basename(os.path.splitext(filename)[0])
    dfsu = mikeio.open(filename)
    items = dfsu.items
    if not os.path.exists(os.path.join(path, f"{actual_filename}_items")):
        os.makedirs(os.path.join(path, f"{actual_filename}_items"))
    for i in range(len(items)):
        part = mikeio.read(filename, items=i)
        part.to_dfs(os.path.join(path, f"{actual_filename}_items", f"{actual_filename}_{items[i]}.dfsu"))
        print(f"File {items[i]} created. {i+1}/{len(items)}")

def depth_corrector(coordinates, correction_value, change_sign=False):
    coordinates[:, 2] = coordinates[:, 2] + correction_value
    if change_sign:
        coordinates[:, 2] = -coordinates[:, 2]
    return coordinates

def get_layers(coordinates):
    layers = np.unique(coordinates[:, 2])
    return layers

def find_layer_coordinates(coordinates, layers, layer_no):
    layer = layers[layer_no]
    layer_coordinates = coordinates[coordinates[:, 2] == layer]
    layer_indices = np.where(coordinates[:, 2] == layer)[0]
    return layer_indices, layer_coordinates

def find_closest_layer_index(layers, observation_depth):
    if observation_depth < layers[0]:
        return np.nan
    if observation_depth > layers[-1]:
        return np.nan
    return np.argmin(np.abs(layers - observation_depth))

def find_values_within_radius_2d(layer_values, layer_coordinates, observation_coordinates, radius):
    distances = np.sqrt((layer_coordinates[:, 0] - observation_coordinates[0]) ** 2 + (layer_coordinates[:, 1] - observation_coordinates[1]) ** 2)
    return layer_values[distances < radius]

def get_plotting_data(X, Y):
    unique_Y = np.unique(Y)
    unique_X = []
    for i, y in enumerate(unique_Y):
        unique_X.append(np.max(X[Y == y]))
    unique_X = np.array(unique_X)
    data = np.vstack((unique_X, unique_Y)).T
    data = data[data[:, 1].argsort()]
    return data

def find_elements_within_radius_3d(array, point, radius):
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
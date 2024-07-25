import mikeio
import os


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
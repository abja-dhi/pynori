from pynori.plotter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mikeio

os.chdir(os.path.dirname(__file__))
#
#model_pr1 = r"\\USDEN1-STOR.DHI.DK\Projects\41806287\Models\Midwater\05_runs\rev3\midwater_plume_pr1test.m3fm - Result Files\Separate files\Total SSC.dfsu"
#dfsu = mikeio.read(model_pr1)[0]
#model_times = dfsu.time
#geometry = dfsu.geometry
#ec = geometry.element_coordinates.copy()
#
#model_depth_correction = -1200
#change_depth_sign = True
#ec = fix_coordinates(ec, model_depth_correction, change_depth_sign)


ROV = pd.read_csv(r"\\USDEN1-STOR.DHI.DK\Projects\41806287\41806287 NORI-D Data\Workflow Execution\ROV CTD Calibration\observations\ROV\near field\01_near_field_ROV_CTD.csv", index_col=0)
mask = (ROV["ROV Depth"] > 1200) & (ROV["ROV Depth"] < 1300)
ROV = ROV[mask]
ROV = ROV[["ROV Depth", "SSC (mg/L)"]]
mask = ROV["SSC (mg/L)"] < 1e-5
ROV = ROV[~mask]
ROV = ROV.dropna()

#time_intersection = (model_times + pd.Timedelta(seconds=1)).intersection(ROV.index)
#ROV = ROV.loc[time_intersection]

ROV_depth = ROV["ROV Depth"].to_numpy()
ROV_SSC = ROV["SSC (mg/L)"].to_numpy()
ROV_depth_bins = np.arange(1200, 1300, 2)
ROV_SSC_bins = np.arange(0, 70, 5)
fig, ax = scatter(ROV_SSC, ROV_depth, ROV_SSC_bins, ROV_depth_bins)
ax.set_xlabel("SSC (mg/L)")
ax.set_ylabel("Depth (m)")
ax.set_title("ROV SSC Calibration")
plt.show()
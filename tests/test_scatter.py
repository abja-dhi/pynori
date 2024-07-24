from pynori.plotter import scatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WS = pd.read_csv("data/WS.csv", header=None).to_numpy().flatten()
Hm0 = pd.read_csv("data/Hm0.csv", header=None).to_numpy().flatten()
X_bins = np.arange(0, 34, 2)
Y_bins = np.arange(0, 10.1, 0.4)

fig, ax = scatter(WS, Hm0, X_bins, Y_bins, density_power=0.5)
#plt.show()
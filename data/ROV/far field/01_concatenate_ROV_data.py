# -*- coding: utf-8 -*-[]
"""
Created on Wed Apr 10 12:30:37 2024

@author: anba
"""

import pandas as pd 
import numpy as np 
import os 
import pyplume as pp 


#%%
fpath   = r'\\USDEN1-STOR.DHI.DK\Projects\41806287\41806287 NORI-D Data\Data\ROV\Island Pride\Position\Processed\01_concatenate\01_IP_ROV_HAIN.csv'
df_hain = pd.read_csv(fpath, index_col = 0, parse_dates = True)

fpath   = r'\\USDEN1-STOR.DHI.DK\Projects\41806287\41806287 NORI-D Data\Data\ROV\Island Pride\Position\Processed\01_concatenate\01_IP_ROV_USBL.csv'
df_usbl = pd.read_csv(fpath, index_col = 0, parse_dates = True)

fpath = r'\\USDEN1-STOR.DHI.DK\Projects\41806287\41806287 NORI-D Data\Data\ROV\Island Pride\CTD\Processed\01_concatenate\01_IP_ROV_CTD_10s_mean.csv'
df_ctd = pd.read_csv(fpath, index_col = 0, parse_dates = True)

fpath = r'\\USDEN1-STOR.DHI.DK\Projects\41806287\41806287 NORI-D Data\Data\PCV\Position\Processed\02 PCV Tracks\02_PCV_tracks.csv'
df_pcv = pd.read_csv(fpath, index_col = 0, parse_dates = True)
df_pcv.index = df_pcv.index.tz_localize(None)

fpath = r'\\USDEN1-STOR.DHI.DK\Projects\41806287\41806287 NORI-D Data\Data\Vessel\Hidden Gem\Riser Return\Processed\Position\02_midwater_discharge_position.csv'
df_mwd = pd.read_csv(fpath, index_col = 0, parse_dates = True)
df_mwd.index = df_mwd.index.tz_localize(None)


#%%

st = df_ctd.index.min()
et = df_ctd.index.max()


df_hain = df_hain.reindex(df_ctd.index,method = 'nearest', tolerance = pd.Timedelta('10s'))
df_usbl = df_usbl.reindex(df_ctd.index,method = 'nearest', tolerance = pd.Timedelta('10s'))
df_pcv  = df_pcv.reindex(df_ctd.index,method = 'nearest', tolerance = pd.Timedelta('10s'))


ctd_offset = -0.57 # z-distance from CTD to position measurement. 

df_ctd['ROV Easting']  = df_hain['Easting'].fillna(df_usbl['Easting'])
df_ctd['ROV Northing'] = df_hain['Northing'].fillna(df_usbl['Northing'])
df_ctd['ROV Depth']    = df_hain['Depth'].fillna(df_usbl['Depth'] + ctd_offset)
df_ctd['ROV Altitude'] = df_hain['Altitude'].fillna(df_usbl['Altitude'])
df_ctd['PCV Test'] = df_pcv['test']
df_ctd = df_ctd.dropna(subset = ['ROV Easting','ROV Northing','ROV Depth'])





df_ctd['PCV Test'] = df_pcv['test']
df_ctd = df_ctd.dropna(subset = ['ROV Easting','ROV Northing','ROV Depth'])

#%% trim off all data not midwater or benthic 

mask = ((df_ctd['ROV Depth']>=1000) & (df_ctd['ROV Depth']<=1400)) | (df_ctd['ROV Depth']>=4100)
df_ctd = df_ctd.loc[mask]


mask = (df_ctd['ROV Depth']>=1000) & (df_ctd['ROV Depth']<=1400)
df_ctd['AREA'] = 'BENTHIC'
df_ctd.loc[mask,'AREA'] = 'MIDWATER'


# mask = df_ctd['PCV Test'].isnull()
# df_ctd = df_ctd[~mask]


#df_ctd = df_ctd.resample('10s').first()



#%% insert nans after long time gaps

new_rows = []
for i in range(len(df_ctd)-1):
    dt = df_ctd.index[i+1] - df_ctd.index[i]
    if dt.total_seconds() > 15:
        new_row = df_ctd.iloc[i].copy()
        new_row.name = new_row.name + pd.Timedelta('10s')
        
        new_row[new_row.index[:-2]] = np.nan
        
        new_rows.append(new_row)
        
        
df_ctd = pd.concat([df_ctd,pd.concat(new_rows,axis = 1).transpose()]).sort_index()

#%% de spike 

mask = [False]
for i in range(1,len(df_ctd)-1):
    if (df_ctd['Turbidity (NTU)'].iloc[i-1]<0.1) & (df_ctd['Turbidity (NTU)'].iloc[i+1]<0.1) & ((df_ctd['Turbidity (NTU)'].iloc[i]>.2)):
        print('a')
        
        mask.append(True)
        
    else:
        mask.append(False)
        
mask.append(False)

df_ctd['Turbidity FILTERED (NTU)'] = df_ctd['Turbidity (NTU)']
df_ctd.loc[mask,'Turbidity FILTERED (NTU)'] = np.nan


df_ctd['SSC (mg/L)'] = df_ctd['Turbidity FILTERED (NTU)']*1.6
#%% trim off all data not midwater or benthic 



df_ctd.to_csv('01_far_field_ROV_CTD.csv')

#%%
import matplotlib.pyplot as plt 

fig,ax = pp.plotting.subplots(nrow = 2, sharex = True)

ax[0].plot(df_ctd.index,df_ctd['Turbidity (NTU)'],lw = 1)
ax[0].plot(df_ctd.index,df_ctd['Turbidity FILTERED (NTU)'],lw = 1.5, label = 'Filtered')
ax[1].plot(df_ctd.index,df_ctd['ROV Depth'],lw = 1)

ax[0].set_ylabel('Turbidity (NTU)')
ax[1].set_ylabel('Depth (m)')
ax[1].set_xlabel('Date')
ax[1].invert_yaxis()

plt.tight_layout()import matplotlib.pyplot as plt 

fig,ax = pp.plotting.subplots(nrow = 2, sharex = True)

ax[0].plot(df_ctd.index,df_ctd['SSC (mg/L)'],lw = 1)
ax[1].plot(df_ctd.index,df_ctd['ROV Depth'],lw = 1)

ax[0].set_ylabel('Turbidity (NTU)')
ax[1].set_ylabel('Depth (m)')
ax[1].set_xlabel('Date')
ax[1].invert_yaxis()

plt.tight_layout()

#%%


fig,ax = pp.plotting.subplots(ncol = 2, sharex = True, sharey = True, figwidth = 10, figheight = 5)


mask = df_ctd['AREA'] == 'MIDWATER'
ax[0].scatter(df_ctd.loc[mask,'ROV Easting'],df_ctd.loc[mask,'ROV Northing'],s = 0.01, c = 'gray', label = 'ROV NTU = 0')
mask = (df_ctd['AREA'] == 'MIDWATER') & (df_ctd['Turbidity (NTU)']>1)
ax[0].scatter(df_ctd.loc[mask,'ROV Easting'],df_ctd.loc[mask,'ROV Northing'],s = 1, c = df_ctd.loc[mask,'Turbidity (NTU)'], vmax = 10, cmap = 'jet')
s1 = ax[0].plot(df_mwd['easting'],df_mwd['northing'], c = 'black', lw = 0.1, label = 'Discharge Path')
ax[0].set_title('MIDWATER')
ax[0].legend()


mask = df_ctd['AREA'] == 'BENTHIC'
ax[1].scatter(df_ctd.loc[mask,'ROV Easting'],df_ctd.loc[mask,'ROV Northing'],s = 0.01, c = 'gray', label = 'ROV NTU = 0')
mask = (df_ctd['AREA'] == 'BENTHIC') & (df_ctd['Turbidity (NTU)']>1)
s2= ax[1].scatter(df_ctd.loc[mask,'ROV Easting'],df_ctd.loc[mask,'ROV Northing'],s = 1, c = df_ctd.loc[mask,'Turbidity (NTU)'], vmax = 10, cmap = 'jet')

ax[1].plot(df_pcv['easting'],df_pcv['northing'], c = 'black', lw = 0.1, label = 'PCV Path')
ax[1].set_title('BENTHIC')
ax[1].legend()


ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_xlabel('Easting (m)')
ax[0].set_ylabel('Northing (m)')
ax[1].set_xlabel('Easting (m)')

fig.colorbar(s2, ax=ax.ravel().tolist(), orientation='horizontal', shrink = 0.25, label = 'Turbidity (NTU)', extend = 'max')
#plt.tight_layout()
fig.suptitle('Far Field ROV')

fig.savefig('far field ROV.png', dpi = 600)







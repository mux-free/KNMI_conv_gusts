#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:55:59 2023

@author: frei
"""
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from data_processing import sel_month, add_datetime64_column, read_df_cell
from plot_case_studies_annimation import plot_cellfield_tas_precip_div
from CellTracking_DataProcessing_Cleaning import process_cells_stats
#%%
#############
# Load Data #
#############
home_dir = '/net/pc200057/nobackup_1/users/frei'
df_cells_raw = pd.read_csv(f'{home_dir}/track_cluster/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist/merge_HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist.txt', delim_whitespace=True)
df_cells_fut_raw = pd.read_csv(f'{home_dir}/CPM_future/track_cluster_fut/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r04_rcp85/merged_df_cell_stats.txt', delim_whitespace=True) 

df_cells = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells.csv')
df_cells_fut = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut.csv')
#====================================================================================================================


#%%
#************************************************#
# NOTE: Function already run, only import scirpt #
#************************************************#
'''
Preprocess raw data:
    
    - Remove cells with center of mass over the ocean
    
    - Remove cells that touch the boundary
    
    - Add a column that gives the average wsgsmax-speed of the domain at the time for each cell
    
    - Add number of cells that occured on each day (for each cell)
'''
preproces_df_cells=True
if preproces_df_cells == False:
    df_cells_raw = process_cells_stats(df=df_cells_raw, time='hist')
    df_cells_fut_raw = process_cells_stats(df_cells_fut_raw, time='future')

    df_cells_raw.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_raw.csv', index=False)
    df_cells_fut_raw.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut_raw.csv', index=False)
    
df_cells_raw = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_raw.csv')
df_cells_fut_raw = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut_raw.csv')
#%%
def change_dtype(df):
    # Keep track of datetime64 columns
    datetime_col = df.select_dtypes(include='datetime64').columns
    # Columns with letters
    exclude_col = datetime_col.append(pd.Index(['touchb']))
    for col in df.columns:
        if col not in exclude_col:
            df[col] = pd.to_numeric(df[col])
    return df

df_cells_raw = change_dtype(df_cells_raw)
df_cells_fut_raw = change_dtype(df_cells_fut_raw)
#%%
#####################################################################################################################
#####################################           PLOTS          ######################################################
#####################################################################################################################

#============================================================#
# First plot:
#     Scatter of log of clarea and wsgsmax
#============================================================#

#--------------------
used_data = 'raw'
# used_data = 'cleaned'
#-------------------

if used_data == 'raw':
    df = df_cells_raw
    df_fut = df_cells_fut_raw
    savefig = '/usr/people/frei/MaxFrei/Max_figures/general_analysis/CellTrack_Analysis/cell_overview_cleaning/Fut_Hist_peakval_grd_clarea_RawData'
elif used_data == 'cleaned':
    df = df_cells
    df_fut = df_cells_fut
    savefig = '/usr/people/frei/MaxFrei/Max_figures/general_analysis/CellTrack_Analysis/cell_overview_cleaning/Fut_Hist_peakval_grd_clarea_CleanedData'
    
#------------------------------------------------------------------------------    
fig = plt.subplots(figsize=(8,4))
gs = GridSpec(nrows=1, ncols=2)

ax=plt.subplot(gs[0,0])
img=ax.scatter(df['grd_clarea'], 
               df['peakVal'], 
               s=5,
               edgecolors = 'k', 
               alpha=0.8 )
ax.set_xlabel('Grid cell area')
ax.set_ylabel('Peak Val')
ax.set_title('', fontsize=10)
ax.set_xscale('log')
ax.set_title('Historic run (1995-2005)', fontsize =10)
ax.set_ylim(20,50)
ax.set_xlim(1, 10000)
textstr = ''.join((
    'Number of cells (1995-2005):  {}'.format(df.shape[0])
    ))
props = dict(boxstyle='round', facecolor='wheat',alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=props)

ax=plt.subplot(gs[0,1])
img=ax.scatter(df_fut['grd_clarea'], 
               df_fut['peakVal'], 
               s=5,
               edgecolors = 'k', 
               alpha=0.8 )
ax.set_xlabel('Grid cell area')
# ax.set_ylabel('Peak Val')
ax.set_title('Future run (2089-2099)', fontsize=10)
ax.set_xscale('log')
ax.set_ylim(20,50)
ax.set_xlim(1, 10000)
textstr = ''.join((
    'Number of cells (1995-2005):  {}'.format(df_fut.shape[0])
    ))
props = dict(boxstyle='round', facecolor='wheat',alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=props)

plt.savefig(savefig, dpi=100)

#%%

#=======================================================================================================================================================
# Plot shows:
#    Scatter plot of Peak-wsgs-Val of cell divided by Domain_mean     VS        Area of Cell
#=======================================================================================================================================================


fig,ax = plt.subplots()
img=ax.scatter(df_cells['cells_per_day'], df_cells['wsgsmax_ratio']  , s =3)
ax.set_xscale('log')
ax.set_ylim(0,10)



xbins = 6 * 10**np.linspace(0, 3, 80)
ybins = np.linspace(0, 15, 60)
counts, _, _ = np.histogram2d(df_cells['grd_clarea'], df_cells['wsgsmax_ratio'], bins=(xbins, ybins))

fig, ax = plt.subplots()
img=ax.pcolormesh(xbins, ybins, counts.T , norm=LogNorm() )
ax.set_xscale('log')
ax.set_xlabel('Area [in gridpoints]')
ax.set_ylabel('Peak Val / Domain_Mean')
# ax.set_title('WG-cells PeakVal with Area under 50x50 (1995-2005)')
plt.colorbar(img, label='Density' , ax=ax)


# plt.savefig('/usr/people/frei/Desktop/Max_figures/gust_cells/peakval_area_2dhist', dpi=500)

counts, _, _ = np.histogram2d(df_cells['grd_clarea'], df_cells['wsgsmax_ratio'], bins=(xbins, ybins))
fig, ax = plt.subplots()
img=ax.pcolormesh(xbins, ybins, counts.T , norm=LogNorm() )
ax.set_xscale('log')
ax.set_xlabel('Area [in gridpoints]')
ax.set_ylabel('Average Val / Domain_Mean')
# ax.set_title('WG-cells PeakVal with Area under 50x50 (1995-2005)')
plt.colorbar(img, label='Density' , ax=ax)










#%%

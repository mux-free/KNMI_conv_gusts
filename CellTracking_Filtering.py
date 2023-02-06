#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:56:35 2022

@author: frei



- This script groups all the tracked cells by the date they occur.

- Then we will extract the dates that show 240 OR LESS (arbitrary value chosen by me)

- This list is then used to filter out all these days that have more than 240 cells in one day (since we suspect that they stem from a synoptic system)
                                                                                                
"""

#%%

# LIBS
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
#%%
#----------
# Load Data
#----------
rootdir = '/net/pc200057/nobackup_1/users/frei/CPMs/'

# cells_dir = '/net/pc200057/nobackup_1/users/frei/track_cluster/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist/allyears_merged/cells.nc'
# ds_cells =  xr.open_dataset(cells_dir).drop_vars('lev')

# Dataframe with cells (excluding ocean and boundary touching)
df_cells_preprocessed = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_preprocessed.csv')
df_cells_preprocessed_fut = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_preprocessed_fut.csv')


df_cells_all = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_ALL_cells.csv')
df_cells_all_fut = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_ALL_fut_cells.csv')


#%%
def filter_cells(df, time, filters='both'):

    if time == 'hist':
        field_dir ='/net/pc200057/nobackup_1/users/frei/CPMs/fields_lonlatbox'
        # xarray.DataSet windgusts
        ds_wsgsmax = xr.open_dataset(f'{field_dir}/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_wsgsmax_merged_lonlatbox/wsgsmax.his.CXFPS025.HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist.1hr_lonlatbox_merged.nc')
        ds_wsgsmax = sel_month(ds_wsgsmax)
        # xarray.DataSet temperature
        
        ds_tas = xr.open_mfdataset(f'{field_dir}/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_tas_lonlatbox/*.nc')
        ds_tas = sel_month(ds_tas)
        
    elif time == 'future':
        ds_wsgsmax = xr.open_mfdataset(f'/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/fields_lonlatbox_selmonth/wsgsmax_lonlatbox_selmonth/wsgsmax.*.nc')
        ds_tas = xr.open_mfdataset(f'/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/fields_lonlatbox_selmonth/tas_lonlatbox_selmonth/tas.*.nc')

    #==============================================================================
    # FILTER DF_CELLS
    #==============================================================================
    if filters == 'both' or filters == 'time':
        print('Filtering for convective time')
        ### FILTER OUT TIMES THAT ARE NOT RELATED TO CONVECTION ACITVITIES
        time_notconvec = ['01:30' , '02:30' , '03:30' , '04:30' , '05:30' , '06:30' , '07:30' , '08:30' , '09:30' , '10:30' , '11:30' , '12:30' , '13:30']
        times = df['datetime'].astype(str)
        
        filt_time = times.str.contains('|'.join(time_notconvec))
        # Apply filter to dataset
        df = df.loc[~filt_time]
    
    #==============================================================================
    ## Filter Points with a value lower than a certain threshold
    #==============================================================================
    if filters == 'both' or filters == 'field_average':
        print('Filtering for threshold ratio')
        ############## CHOOSE A RATIO THAT DISTINGUISH CONVECTIVE AND SYNOPTIC SCALES ####################
        ratio_threshold = 3
        #-------------------------------------------------------------------------------------------------
        ## Plot cells that show an arbitrary ratio of the windspeed within the cell compared to the domain average
        filt = df['wsgsmax_ratio'] > ratio_threshold
        #filt = df['wsgsav_ratio'] > ratio_threshold
        # Apply filter to dataset
        df = df.loc[filt]    
    
    # Reset index of dataset
    df.reset_index(inplace=True,drop=True)
    
    return df

#%%
#======================================== Apply Filtering Function ==========================================================================
df_cells = filter_cells(df_cells_preprocessed, time='hist')
df_cells_fut = filter_cells(df_cells_fut, time='future')
    
#%%

#===================================================================================================================
#===================================================================================================================
###########################   CALCULATE TEMPATURE STATISTICS FOR EACH CELL    ######################################
#===================================================================================================================
#===================================================================================================================

#%%

###################################
# Add Wsgsmax-Cells TAS quantiles #
###################################
def add_temp_stats(df, time):  
    ## Import Data
    if time == 'hist':
        print('Add historic tempatrue stats')
        
        if 'preprocess' in df:
            print('We add tempstats to pre-processed cells')
            df_tempstats = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/cell_temp_drop_stats/df_cells_preprocessed_tempstats.csv')
        else:
            print('Add tempstats to filtered cells')
            df_tempstats = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/cell_temp_drop_stats/df_cells_tempstats.csv')
    
    
    elif time == 'future':
        print('Add future tempature stats')
        
        if 'preprocess' in df:
            print('We add tempstats to pre-processed cells')
            df_tempstats = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/cell_temp_drop_stats/df_cells_preprocessed_tempstats_fut.csv')
        else:
            print('Add tempstats to filtered cells')
            df_tempstats = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/cell_temp_drop_stats/df_cells_tempstats_fut.csv')
        
    ## Add TAS_2h tp df_cell
    df['cell_mean_2h'] = df_tempstats['cell_mean_2h']
    df['cell_q90_2h'] = df_tempstats['cell_q90_2h']
    df['cell_q75_2h'] = df_tempstats['cell_q75_2h']
    df['cell_q50_2h'] = df_tempstats['cell_q50_2h']
    df['cell_q25_2h'] = df_tempstats['cell_q25_2h']
    df['cell_q10_2h'] = df_tempstats['cell_q10_2h']
    
    return df
#%%
# Apply function to add temperature statistics of cells for future and past dataset
df_cells           =    add_temp_stats(df_cells,     time='hist')
df_cells_fut       =    add_temp_stats(df_cells_fut, time='future')
# Inspect NaN values
df_cells_fut_nan = df_cells_fut[df_cells_fut['cell_q75_2h'].isna()]

# ############ DROP NaN values (there are 3) ################
df_cells.dropna(axis=0, inplace=True)
df_cells_fut.dropna(axis=0, inplace=True)

## Save DataFrames
# df_cells.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells.csv', index=False)
# df_cells_fut.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut.csv', index=False)

#%%
# #---------------------------------------------------------
# # Create filter for df_cells: 
# #  - Only keep days with 240 or less cells!
# #---------------------------------------------------------
# # Create Filter
# filt = df_cellday['number_of_cells'] < 240
# # Apply filter to dataset to get the dates 
# date_convec = df_cellday.loc[filt, 'date']
# # Create a filter based on df_cell, where the days with more than 240 cells are FALSE
# filt_dates = df_cell['date(YYYYMMDD)'].isin(date_convec)
# # Apply this filter to df_cell to create df_cell_reduced
# df_cell_reduced = df_cell.loc[filt_dates]
# # For thest purposes, create a DataSet with only many cells per day
# df_cell_syno = df_cell.loc[~filt_dates]


#=======================================================================================================================================================


# ## Plot cells that occur during convective times
# times = df_cell_wgratio['datetime'].astype(str)
# filt_time = times.str.contains('|'.join(time_notconvec))
# df_cell_wgratio_convtime = df_cell_wgratio.loc[~filt_time]





# ############################
# # Effectiveness of fitlers #
# ############################

# # df_cells_raw1 = df_cells
# # df_cells_fut_raw1 = df_cells_fut

# ## Apply only time filter
# df_filter_time = filter_cells(df_cells_raw1, 'hist', filters='time')
# df_filter_fut_time = filter_cells(df_cells_fut_raw1, 'future', filters='time')

# ## Apply only field homogenity filter
# df_filter_homo = filter_cells(df_cells_raw1, 'hist', filters='field_average')
# df_filter_fut_homo = filter_cells(df_cells_fut_raw1, 'future', filters='field_average')

# ## Quick test that filter yield the same if applied consequtively
# df_test = filter_cells(df_filter_time, 'hist', filters='field_average')

# ## Saving fields
# df_filter_time.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/filter_effectiveness/df_filter_time.csv', index=False)
# df_filter_fut_time.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/filter_effectiveness/df_filter_fut_time.csv', index=False)

# df_filter_homo.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/filter_effectiveness/df_filter_homo.csv', index=False)
# df_filter_fut_homo.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/filter_effectiveness/df_filter_fut_homo.csv', index=False)





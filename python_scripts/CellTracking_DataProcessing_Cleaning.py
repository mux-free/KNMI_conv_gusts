#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 07:33:29 2022

@author: frei

This script tries to contain all the preprocessing and cleaning steps that are applied to the raw celltracker output. 

"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from data_processing import add_datetime64_column 
#%%
#----------------------
# Load CellTrack Output
#----------------------

## Import cells from cell-tracking ouput --> Note output has to be merged before importing
data_dir = '/net/pc200057/nobackup_1/users/frei/track_cluster/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist'
df_cells_all = pd.read_csv(f'{data_dir}/merge_HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist.txt', delim_whitespace=True)
df_cells_all_fut = pd.read_csv('/net/pc200057/nobackup_1/users/frei/CPM_future/track_cluster_fut/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r04_rcp85/merged_df_cell_stats.txt', delim_whitespace=True)


## Convert all columns with numbers to float
df_cells_all =  df_cells_all.apply(lambda i: i.apply(lambda x: float(x) if str(x).replace('.','',1).isdigit() else x))
df_cells_all_fut =  df_cells_all_fut.apply(lambda i: i.apply(lambda x: float(x) if str(x).replace('.','',1).isdigit() else x))


#%%
def process_cells_stats(df, time):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    time : TYPE
        Either:
            'hist'
        OR
            'future'

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    #-------------------------
    # Drop redunant title rows
    df_duplic = df[ df.duplicated(keep=False)]
    # print(f'\nSize of df_duplicates (repeating title rows due to merging):\t{df_duplic.shape}\n')
    df = df[~df.duplicated(keep=False)]
    # reset index
    df.reset_index(inplace=True, drop=True)
    
    #-------------------------------
    # Drop cells that touch boundary
    #-------------------------------
    mask = df['touchb']=='F'
    df = df[mask]       # 7112 cells (rows) touch the boundary
    df.reset_index(inplace=True, drop=True)
    print('\nStep 1: Drop cells that touch boundary. \n')
    #--------------------------------------------------
    # Drop cells with the centre of mass over the ocean
    #--------------------------------------------------
    ds_ocean= xr.open_dataset('/net/pc200057/nobackup_1/users/frei/track_cluster/output_tracking_sdomain/celltrack/sftof.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.fx_lonlatbox.nc')
    da_ocean=ds_ocean.sftof
    ocean_idx =[]
    count,pct = 0,0
    iter_range = df.shape[0]
    for i in range(iter_range):
        count +=1
        if da_ocean.sel(x=df.loc[i, 'clcmassX'], y=df.loc[i, 'clcmassY'], method="nearest") == 100:
            ocean_idx.append(i)
            df.drop(i, inplace=True)
        if count == round(iter_range/10):
            pct += 10
            print(f'Step 2: Drop cells over ocean \t--\t Loop progress: {pct}%')
            count = 0
    df.reset_index(inplace=True, drop=True)
    print('Done\n')
    #--------------------------------------------------------------------
    # Create 3 new DATETIME columns (half-hour, hour_floor and hour_ceil)
    #--------------------------------------------------------------------
    print('Step3: Add datetime column to dataframe(floor,inbetween and ceil) to df')
    add_datetime64_column(df)
    print('Done\n')
    #--------------------------------------------------------------------
    # Get a measure for variability of wind field
    #-------------------------------------------------------------
    #client = Client(n_workers=2, threads_per_worker=2)
    #chunks={'time' : 10,  }
    #-------------------------------------------------------------- HISTORIC OR FUTURE DATASET --------------------------------------------------------------------------------------------------------------------------
    if time == 'hist':
        rootdir = '/net/pc200057/nobackup_1/users/frei/'
        #ds_wsgsmax = xr.open_dataset(f'{rootdir}/CPMs/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_wsgsmax_merged_lonlatbox/wsgsmax.his.CXFPS025.HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist.1hr_lonlatbox_merged.nc', chunks=chunks)
        #ds_wsgsmax = sel_month(ds_wsgsmax)  
        mean_wsgsmax = xr.open_dataset('/net/pc200057/nobackup_1/users/frei/CPMs/fields_lonlatbox/sel_MJJAS/stat_fields/wsgsmax.his.CXFPS025.HCLIM38h1_CXFPS_RACMOfECEARTH_r14_rcp85_spatial_average.nc')
    elif time == 'future':
        #ds_wsgsmax = xr.open_mfdataset(f'/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r04_rcp85_wsgsmax_lonlatbox/wsgsmax.*.nc', chunks=chunks)
        mean_wsgsmax = xr.open_dataset('/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/fields_lonlatbox_selmonth/stat_fields/wsgsmax.his.CXFPS025.HCLIM38h1_CXFPS_RACMOfECEARTH_r14_rcp85_spatial_average.nc')
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Create a df with time and mean windspeed of domain
    print('Step 4: Compute mean of wsgsmax field for domain \n')
    
    # Convert to dataframe
    df_mean_wsgsmax = mean_wsgsmax.wsgsmax.to_dataframe()
    # Drop height column
    df_mean_wsgsmax.drop('height',axis=1,inplace=True)
    # Reset indeces and create time column
    df_mean_wsgsmax.reset_index(inplace=True)
    df_mean_wsgsmax['time'] = df_mean_wsgsmax['time'].astype(str)
    
    # Add a time_merge column to df_cell to merge the two dtaaframes (time_merge is only used to merge and will be dropped later on)
    df['time_merge'] = df['datetime'].astype(str)
    df_mean_wsgsmax['time_merge'] = np.nan
    print('Merge df_cells with wsgsmax_domain_average \n')
    for i in range(df_mean_wsgsmax.shape[0]):
        df_mean_wsgsmax.loc[i, 'time_merge'] = df_mean_wsgsmax.loc[i, 'time'].replace(' ', 'T') + '.000000000'
    
    ## Merge df_cell with mean_wsgsmax
    df = pd.merge(df,df_mean_wsgsmax, how='left', on='time_merge')
    df.drop(['time','time_merge'] , axis=1,inplace=True)
    
    # Calculate ratio between Cell wsgsmax and average wsgs field of domain
    print('Step 5: Compute wind-strenght ratio between Cell wsgsmax and average domain wind-field \n')
    df['wsgsmax_ratio'] = pd.to_numeric(df['peakVal']) / pd.to_numeric(df['wsgsmax'])
    df['wsgsav_ratio'] = pd.to_numeric(df['avVal']) / pd.to_numeric(df['wsgsmax'])
    
    
    #--------------------------------------------------------------------
    # Compute number of cells that occured on each day
    #--------------------------------------------------------------------
    print('Compute number of cells per day \n')
    df_cell_groupdate = df.groupby(df['date(YYYYMMDD)'])
    # Get numbers of entries for different groups
    celldate_grouped_size = df_cell_groupdate.size()
    
    # Create a DataFrame with the date and the numbers of cells thaty occured on that day
    data = {"date": celldate_grouped_size.index, "number_of_cells": celldate_grouped_size.values, }
    df_cellday = pd.DataFrame(data=data).sort_values(by=['number_of_cells'], ascending=False, ignore_index=True)
    
    # Get DataSet of day with most cells
    df_max_date = df_cell_groupdate.get_group(df_cellday.date.iloc[0])
    
    ## Add the number of cells that occured on that day as a column to df_cell
    df['cells_per_day'] = df.groupby('date(YYYYMMDD)')['date(YYYYMMDD)'].transform('count')
    print('ALL pre-processing done!! \n')
    return df

#%%
if __name__ == '__main__':
    
    
    # Save the unprocessed cells
    df_cells_all.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_ALL_cells.csv', index=False)
    df_cells_all_fut.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_ALL_fut_cells.csv', index=False)

    # Apply pre-processing function
    df_cells_preprocessed = process_cells_stats(df_cells_all, time='hist')
    df_cells_preprocessed_fut = process_cells_stats(df_cells_all_fut, time='future')
    
    ## Save the pre-processed cells (after applying function in this module and before filtering)
    df_cells_preprocessed.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_preprocessed.csv', index=False)
    df_cells_preprocessed_fut.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_preprocessed_fut.csv', index=False)
        
    print(df_cells_preprocessed.isnull().values.sum())


#%%
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                         TESTS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ds_cells =  xr.open_dataset(f'{data_dir}/cells.nc').drop_vars('lev')

# foo = 595
# ds_cell_case = ds_cells.sel(time=df_foo.loc[foo,'datetime'])

# fig,ax =plt.subplots()
# da_ocean.plot(ax=ax, alpha=0.5)
# (ds_cell_case.cellID[0,:,:] == df_cells.loc[foo, 'clID'] ).plot(ax=ax, alpha=0.5)


# xbins = 5 * 10**np.linspace(0, 4, 50)
# ybins = np.linspace(20, 45, 50)
# counts, _, _ = np.histogram2d(df_cell['grd_clarea'], df_cell['peakVal'], bins=(xbins, ybins))

# fig, ax = plt.subplots()
# img=ax.pcolormesh(xbins, ybins, counts.T , norm=LogNorm() )
# ax.set_xscale('log')
# ax.set_xlabel('Area [in gridpoints]')
# ax.set_ylabel('Peak Val')
# ax.set_title('WG-cells PeakVal with Area under 50x50 (1995-2005)')
# plt.colorbar(img, label='Density' , ax=ax)
# plt.savefig('/usr/people/frei/Desktop/Max_figures/gust_cells/peakval_area_2dhist', dpi=500)


# #--------------------------------------------------------------------------------
# # Old method to drop ocean , coord_x and coord_y are all indexes of x and y axis over ocan
# #--------------------------------------------------------------------------------
# df_noocean_m2 = df_cells.copy(deep=True)
# ocean_x = da_ocean.x.where(da_ocean == 100).values.flatten()
# ocean_x = ocean_x[~np.isnan(ocean_x)] / 2500 - ocean_x.min()
# ocean_y = da_ocean.y.where(da_ocean == 100).values.flatten()
# ocean_y = ocean_y[~np.isnan(ocean_y)] / 2500 - ocean_y.min()

# # Create a list with pairs of coords that are located over the ocean
# coord_list = list(zip(coord_x, coord_y))
# # loop trough the dataframe 
# ocean_idx = []
# for i in range(df_cells.shape[0]):
#     if (round(df_cells.loc[i, 'grd_clcmassX']), round(df_cells.loc[i, 'grd_clcmassY']) ) in coord_list:
#         df_noocean_m2.drop(i, inplace=True)
#         ocean_idx.append(i)
#%%

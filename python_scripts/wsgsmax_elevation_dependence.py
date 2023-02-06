#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:14:56 2022

@author: frei

This script plots the relation of wsgsmax and height in a 2d histogram plot. For wsgsmax we can either use the max value over the whole time or the p99 or p95
"""

import matplotlib
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
import math
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
from scipy import stats


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap


from data_processing import sel_month, add_datetime64_column, drop_ocean, downsize_domain
from plotting_maxf import contour_2_1_plot

#%%
#-------
# Test
#-------
# ds['sftof'] = ds_land_ocean['sftof']
# mask = ds_land_ocean.sftof < 1
# ds_wsgsmax.coords['mask'] = (('y', 'x'), mask.values )
#%%

def plot_height_wsgsmax(stat = 'max_value', model='hist', domain='whole_domain'):

    #--------------------------------
    # Import Files depending on model
    #--------------------------------
    if 'whole' in domain:
        
        print('Height-dependance for whole domain')
        ds_topo = xr.open_dataset('/usr/people/vries/NOBACKUP1/DATA/CORDEX-FPS/HCLIM38h1_CXFPS1999_fRACMO/orog.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.fx.nc')
        rootdir = '/net/pc200057/nobackup_1/users/frei/wsgs_overview/MJJAS/stats/field_max'
        if 'hist' in model:
            print('Historic run (1995-2005)')
            ds_wsgsmax = xr.open_dataset(f'{rootdir}/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_merged_MJJAS_max.nc')
        if 'fut' in model:
            print('Future run (2089-2099')
            ds_wsgsmax = xr.open_dataset(f'{rootdir}/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_fut_wsgsmax_merged_MJJAS_max.nc')
    
    else:
        print('Height-dependance for NL-domain')
        rootdir = '/net/pc200057/nobackup_1/users/frei/wsgs_overview/MJJAS/stats_lonlatbox'
        ds_topo = xr.open_dataset(f'{rootdir}/orog.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.nc')
        if 'hist' in model:
            print('Historic run (1995-2005)')
            ds_wsgsmax = xr.open_dataset(f'{rootdir}/field_max/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_merged_MJJAS_lonlatbox_max.nc')
        if 'fut' in model:
            print('Future run (2089-2099)')
            ds_wsgsmax = xr.open_dataset(f'{rootdir}/field_max/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_fut_wsgsmax_merged_MJJAS_lonlatbox_max.nc')
    
    #rootdir = '/net/pc200057/nobackup_1/users/frei/wsgs_overview/MJJAS/stats/field_max'
    #ds_wsgsmax = xr.open_dataset(f'{rootdir}/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_merged_MJJAS_max.nc')
    #ds_topo = xr.open_dataset('/usr/people/vries/NOBACKUP1/DATA/CORDEX-FPS/HCLIM38h1_CXFPS1999_fRACMO/orog.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.fx.nc')
    #---------------------------
    # Import topo and ocean data
    ds_land_ocean= xr.open_dataset(f'/usr/people/vries/NOBACKUP1/DATA/CORDEX-FPS/HCLIM38h1_CXFPS1999_fRACMO/sftof.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.fx.nc')
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Drop ocean of wsgsmax file
    da_wsgsmax_land = drop_ocean(ds_wsgsmax.wsgsmax)
    
    # Drop ocean of topgraphie file
    ds_topo['sftof'] = ds_land_ocean['sftof']
    ds_topo = ds_topo.where(ds_topo.sftof < 1)
    #-------------------------------------------------
    
    #############  Select the variable of interest  ###############
    variable = da_wsgsmax_land
    #---------------------------------------------------
    # SET TITLE OF FIG
    if 'hist' in model:
        title = 'Maximal windgust (1995-2005)'
    if 'fut' in model:
        title = 'Maximal windgust (2089-2099)'
    #---------------------------------------------------
    # calculate percentiles of variable
    variable['orog'] = ds_topo.orog
    bins = list( np.arange(   ds_topo.orog.min()   ,    math.ceil(ds_topo.orog.max()/100 +1)*100   ,   25) )
    ds_grouped = variable.groupby_bins(variable['orog'], bins=bins)
    da_grouped = variable.groupby_bins(variable['orog'], bins=bins)
    
    print('Calculate percentiles of max wsgsmax')
    group_mean = ds_grouped.mean().values.flatten()
    group_q25 = da_grouped.quantile(0.25).values.flatten()
    group_q50 = da_grouped.quantile(0.5).values.flatten()
    group_q75 = da_grouped.quantile(0.75).values.flatten()
    group_q90 = da_grouped.quantile(0.9).values.flatten()
    group_q95 = da_grouped.quantile(0.95).values.flatten()
    
    #--------------------------------------------------------------------
    ## Flatten 2D DataArrays to 1D numpy arrays in order to plot 2d hist
    # Wsgsmax Data
    print('Flatten  arrays')
    wsgs_data = variable.values
    variable = wsgs_data[~np.isnan(wsgs_data)].flatten()         
    # Orographie data
    orog_data = ds_topo.orog.values
    orog_data = orog_data[~np.isnan(orog_data)].flatten() 
    #-------------------------------------------------------------------
    orog_binned = pd.cut(orog_data, len(bins) - 2 , retbins=True)
    
    
    print('Start plotting 2dhistogram')
    fig,ax = plt.subplots(figsize=(10,8))
    s=ax.hist2d(variable, 
                orog_data, 
                bins=[np.linspace(variable.min(),math.ceil(variable.max()/10)*10,50), np.linspace(0,math.ceil(orog_data.max()/100 +1)*100,  50)],
                norm=mpl.colors.LogNorm(), 
                cmap=plt.cm.jet, 
                alpha=0.8)
    plt.colorbar(s[3], ax=ax, label= r'density')
    
    ax.set_xlabel('Wsgsmax [m/s]', fontsize=12)
    ax.set_ylabel('Elevation [m]', fontsize=12)
    ax.set_title(f'{title} binned for elevation')
    
    img_mean=ax.plot(group_mean, orog_binned[1] , 'k-', label='mean', alpha=0.5)
    img25=ax.plot(group_q25, orog_binned[1] , 'w--', label='q25')
    img50=ax.plot(group_q50, orog_binned[1] , 'w-',  label='q50')
    img75=ax.plot(group_q75, orog_binned[1] , 'w--', label='q75')
    img90=ax.plot(group_q90, orog_binned[1] , color='gray', linestyle='-.',linewidth=1.5, label='q90')
    img95=ax.plot(group_q95, orog_binned[1] , color='gray', linestyle=':', linewidth = 2, label='q95')
    
    legend = ax.legend()
    frame = legend.get_frame()
    frame.set_color('lightsteelblue')


    plt.savefig(f'/usr/people/frei/MaxFrei/Max_figures/general_analysis/Height_dependance/{model}_max_wsgsmax_vs_height_hist2d_{domain}', dpi=150)






    df = pd.DataFrame({'wsgsmax':variable, 'orog':orog_data})
    bins = list(np.linspace(0,orog_data.max(),5))
    df['binned'] = pd.cut(df['orog'], bins)

    s = df.groupby(pd.cut(df['orog'], bins=bins)).size()
    grouped = df.groupby(pd.cut(df['orog'], bins=bins))


    fig, axes = plt.subplots(figsize=(8,6), nrows=len(bins)-1,ncols=1, sharex=True)    
    print('Start plotting CDF')
    for df, ax in zip(grouped, enumerate(axes.flatten())):
        if ax[0]==0:
            ax[1].set_title(f'{title} in {domain} binned for elevation')
        ###################################  CDF  ################################################
        # CDFpdf = 'CDF'
        # ax[1].hist(df[1]['wsgsmax'],bins=100, density=True, histtype='step', cumulative=True,
        #             label=f'{round(df[0].left)} to {round(df[0].right)} MASL', log=True)
        ###########################  Create CDF -> 1 - CDF  ######################################
        data_sorted = np.sort(df[1]['wsgsmax'])
        cdf = np.linspace(0,1,len(data_sorted))
        one_minus_cdf = 1-cdf
        ax[1].plot(data_sorted, one_minus_cdf, label=f'{round(df[0].left)} to {round(df[0].right)} MASL')
        # print(f'{round(df[0].left)} to {round(df[0].right)} MASL')
        legend1 = ax[1].legend(fontsize=8, loc='upper right')
        frame1 = legend1.get_frame()
        # frame1.set_color('lightsteelblue')

        ax[1].axvline(df[1]['wsgsmax'].median(), color='black', alpha=0.9, linewidth=1  )#, label='q50')
        ax[1].axvline(df[1]['wsgsmax'].quantile(0.75), color='black', linestyle='--', linewidth=0.75, alpha=0.75)#, label='q75')
        ax[1].axvline(df[1]['wsgsmax'].quantile(0.25), color='black', linestyle='--', linewidth=0.75, alpha =0.75)#, label='q25')
        ax[1].axvline(df[1]['wsgsmax'].quantile(0.90), color='darkgrey' , linestyle='-.', linewidth=1)#, label='q90')
        ax[1].axvline(df[1]['wsgsmax'].quantile(0.95), color='darkgrey' , linestyle=':' , linewidth = 1)#, label='q95')
        ##########################################################################################   
        #ax[1]
        x1,x2,y1,y2 = ax[1].axis()
        
        # ax[1].set_yscale('log')

        ax[1].axis((math.ceil(variable.min()-3),math.ceil(variable.max()+2),y1,1))
        ax[1].set_xlim(20,45)
        ax[1].set_ylim(0,1)
    
    lines = ax[1].get_lines()
    legend2 = ax[1].legend([lines[i] for i in [1,2,4,5]], ["median", "q25, q75", "q90", "q95"], loc='upper left', fontsize=8)
    frame2 = legend2.get_frame()
    frame2.set_color('lightsteelblue')

    
    plt.xlabel(f'Maximal value of wsgsmax for different elevation categories')
    
    plt.savefig(f'/usr/people/frei/MaxFrei/Max_figures/general_analysis/Height_dependance/{model}_max_wsgsmax_vs_BinnedHeight_CDF_{domain}', dpi=150)



    
#%%
#=========================#
## Run plotting function ##
#=========================#
if __name__ == '__main__':    
    plot_height_wsgsmax(stat = 'max_value', model='hist', domain='whole_domain')
    plot_height_wsgsmax(stat = 'max_value', model='hist', domain='NL_domain')

    plot_height_wsgsmax(stat = 'max_value', model='fut', domain='whole_domain')
    plot_height_wsgsmax(stat = 'max_value', model='fut', domain='NL_domain')


#%%
#==============================================================================
# Old Stuff
#==============================================================================
#----------------------------------------
## Select variable that should be plotted

# wsgsmax_AT_max = np.asarray(ds_land.wsgsmax.max(dim='time'))
# AllTime_max = wsgsmax_AT_max[~np.isnan(wsgsmax_AT_max)].flatten()

# p95_dat = np.asarray(ds_p95.wsgsmax.max(dim='time'))
# p95_dat = p95_dat[~np.isnan(p95_dat)].flatten()    # drop all NaN values and flatten 2D arrat to 1D array

# p99_dat = np.asarray(ds_p99.wsgsmax.max(dim='time'))
# p99_dat = p99_dat[~np.isnan(p99_dat)].flatten()    # drop all NaN values and flatten 2D arrat to 1D array

# orog_data = np.asarray(ds_land.orog)
# orog_data = orog_data[~np.isnan(orog_data)].flatten()
# #------------------------------------------------------------------------------



# df = pd.DataFrame({'wsgsmax':variable, 'orog':orog_data})
# df['binned'] = pd.cut(df['orog'], bins)

# df_grouped = df.groupby(pd.cut(df['orog'], bins=bins))
# df_grouped.agg(['mean','median'])

# df_groups = df_grouped.describe()
# df_groups = df_groups['wsgsmax']


# p25 = np.asarray(df_groups['25%'])[0:len(np.asarray(df_groups['25%']))]  # Exclude last value of array since it is weird
# p50 = np.asarray(df_groups['50%'])[0:len(np.asarray(df_groups['25%']))]
# p75 = np.asarray(df_groups['75%'])[0:len(np.asarray(df_groups['25%']))]


# p90 = df_grouped.wsgsmax.quantile(0.9)
# p95 = df_grouped.wsgsmax.quantile(0.95)



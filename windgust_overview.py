#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 08:48:35 2023

@author: frei
"""
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

#%%

#####################################################
### Load daily max files of wsgsmax hourly output ###
#####################################################

data_dir1 = '/net/pc200057/nobackup_1/users/frei/CPMs/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_dailymax_MJJAS'
ds_wsgsmax_dmax_hist = xr.open_mfdataset(f'{data_dir1}/wsgsmax.*.nc')

#data_dir2 = '/net/pc200057/nobackup_1/users/frei/CPM_future/HCLIM38h1_CXFPS_fRACMOfECEARTH_r04_rcp85_wsgsmax_dailymax_MJJAS'
#ds_wsgsmax_dmax_fut = xr.open_mfdataset(f'{data_dir2}/wsgsmax.*.nc')


wsgsmax_hist = ds_wsgsmax_dmax_hist.wsgsmax.values.flatten()
#wsgsmax_fut = ds_wsgsmax_dmax_fut.wsgsmax.values.flatten()

#q99_hist = np.quantile(wsgsmax_hist, 0.99)
#q99_fut = np.quantile(wsgsmax_fut, 0.99)

#%%
#-------------
# Plotting
#-------------
# fig, ax = plt.subplots(figsize=(8,5))
# # gs  = GridSpec(nrows=1, ncols=1, width_ratios=[1, 0, 1])  # arange plot windows


# ax.hist(wsgsmax_hist, bins=100, density=True, histtype='step', label='Historic (1995-2005)')
# ax.hist(wsgsmax_fut, bins=100, density=True, histtype='step' , label='Future (2089-2099)')
# # ax.set_yscale('log')
# ax.set_xlim(0,30)

# ax.axvline(q99_hist, linestyle='--', color='blue', alpha = 0.8, label='Q99 hist')
# ax.axvline(q99_fut, linestyle='--', color='orange', alpha=0.8, label='Q99 future')
# ax.axvline(20, linestyle='--', color='k')

# ax.set_xlabel('wsgsmax [m/s]')
# ax.legend()
# ax.set_title('Histogram of all daily wind speed maxima')

# plt.savefig('/usr/people/frei/MaxFrei/Max_figures/general_analysis/wsgsmax_distri.png', dpi=300)

#%%
#############################################
###   Wind gst versus geostrophic winds   ###
#############################################
##-----------------------------------------------------------------------------
## Load geostrophic data ##
ds_ua1000 = xr.open_mfdataset('/net/pc200057/nobackup_1/users/frei/CPMs/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_uv1000/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_ua1000_dailymax/ua1000.*.nc')
ds_va1000 = xr.open_mfdataset('/net/pc200057/nobackup_1/users/frei/CPMs/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_uv1000/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_va1000_dailymax/va1000.*.nc')

# # Select years from 1996 to 2005 (because in ua and va year 1995 is missing)
# years=[1996+i for i in range(10)]
# ds_ua1000 = ds_ua1000.sel(time=ds_ua1000.time.dt.year.isin(years))
# ds_va1000 = ds_va1000.sel(time=ds_va1000.time.dt.year.isin(years))

## Load the windgust data and delete the first year
da_wsgsmax_dmax_1996 = ds_wsgsmax_dmax_hist.sel(time=ds_wsgsmax_dmax_hist.time.dt.year.isin(years)).wsgsmax
#----------------------------------------------------------------------------##

##------------------------------------------------------------------------------
## Calcualte the resulting geostropic wind on 1000hPa using Pytagoras
da_gswind = np.sqrt(ds_ua1000.ua1000**2 + ds_va1000.va1000**2)
#-----------------------------------------------------------------------------##

#%%

##-----------------------------------------------------------------------------
## Choose year that should be plotted
######################################
plot_year = 2000  # CHANGE YEAR HERE #
######################################
da_gswind_plt = da_gswind.sel(time=da_gswind.time.dt.year.isin(plot_year))
da_wsgsmax_plt = da_wsgsmax_dmax_1996.sel(time=da_wsgsmax_dmax_1996.time.dt.year.isin(plot_year))

## Flatten the arrays and make ready to plot
array_gswind = da_gswind_plt.values.flatten()
array_wsgsmax = da_wsgsmax_plt.values.flatten()

# Assert that both arrays are of equal length 
assert array_gswind.shape[0] == array_wsgsmax.shape[0]


#%%
#------
## Plot

cmap = plt.cm.nipy_spectral     # mpl.cm.viridis    # plt.cm.RdYlBu   
levs = np.round(np.linspace(0,10000, 10)*10)/10                     # np.round(upper_levs/20,1))
norm = mpl.colors.BoundaryNorm(levs,cmap.N)

fig = plt.subplots(figsize=(8,8))
gs  = GridSpec(1, 3,width_ratios=[1,-0.1,0.03])

ax  = plt.subplot(gs[0,0])  # images
cax1 = plt.subplot(gs[0,2])  # colorbar bottom


ax.set_xlim(0,51)
ax.set_ylim(0,51)
ax.set_xlabel('wsgsmax [m/s]')
ax.set_ylabel('gs-wind at 1000 hPa [m/s]')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='k', zorder=2, alpha=0.8)

# ax.scatter(array_wsgsmax , array_gswind , s=1, alpha=0.7)

img1=ax.hist2d(array_wsgsmax , array_gswind , bins=[ np.arange(0,50,1) , np.arange(0,50,1) ],  norm=mpl.colors.LogNorm(), cmap=plt.cm.jet)
plt.colorbar(mappable=img1[3], ax=ax, orientation="vertical", cax=cax1)

plt.savefig('/usr/people/frei/MaxFrei/Max_figures/general_analysis/wsgsmax_vs_gswinds.png', dpi=300)
















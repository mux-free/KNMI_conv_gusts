#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:12:14 2023

@author: frei
"""

import matplotlib
import xarray as xr
import numpy as np
# import dask
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.gridspec import GridSpec


# from matplotlib.colors import from_levels_and_colors
# from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
# import math
# from scipy.odr import Model, Data, ODR
# from scipy.stats import linregress
# from scipy import stats


# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.basemap import Basemap


# from data_processing import sel_month, add_datetime64_column, drop_ocean, downsize_domain
# from plotting_maxf import contour_2_1_plot


# from dask.diagnostics import ProgressBar, progress
# from dask import compute
# import dask.array as da
#%%

#--------------------------------
# Import Files
#--------------------------------

print('Height-dependance for whole domain')
ds_topo = xr.open_dataset('/usr/people/vries/NOBACKUP1/DATA/CORDEX-FPS/HCLIM38h1_CXFPS1999_fRACMO/orog.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.fx.nc')
rootdir = '/net/pc200057/nobackup_1/users/frei/wsgs_overview/MJJAS/stats/field_max'
dir_pctl='/net/pc200057/nobackup_1/users/frei/wsgs_overview/MJJAS/stats/field_pctl'

print('Historic run (1995-2005)')
ds_wsgsmax_hist = xr.open_dataset(f'{rootdir}/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_merged_MJJAS_max.nc')
## Load percentilePlots
ds_wsgsmax_p95_hist = xr.open_dataset(f'{dir_pctl}/pctl/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_MJJAS_P95.nc')
ds_wsgsmax_p99_hist = xr.open_dataset(f'{dir_pctl}/pctl/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_MJJAS_P99.nc')

print('Future run (2089-2099')
ds_wsgsmax_fut = xr.open_dataset(f'{rootdir}/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_fut_wsgsmax_merged_MJJAS_max.nc')
## Load percentilePlots
ds_wsgsmax_p95_fut = xr.open_dataset(f'{dir_pctl}/pctl_fut/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_fut_wsgsmax_MJJAS_P95.nc')
ds_wsgsmax_p99_fut = xr.open_dataset(f'{dir_pctl}/pctl_fut/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_fut_wsgsmax_MJJAS_P99.nc')


print('Height-dependance for NL-domain')
rootdir = '/net/pc200057/nobackup_1/users/frei/wsgs_overview/MJJAS/stats_lonlatbox'
ds_topo_nl = xr.open_dataset(f'{rootdir}/orog.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.nc')
dir_pctl = '/net/pc200057/nobackup_1/users/frei/wsgs_overview/MJJAS/stats_lonlatbox/field_pctl'
print('Historic run (1995-2005)')
ds_wsgsmax_nl_hist = xr.open_dataset(f'{rootdir}/field_max/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_merged_MJJAS_lonlatbox_max.nc')
ds_wsgsmax_p95_nl_hist = xr.open_dataset(f'{dir_pctl}/pctl/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_MJJAS_lonlatbox_P95.nc')
ds_wsgsmax_p99_nl_hist = xr.open_dataset(f'{dir_pctl}/pctl/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_wsgsmax_MJJAS_lonlatbox_P99.nc')

print('Future run (2089-2099)')
ds_wsgsmax_nl_fut = xr.open_dataset(f'{rootdir}/field_max/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_fut_wsgsmax_merged_MJJAS_lonlatbox_max.nc')
ds_wsgsmax_p95_nl_fut = xr.open_dataset(f'{dir_pctl}/pctl_fut/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_fut_wsgsmax_MJJAS_lonlatbox_P95.nc')
ds_wsgsmax_p99_nl_fut = xr.open_dataset(f'{dir_pctl}/pctl_fut/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_fut_wsgsmax_MJJAS_lonlatbox_P99.nc')



#%%
#------------------------------------------------#
# Create Plot -> Orography + subdomain rectangle #
#------------------------------------------------#
# Extract the 2D arrays of longitutde and latitude
array_lon = ds_topo.lon.values
array_lat = ds_topo.lat.values

array_lon_nl = ds_topo_nl.lon.values
array_lat_nl = ds_topo_nl.lat.values

#%%


# Drop ocean in ds_topo
da_topo_land = drop_ocean(ds_topo.orog)

X,Y = da_topo_land.lon, da_topo_land.lat
fig = plt.subplots(figsize=(8,5))

# Only 2 colorbar for top 4 figures
gs  = GridSpec(nrows=1, ncols=3, width_ratios=[1, -0.1, 0.02])  # arange plot windows


upper_levs = np.ceil(da_topo_land.max()/1000)*1000
levs = np.round(np.linspace(0,upper_levs, 9)*10)/10                     # np.round(upper_levs/20,1))


# Define location of plot (ax) and colorbar (cax)
cax1 = plt.subplot(gs[0,2])  # colorbar below top right fig
ax  = plt.subplot(gs[0,0], projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

#--------------------------------------------------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,da_topo_land, extend='max', levels=levs, transform=ccrs.PlateCarree())
plt.colorbar(s,label='Elevation [m]',ax=ax,orientation="vertical",cax=cax1)      ### Took out formatter: ,format='%3i'


# ax.set_title('Topography map')
# Recatngle for bigger domain
linsty ='--' 
linwi=1
ax.plot(array_lon[:,0], array_lat[:,0],  transform=ccrs.PlateCarree(), linewidth=linwi, linestyle= linsty,  color='k')
ax.plot(array_lon[:,-1], array_lat[:,-1],  transform=ccrs.PlateCarree(), linewidth=linwi, linestyle=linsty,  color='k')
ax.plot(array_lon[0,:], array_lat[0,:],  transform=ccrs.PlateCarree(), linewidth=linwi, linestyle= linsty,   color='k')
ax.plot(array_lon[-1,:], array_lat[-1,:],  transform=ccrs.PlateCarree(), linewidth=linwi, linestyle=linsty,    color='k')

ax.text(0.03,0.4,'HARMONIE \ndomain', bbox=dict(facecolor='green', alpha=0.5), transform=ax.transAxes)

ax.text(0.5,0.88,'Subdomain', bbox=dict(facecolor='green', alpha=0.5), transform=ax.transAxes)

# Rectangle for subdomain
ax.plot(array_lon_nl[:,0], array_lat_nl[:,0],  transform=ccrs.PlateCarree(), linewidth=2, color='k')
ax.plot(array_lon_nl[:,-1], array_lat_nl[:,-1],  transform=ccrs.PlateCarree(), linewidth=2, color='k')
ax.plot(array_lon_nl[0,:], array_lat_nl[0,:],  transform=ccrs.PlateCarree(), linewidth=2, color='k')
ax.plot(array_lon_nl[-1,:], array_lat_nl[-1,:],  transform=ccrs.PlateCarree(), linewidth=2, color='k')


ax.set_extent([
    da_topo_land.lon.min() - (da_topo_land.lon.max() - da_topo_land.lon.min()) * 1/30 ,          #lonmin
    da_topo_land.lon.max() + (da_topo_land.lon.max() - da_topo_land.lon.min()) * 1/30 ,          #lonmax 
    da_topo_land.lat.min() - (da_topo_land.lat.max() - da_topo_land.lat.min()) * 1/50 ,          #latmin
    da_topo_land.lat.max() + (da_topo_land.lat.max() - da_topo_land.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )

plt.savefig('/usr/people/frei/MaxFrei/Max_figures/general_analysis/domain_selection/domains_topo.png', dpi=300)

#%%
#----------------------------------------------#
# Create plots - MaxVal, P99 and MaxVal / P99  #
#----------------------------------------------#
# Define colormap
cmap = plt.cm.nipy_spectral     # mpl.cm.viridis    # plt.cm.RdYlBu   
# Define levels of colorbar
upper_levs = 40
levs = np.round(np.linspace(0,upper_levs, 9)*10)/10                     # np.round(upper_levs/20,1))
norm = mpl.colors.BoundaryNorm(levs,cmap.N)
# Define meshgrid
X,Y = ds_wsgsmax_hist.wsgsmax.lon, ds_wsgsmax_hist.wsgsmax.lat

### Open Plot ###
fig = plt.subplots(figsize=(15,15))

# Only 2 colorbar for top 4 figures
gs  = GridSpec(nrows=3, ncols=4,height_ratios=[1,1,1], width_ratios=[1,1, -0.15 , 0.02])  # arange plot windows


# Define colorbar locations
cax1 = plt.subplot(gs[0,3])  # colorbar below top right fig
cax2 = plt.subplot(gs[1,3])  # colorbar below bottom right fig
cax3 = plt.subplot(gs[2,3])

#=================================================================================================================================
# Figure top left
#----------------
ax  = plt.subplot(gs[0,0],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

#--------------------------------------------------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,ds_wsgsmax_hist.wsgsmax[0,:,:], cmap=cmap,levels=levs,norm=norm,extend='max',transform=ccrs.PlateCarree())
ax.set_title('Maximal wind gust (1995-2005)')
#plt.colorbar(s,label='Wsgsmax [m/s]',ax=ax,orientation="horizontal",cax=cax1)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_fut.lon.min() - (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_fut.lon.max() + (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_fut.lat.min() - (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_fut.lat.max() + (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------

#=================================================================================================================================
# Figure top right
#-------------------
ax  = plt.subplot(gs[0,1],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,ds_wsgsmax_fut.wsgsmax[0,:,:], cmap=cmap,levels=levs,norm=norm,extend='max',transform=ccrs.PlateCarree())
ax.set_title('Maximal wind gust (2089-2099)')
plt.colorbar(s,label='Wsgsmax [m/s]',ax=ax,orientation="vertical",cax=cax1)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_fut.lon.min() - (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_fut.lon.max() + (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_fut.lat.min() - (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_fut.lat.max() + (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------


#=================================================================================================================================
# Figure middle left
#-------------------
ax  = plt.subplot(gs[1,0],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,ds_wsgsmax_p99_hist.wsgsmax[0,:,:], cmap=cmap,levels=levs,norm=norm,extend='max',transform=ccrs.PlateCarree())
ax.set_title('P99 wingust (1995-2005)')
# plt.colorbar(s,label='Wsgsmax [m/s]',ax=ax,orientation="horizontal",cax=cax3)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_fut.lon.min() - (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_fut.lon.max() + (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_fut.lat.min() - (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_fut.lat.max() + (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------


#=================================================================================================================================
# Figure middle right
#--------------------
ax  = plt.subplot(gs[1,1],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,ds_wsgsmax_p99_fut.wsgsmax[0,:,:], cmap=cmap,levels=levs,norm=norm,extend='max',transform=ccrs.PlateCarree())
ax.set_title('P99 wingust (2089-2099)')
plt.colorbar(s,label='Wsgsmax [m/s]',ax=ax,orientation="vertical",cax=cax2)      ### Took out formatter: ,format='%3i'

ax.set_extent([
    ds_wsgsmax_p99_fut.lon.min() - (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_fut.lon.max() + (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_fut.lat.min() - (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_fut.lat.max() + (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------



##########################
### Define new levels! ###
##########################
upper_level1 = 3
levs1 = np.linspace(0, upper_level1, upper_level1*2+1)                     # np.round(upper_levs/20,1))
norm1 = mpl.colors.BoundaryNorm(levs1,cmap.N)


#=================================================================================================================================
# Figure bottom left
#-------------------
ax  = plt.subplot(gs[2,0],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y, ds_wsgsmax_hist.wsgsmax[0,:,:] / ds_wsgsmax_p99_hist.wsgsmax[0,:,:]  , levels=levs1, extend='max', transform=ccrs.PlateCarree())
ax.set_title('Maximal WG / P99 WG (1995-2005)')
# plt.colorbar(s,label='Maximal WG / P99 WG',ax=ax,orientation="horizontal",cax=cax)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_fut.lon.min() - (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_fut.lon.max() + (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_fut.lat.min() - (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_fut.lat.max() + (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------


#=================================================================================================================================
# Figure bottom right
#--------------------
ax  = plt.subplot(gs[2,1],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,  ds_wsgsmax_fut.wsgsmax[0,:,:] / ds_wsgsmax_p99_fut.wsgsmax[0,:,:] , levels=levs1,  extend='max',transform=ccrs.PlateCarree())
ax.set_title('Maximal WG / P99 WG (2089-2099)')
plt.colorbar(s,label='Maximal WG / P99 WG',ax=ax,orientation="vertical",cax=cax3)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_fut.lon.min() - (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_fut.lon.max() + (ds_wsgsmax_p99_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_fut.lat.min() - (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_fut.lat.max() + (ds_wsgsmax_p99_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------

plt.savefig('/usr/people/frei/MaxFrei/Max_figures/general_analysis/domain_selection/Big_domain.png', dpi=300)






#%%
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                          NL DOMAIN                                                                      #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#%%
#------------------
## Create plots
#------------------
# Define colormap
cmap = plt.cm.nipy_spectral     # mpl.cm.viridis    # plt.cm.RdYlBu   
# Define levels of colorbar
upper_levs = 40
levs = np.round(np.linspace(0,upper_levs, 9)*10)/10                     # np.round(upper_levs/20,1))
norm = mpl.colors.BoundaryNorm(levs,cmap.N)
# Define meshgrid
X,Y = ds_wsgsmax_nl_hist.wsgsmax.lon, ds_wsgsmax_nl_hist.wsgsmax.lat

# Set plot extend
lon_min, lon_max = ds_wsgsmax_p99_nl_fut.lon.min(), ds_wsgsmax_p99_nl_fut.lon.max()
lat_min, lat_max  = ds_wsgsmax_p99_nl_fut.lat.min(), ds_wsgsmax_p99_nl_fut.lat.max()

### Open Plot ###
fig = plt.subplots(figsize=(15,15))

# Only 2 colorbar for top 4 figures
gs  = GridSpec(nrows=3, ncols=4,height_ratios=[1,1,1], width_ratios=[1,1, -0.15 , 0.02])  # arange plot windows


# Define colorbar locations
cax1 = plt.subplot(gs[0,3])  # colorbar below top right fig
cax2 = plt.subplot(gs[1,3])  # colorbar below bottom right fig
cax3 = plt.subplot(gs[2,3])

#=================================================================================================================================
# Figure top left
#----------------
ax  = plt.subplot(gs[0,0],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

#--------------------------------------------------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,ds_wsgsmax_nl_hist.wsgsmax[0,:,:], cmap=cmap,levels=levs,norm=norm,extend='max',transform=ccrs.PlateCarree())
ax.set_title('Maximal wind gust (1995-2005)')
#plt.colorbar(s,label='Wsgsmax [m/s]',ax=ax,orientation="horizontal",cax=cax1)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_nl_fut.lon.min() - (ds_wsgsmax_p99_nl_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_nl_fut.lon.max() + (ds_wsgsmax_p99_nl_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_nl_fut.lat.min() - (ds_wsgsmax_p99_nl_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_nl_fut.lat.max() + (ds_wsgsmax_p99_nl_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------

#=================================================================================================================================
# Figure top right
#-------------------
ax  = plt.subplot(gs[0,1],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,ds_wsgsmax_nl_fut.wsgsmax[0,:,:], cmap=cmap,levels=levs,norm=norm,extend='max',transform=ccrs.PlateCarree())
ax.set_title('Maximal wind gust (2089-2099)')
plt.colorbar(s,label='Wsgsmax [m/s]',ax=ax,orientation="vertical",cax=cax1)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_nl_fut.lon.min() - (ds_wsgsmax_p99_nl_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_nl_fut.lon.max() + (ds_wsgsmax_p99_nl_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_nl_fut.lat.min() - (ds_wsgsmax_p99_nl_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_nl_fut.lat.max() + (ds_wsgsmax_p99_nl_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------


#=================================================================================================================================
# Figure middle left
#-------------------
ax  = plt.subplot(gs[1,0],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,ds_wsgsmax_p99_nl_hist.wsgsmax[0,:,:], cmap=cmap,levels=levs,norm=norm,extend='max',transform=ccrs.PlateCarree())
ax.set_title('P99 wingust (1995-2005)')
# plt.colorbar(s,label='Wsgsmax [m/s]',ax=ax,orientation="horizontal",cax=cax3)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_nl_fut.lon.min() - (ds_wsgsmax_p99_nl_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_nl_fut.lon.max() + (ds_wsgsmax_p99_nl_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_nl_fut.lat.min() - (ds_wsgsmax_p99_nl_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_nl_fut.lat.max() + (ds_wsgsmax_p99_nl_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------


#=================================================================================================================================
# Figure middle right
#--------------------
ax  = plt.subplot(gs[1,1],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,ds_wsgsmax_p99_nl_fut.wsgsmax[0,:,:], cmap=cmap,levels=levs,norm=norm,extend='max',transform=ccrs.PlateCarree())
ax.set_title('P99 wingust (2089-2099)')
plt.colorbar(s,label='Wsgsmax [m/s]',ax=ax,orientation="vertical",cax=cax2)      ### Took out formatter: ,format='%3i'

ax.set_extent([
    ds_wsgsmax_p99_nl_fut.lon.min() - (ds_wsgsmax_p99_nl_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_nl_fut.lon.max() + (ds_wsgsmax_p99_nl_fut.lon.max() - df.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_nl_fut.lat.min() - (ds_wsgsmax_p99_nl_fut.lat.max() - df.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_nl_fut.lat.max() + (ds_wsgsmax_p99_nl_fut.lat.max() - df.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------



##########################
### Define new levels! ###
##########################
upper_level1 = 3
levs1 = np.linspace(0, upper_level1, upper_level1*2+1)                     # np.round(upper_levs/20,1))
norm1 = mpl.colors.BoundaryNorm(levs1,cmap.N)


#=================================================================================================================================
# Figure bottom left
#-------------------
ax  = plt.subplot(gs[2,0],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y, ds_wsgsmax_nl_hist.wsgsmax[0,:,:] / ds_wsgsmax_p99_nl_hist.wsgsmax[0,:,:]  , levels=levs1, extend='max', transform=ccrs.PlateCarree())
ax.set_title('Maximal WG / P99 WG (1995-2005)')
# plt.colorbar(s,label='Maximal WG / P99 WG',ax=ax,orientation="horizontal",cax=cax)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_nl_fut.lon.min() - (ds_wsgsmax_p99_nl_fut.lon.max() - ds_wsgsmax_p99_nl_fut.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_nl_fut.lon.max() + (ds_wsgsmax_p99_nl_fut.lon.max() - ds_wsgsmax_p99_nl_fut.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_nl_fut.lat.min() - (ds_wsgsmax_p99_nl_fut.lat.max() - ds_wsgsmax_p99_nl_fut.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_nl_fut.lat.max() + (ds_wsgsmax_p99_nl_fut.lat.max() - ds_wsgsmax_p99_nl_fut.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------


#=================================================================================================================================
# Figure bottom right
#--------------------
ax  = plt.subplot(gs[2,1],projection=ccrs.PlateCarree())  # images
# map
ax.coastlines(linewidth=1)
ax.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
# set the ticks
ax.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
ax.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
# format the ticks as e.g 60302260W
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
#---------------------------------
# plot contour lines wsgsmax & SLP
s= ax.contourf(X,Y,  ds_wsgsmax_nl_fut.wsgsmax[0,:,:] / ds_wsgsmax_p99_nl_fut.wsgsmax[0,:,:] , levels=levs1,  extend='max',transform=ccrs.PlateCarree())
ax.set_title('Maximal WG / P99 WG (2089-2099)')
plt.colorbar(s,label='Maximal WG / P99 WG',ax=ax,orientation="vertical",cax=cax3)      ### Took out formatter: ,format='%3i'
ax.set_extent([
    ds_wsgsmax_p99_nl_fut.lon.min() - (ds_wsgsmax_p99_nl_fut.lon.max() - ds_wsgsmax_p99_nl_fut.lon.min()) * 1/30 ,          #lonmin
    ds_wsgsmax_p99_nl_fut.lon.max() + (ds_wsgsmax_p99_nl_fut.lon.max() - ds_wsgsmax_p99_nl_fut.lon.min()) * 1/30 ,          #lonmax 
    ds_wsgsmax_p99_nl_fut.lat.min() - (ds_wsgsmax_p99_nl_fut.lat.max() - ds_wsgsmax_p99_nl_fut.lat.min()) * 1/50 ,          #latmin
    ds_wsgsmax_p99_nl_fut.lat.max() + (ds_wsgsmax_p99_nl_fut.lat.max() - ds_wsgsmax_p99_nl_fut.lat.min()) * 1/50            #latmax
               ], 
              ccrs.PlateCarree() )
#---------------------------------------------------------------------------------------------------------------------------------

plt.savefig('/usr/people/frei/MaxFrei/Max_figures/general_analysis/domain_selection/NL_domain.png', dpi=300)

#%%






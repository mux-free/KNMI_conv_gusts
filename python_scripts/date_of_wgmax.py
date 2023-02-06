#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:16:21 2022

@author: frei
"""

#%%
 #-------------------------------------------------
 # Define functions to get dates of max windgusts
 #-------------------------------------------------
 # Functions convert the xr.Dataset to a pd.Dataframe object 
 # then extract timestep of maximum wsgsmax for each lat-lon(-year)

def convert_to_df(ds):
     df = ds.to_dataframe()
     df.reset_index(inplace=True)
     return df

def calculate_year_month_day_cols(df):
     assert 'time' in df.columns, f"time should be in df.columns. Currently: {[c for c in df.columns]}"
     df['year'] = df.time.map(lambda x: x.year)
     df['month'] = df.time.map(lambda x: x.month)
     df['day'] = df.time.map(lambda x: x.day)
     return df

def calculate_day_of_max_value(df, value_col):
     """
     Arguments
     ---------
     df : pd.DataFramedf = calculate_day_of_max_value(df, value_col=variable)
         dataframe converted from xarray with ['lat','lon', 'year', value_col] columns

     value_col : str
         column that you want to find the month of maximum for 
         e.g. Which month (int) in each pixel (lat,lon) has the highest runoff
     """
     #------------------------------------------------------------------------
     # Drop all NaN
     df = df.dropna(how='any')
     #------------------------------------------------------------------------
     # max_days = df.loc[df.groupby(["lat","lon","year"])[value_col].idxmax()]
     max_days = df.loc[df.groupby(["lat","lon"])[value_col].idxmax()]
     return max_days

def convert_dataframe_to_xarray(df, index_cols=['lat','lon']):
     """
     Arguments
     ---------
     df: pd.DataFrame
         the dataframe to convert to xr.dataset

     index_cols: List[str]
         the columns that will become the coordinates 
         of the output xr.Dataset
     Returns
     -------1999-09-30T12:00:00.000000000
     xr.Dataset
     """
     out = df.set_index(index_cols).dropna()
     ds = out.to_xarray()
     ds = ds.set_coords(('lat','lon'))
     return ds

def calculate_annual_day_of_max(ds, variable, threshold = None):
     """for the `variable` in the `ds` calculate the 
     month of maximum for a given pixel-year.
     Returns:
     -------
     xr.Dataset
     """
     if threshold != None:
         ds = ds.where(ds.wsgsmax > threshold)
     # convert to a dataframe
     df = convert_to_df(ds)
     df = calculate_year_month_day_cols(df)
     # calculate the month of maximum
     df = calculate_day_of_max_value(df, value_col=variable)
     # reconstitute the dataframe object
     #ds_out = convert_dataframe_to_xarray(df, index_cols=['lat','lon','year','month'])
     #ds_out = convert_dataframe_to_xarray(df, index_cols=['lat','lon'])
     return df
    
 
#with dask.config.set(**{'array.slicing.split_large_chunks': False}):
#    array[indexer]

#%%

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap
import math


from data_processing import downsize_domain, sel_month
import plotting_maxf

#%%
#--------------------------------
# Load Data
#--------------------------------
rootdir = '/net/pc200057/nobackup_1/users/frei/CPMs/'
#------------------------------------------------------------------------------
## Read daily Max data (~7.5 GB)
ds_dMax = xr.open_mfdataset(f'{rootdir}/HCLIM_r14_dailymax/*.nc')
#-----------bla-------------------------------------------------------------------
## total land = ocean map (ocean+==100 && land==0)
ds_land_ocean= xr.open_dataset(f'/usr/people/vries/NOBACKUP1/DATA/CORDEX-FPS/HCLIM38h1_CXFPS1999_fRACMO/sftof.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.fx.nc')
#------------------------------------------------------------------------------
ds_dMax = sel_month(ds_dMax)
#%%
#--------------------------------
# Drop all values over ocean                <---- Make a fucntion in data_processing out of this!!
#--------------------------------
# reduce the domai to nrht of alps-Europe
ds_dMax = downsize_domain(ds_dMax, lon_min=3, lon_max=13, lat_min=47, lat_max=54)

ds_dMax['sftof'] = ds_land_ocean['sftof']
#ds_dMax['orog'] = ds_topo['orog']
ds_land = ds_dMax.where(ds_dMax.sftof < 1)
#%%

#ds_test = ds_land.sel(time=slice('1995-05-01T12:00:00.000000000', '1996-09-30T12:00:00.000000000'))
ds_test = ds_land.sel(time=slice('1995-05-01T12:00:00.000000000', '1995-09-30T12:00:00.000000000'))
ds_test = ds_test.dropna(dim='x', subset=['wsgsmax'], how='all')
ds_test = ds_test.dropna(dim='y', subset=['wsgsmax'], how='all')
ds_test = ds_test.dropna(dim='time', subset=['wsgsmax'], how='all')

# Extract what year is shown
year_shown = pd.DatetimeIndex([ds_test.time.data[0]]).year
#%%
#---------------------------------------
### Apply functions
#---------------------------------------

#-------------
threshold = 15
#-------------

date_of_max = calculate_annual_day_of_max(ds_test, variable='wsgsmax', threshold=threshold)
#%%
# Convert df to xarray_ds
ds_date = convert_dataframe_to_xarray(date_of_max, index_cols=['y','x'])
ds_date = ds_date.set_coords(('lat','lon'))


da_datemax = ds_date.date.astype('float') -19950000
# Apply plot function

# date_array = ds_date.date.data.astype('float') -19950000
# plt.contourf(date_array)

#%%

input_array = da_datemax.data
label_cbar = 'date'

cmap = plt.cm.nipy_spectral     # mpl.cm.viridis    # plt.cm.RdYlBu
    
    
upper_levs = np.around(da_datemax.max(),1)
upper_levs = np.around(da_datemax.min(),1)

levs = np.linspace( 500, da_datemax.max() , 20 )


norm = mpl.colors.BoundaryNorm(levs,cmap.N)

X,Y = da_datemax.lon, da_datemax.lat

### Open Plot ###
fig,ax = plt.subplots()
gs  = GridSpec(3, 1,height_ratios=[15,-0.5,1.2])  # arange plot windows
ax  = plt.subplot(gs[0,0],projection=ccrs.PlateCarree())  # images
cax1 = plt.subplot(gs[2,0])  # colorbar bottom

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
img=ax.contourf(X,Y,input_array,cmap=cmap,levels=levs,norm=norm,extend='both',transform=ccrs.PlateCarree())
plt.colorbar(img, label=label_cbar, ax=ax, orientation="horizontal", cax=cax1)      ### Took out formatter: ,format='%3i'

ax.set_title(f'Date of max wingust during {year_shown[0]} with a threshold of {threshold} m/s')

df = da_datemax.to_dataframe().dropna()

ax.set_extent([
    df.lon.min() - (df.lon.max() - df.lon.min()) * 1/10 ,          #lonmin
    df.lon.max() + (df.lon.max() - df.lon.min()) * 1/10 ,          #lonmax 
    df.lat.min() - (df.lat.max() - df.lat.min()) * 1/10 ,          #latmin
    df.lat.max() + (df.lat.max() - df.lat.min()) * 1/10            #latmax
               ], 
              ccrs.PlateCarree()
              )

#--------------------------------------------------------------------------
#plt.savefig(f'/net/pc200057/nobackup_1/users/frei/CPMs/figures/misc/date_of_max_wg_{threshold}', dpi=500)
plt.savefig(f'/net/pc200057/nobackup_1/users/frei/CPMs/figures/misc/date_of_max_wg_{threshold}_year[year_shown]', dpi=500)



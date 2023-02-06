#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 13:50:16 2022

@author: frei
"""
import xarray as xr
import numpy as np
import pandas as pd




#%%
#-----------------------------------------------------#
## DO NOT USE THIS ANYMORE, DOWNSIZE DOMAIN WITH CDO!!#
#-----------------------------------------------------#
def downsize_domain(ds, lon_min=3, lon_max=13, lat_min=48, lat_max=55):
    ds = ds.where((ds.lon > lon_min) & (ds.lon < lon_max) & (ds.lat > lat_min) & (ds.lat < lat_max))

    return ds

#%%

def sel_month(input_ds, months = [5,6,7,8,9]):
    input_ds = input_ds.sel(time=input_ds.time.dt.month.isin(months))  
    return input_ds
#%%
#------------------------------------------------------------------------------
# ## total land = ocean map (ocean==100 && land==0)
ds_land_ocean= xr.open_dataset(f'/usr/people/vries/NOBACKUP1/DATA/CORDEX-FPS/HCLIM38h1_CXFPS1999_fRACMO/sftof.clim.CXFPS025.HCLIM38h1_CXFPS1999_fRACMO.fx.nc')

def drop_ocean(ds):
    ds['sftof'] = ds_land_ocean['sftof']
    ds = ds.where(ds.sftof < 1)
    return ds


#%%
def purely_indomain(df):
    #return df.where(df['touchb']=='F').dropna(how='all')
     pass
 

def to_df_dropna(ds):
    df = ds.to_dataframe().dropna()
    return ds

    
#%%

def add_datetime64_column(df):
    """
    Parameters
    ----------
    df : TYPE: Pandas dataframe
        Output of cell tracker algorithm with a date(YYYMMDD) and time(hhmmss) column.

    Returns
    -------
    DataFrame with a numpy datetime64 colume, a ceiled (next full hour) and floored (prior full hour) datetime64 column

    """
    # # Create an array that will be manipulated
    # date_array = (df['date(YYYYMMDD)']*1e6 + df['time(hhmmss)']).astype(str) 
    # # Add ccolumn for numpy datetime64 
    # df['datetime'] = np.nan
    # for i in range(len(df['datetime'])):
    #     df.loc[i,'datetime'] = np.datetime64(date_array.iloc[i][:4] + '-' + date_array.iloc[i][4:6] + '-' + date_array.iloc[i][6:8] + ' ' + date_array.iloc[i][8:10] + ':' + date_array.iloc[i][10:12] + ':00.000000000')

    df['datetime'] = ( pd.to_numeric(df['date(YYYYMMDD)'])*1e6 + pd.to_numeric(df['time(hhmmss)']) ).astype(str) 
    for idx, datetime in enumerate(df['datetime']):
        df.loc[idx,'datetime'] = np.datetime64(datetime[:4] + '-' + datetime[4:6] + '-' + datetime[6:8] + ' ' + datetime[8:10] + ':' + datetime[10:12] + ':00.000000000')



    # Loop through df to convert strig to np.datetime64
    
    df['datetime_ceil'] = df['datetime'] + np.timedelta64(30,'m')
    df['datetime_floor'] = df['datetime'] - np.timedelta64(30,'m')
    
    return df

#%%



def read_df_cell(csv_dir):
    df_csv_cell = pd.read_csv(csv_dir)
    df_csv_cell['datetime'] = pd.to_datetime(df_csv_cell['datetime'])
    df_csv_cell['datetime_ceil'] = pd.to_datetime(df_csv_cell['datetime_ceil'])
    df_csv_cell['datetime_floor'] = pd.to_datetime(df_csv_cell['datetime_floor'])
    return df_csv_cell
















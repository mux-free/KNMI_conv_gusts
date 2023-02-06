#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:40:17 2022

@author: frei
"""
# Important libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from matplotlib.gridspec import GridSpec
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm


#from mpl_toolkits.basemap import Basemap
#from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
from scipy import stats

import os



def contour_2_1_plot(
        da,
        label_cbar = 'some label',
        contour1d = False,
        da2 = None,
        contour_c = None,
        fig_path = None,
        fig_name = None,
        save_figure=True,
        **kwargs,
        #**fig_title
        ):
    """
    Parameters
    ----------
    da : <xarray.DataArray>
        Xarray DataArray with the variable of intereest that will be plotted (has to be 2D)
    label_cbar : Type: string, optional
        Colorbar label. The default is 'some label'.
    contour1d : True or False (Boolean), optional
        If True, contourlines will be plotted. The default is False.
    da2 : Same type as da, optional
        The filed plotted as contour lines. The default is False.
    
    Returns
    kwargs:
        max_val
    -------
    None.
    """
    #--------------------------------------------------------------------------
    # Process dataset, to get an np.array that can be plotted using matplotlib

    try: 
        input_array = da.data[0,:,:] 
    except IndexError:
        input_array = da.data
    
    

    if da2 is None:
        print('No contour lines will be plotted')
    else:
        input_array2 = da2.data
        print('Shape of contour-lines is: ', da2.data.shape)

    #--------------------------------------------------------------------------
    #Plotting fucntions:
    #-------------------
    
    cmap = plt.cm.nipy_spectral     # mpl.cm.viridis    # plt.cm.RdYlBu   
    upper_levs = kwargs.get('max_val', np.around(da.max(),1))
    levs = np.round(np.linspace(0,upper_levs, 10)*10)/10                     # np.round(upper_levs/20,1))
    norm = mpl.colors.BoundaryNorm(levs,cmap.N)

    X,Y = da.lon, da.lat
    
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
    
    
    
    
    #-------------
    # Title
    
    ax.set_title(kwargs.get('fig_title'))

    #--------------------------------------------------------------------------
    # plot contour lines wsgsmax & SLP
    img=ax.contourf(X,Y,input_array,cmap=cmap,levels=levs,norm=norm,extend='both',transform=ccrs.PlateCarree())
    plt.colorbar(img,label=label_cbar,ax=ax,orientation="horizontal",cax=cax1)      ### Took out formatter: ,format='%3i'
    
    if contour1d == True:
        cont_dens = 6
        cont_lines = cont_dens + 1
        
        if contour_c != None:
            color = contour_c
        else:
            color = 'k'    
        
        C=ax.contour(X,Y , input_array2 , levels=np.linspace(0,input_array2.max(),cont_lines) , fontproperties = 10,
                     linewidths=1, linestyles ='--', colors=color, alpha=1 , transform=ccrs.PlateCarree())
        
        c_label = Fa30lse
        if c_label == True:
            plt.clabel(C, inline=1, fontsize=10,fmt='%i')
   
    #--------------------------------------------------------------------------
    # Set extend of plot-window    
    df = da.to_dataframe().dropna()
    ax.set_extent([
        df.lon.min() - (df.lon.max() - df.lon.min()) * 1/20 ,          #lonmin
        df.lon.max() + (df.lon.max() - df.lon.min()) * 1/20 ,          #lonmax 
        df.lat.min() - (df.lat.max() - df.lat.min()) * 1/20 ,          #latmin
        df.lat.max() + (df.lat.max() - df.lat.min()) * 1/20            #latmax
                   ], 
                  ccrs.PlateCarree()
                  )
    
    
    if save_figure == True:
        #--------------------------------------------------------------------------
        # Save figure   
        rootdir1 = '/net/pc200057/nobackup_1/users/frei/'
        if fig_path == None:
            plt.savefig(f'/usr/people/frei/MaxFrei/Max_figures/overview/fig', dpi=500)
        else:
            save_path = f'/usr/people/frei/MaxFrei/Max_figures/overview/{fig_path}'
            # Create outpath if it doesn't exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig( save_path+fig_name , dpi=500 )

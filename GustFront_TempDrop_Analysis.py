#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:01:07 2023

@author: frei
"""

#################################################
# Compute median, 75 and 25 percentile          #
#################################################
def percentile(n):
    def percentile_(pct):
        return pct.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_


def plot_2dhist_quantiles(x,y,df, cax=None, xtitle = False, ytitle=False, plot_dist = True,color='k', **kwargs):
    """
    Parameters
    ----------
    x : TYPE
        pandas Series or 1d npArray
    y : TYPE
        pandas Series or 1d npArray
    cax : TYPE, optional
        Specify if plot should be accompannied by colorbar and specify place in gs. The default is None.
    xtitle : TYPE, Boolean
        True/False if there is a x-axis title. The default is False.
    ytitle : TYPE, Boolean
        True/False if there is a y-axis title. The default is False.
    **kwargs : TYPE
        bin_width = width of bins in K
        bin_height = heihgt of bins in m/s
        v_max = max vlaue of colorbar

    Returns
    -------
    TYPE
        Plot: hist2d with 25th, 50th and 75th percentile line for those coulmns with more than 50 measurments
    """

    ##############################################################
    ## Adjust bin size to total points in df and colorbar range ##
    ##############################################################
    
    delta_xbin = 1
    delta_xbin = kwargs.get('bin_width')
    delta_ybin = 1
    delta_ybin = kwargs.get('bin_height')

    y_bins = np.arange(20,45.5,delta_ybin)
    
    #--------------------------------------------------------
    ### Group data by temperature drop magnitude into 50 bins
    x_bins = np.arange(-17,8,delta_xbin) 
    dtas_cell_grouped = df.groupby(pd.cut(x, bins=x_bins)) 
    wsgsmax_mean_stats = dtas_cell_grouped['peakVal'].agg([percentile(0.25), percentile(0.50), percentile(0.75)])
    wsgsmax_mean_stats_90 = dtas_cell_grouped['peakVal'].agg([percentile(0.90)])
    
    ### Add sizes of each dTAS bin and drop the bins with less than a certain number of values
    quantile_cutoff = 50
    quantile_cutoff = kwargs.get('quantile_cutoff')
    quantile_cutoff_90 = 100
    
    # Check if there are enough values in column to compute the quantile
    wsgsmax_mean_stats['size'] = dtas_cell_grouped.size().values
    wsgsmax_mean_stats.where(wsgsmax_mean_stats['size']>quantile_cutoff, inplace = True)
    wsgsmax_mean_stats_90['size'] = dtas_cell_grouped.size().values
    wsgsmax_mean_stats_90.where(wsgsmax_mean_stats_90['size']>quantile_cutoff_90, inplace = True)

    # Create an array with beginnings of intervals for the plot
    bla_x = pd.cut(y, bins=x_bins, retbins=True)
    #---------------------------------------------------------
 
    v_max = 1e3
    v_max = kwargs.get('v_max')
    #################################################
    # PLOT
    #################################################
    #--------------------------------------------------------------------------
    # Show distrbuition part of plot
    #------------------------------------------------    
    if plot_dist == True:
        img=ax.hist2d(x, 
                      y, 
                      bins=[x_bins ,y_bins], #np.arange(3,10,0.12)],
                      norm=mpl.colors.LogNorm(vmax=v_max), 
                      cmap=plt.cm.jet, 
                      alpha=0.8)
        if cax is not None:    
            plt.colorbar(img[3], ax=ax, label= r'density', cax=cax)

    if ytitle == True:
        ax.set_ylabel(r'Wsgsmax $[ms^{-1}]$', fontsize=12)
        #ax.set_ylabel(r'WG_cellAv / WG_domainAv $[ms^{-1}]$', fontsize=12)
    if xtitle == True:
        ax.set_xlabel(r'$\Delta T \, [K]$', fontsize=12)
    #-------------------------------------------------------------------------

    
    ## Set quantile lines
    x_arr  = bla_x[1][:bla_x[1].shape[0]-1] + delta_xbin/2
    img90=ax.plot(x_arr , wsgsmax_mean_stats_90.iloc[:,0], color=color, linestyle='--', label='p90')
    img75=ax.plot(x_arr , wsgsmax_mean_stats.iloc[:,2], color=color , linestyle='-.', label='p75')
    img50=ax.plot(x_arr , wsgsmax_mean_stats.iloc[:,1], color=color , linestyle='-' , label='p50')
    img25=ax.plot(x_arr , wsgsmax_mean_stats.iloc[:,0], color=color , linestyle='--', label='p25')

    #------------------------------
    # Set Plot layout
    ax.set_title(f'{x.name}')
    ax.set_ylim(y_bins.min(), y_bins.max())
    ax.set_xlim( x_bins.min() , x_bins.max())
    ax.invert_xaxis()

    # For plotting binned wsgsmax data
    #img=ax.plot(cell_mean_stats.iloc[:,0] ,  bla[1] , 'k--', label='p25')
    #img50=ax.plot(cell_mean_stats.iloc[:,1], bla[1] , 'k-' , label = 'p50')
    #img75=ax.plot(cell_mean_stats.iloc[:,2], bla[1] , 'k--' , label='p75')

    ## Set y extend
    y_range = kwargs.get('y_range')
    if y_range is not None:
        print('Set y_range')
        ax.set_ylim(20,y_range)

    plot_legend = kwargs.get('plot_legend')
    if plot_legend is not None:      
        legend = ax.legend(loc='upper left')
        frame = legend.get_frame()
        frame.set_color('lightsteelblue')



#%%

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_processing import sel_month, add_datetime64_column, read_df_cell

#%%
#----------
# Load Data
#----------
rootdir = '/net/pc200057/nobackup_1/users/frei/CPMs/'

df_cells = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells.csv')
df_cells_fut = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut.csv')

df_cells.name = 'Historic run (1995-2005)'
df_cells_fut.name = 'Future run (2089-2099)'

#%%

#---------------------------------- THRESHOLDS --------------------------------
threshold = 8
#------------------------------------------------------------------------------
df_cells.name = 'All cells (1995-2005)'
df_cells_fut.name = 'All cells (2089-2099)'

df_cells_fut_small = df_cells_fut[df_cells_fut['grd_clarea'] <= threshold]
df_cells_fut_small.name =  f'Future Large Cells (smaller, equal than {threshold} gridboxes)'


df_cells_big = df_cells[df_cells['grd_clarea'] > threshold]
df_cells_big.name =  f'Hist Large Cells (more than {threshold} gridboxes)'

df_cells_fut_big = df_cells_fut[df_cells_fut['grd_clarea'] > threshold]
df_cells_fut_big.name =  f'Future Large Cells (more than {threshold} gridboxes)'

#%%

##########################
# Plot for whole df_cell #
##########################

#------- Define metircs and y-variable ----------------------------------------
y = df_cells['peakVal']
y_fut = df_cells_fut['peakVal']
metrics = [ 'cell_mean_2h', 'cell_q25_2h', 'cell_q10_2h' ]


## Select Plot characterisitcs
v_max = 100
bin_width = 2
bin_height = 1
quantile_cutoff = 35

#----------------------------------------threshold=8--------------------------------------
fig = plt.subplots(figsize=(15,14))
gs = GridSpec(nrows=5, ncols=4, height_ratios=[10, 1, 10, 1, 10], width_ratios=[20, 20, 20, 0.7])

plt.suptitle(f'Tor row: {df_cells.name} = {df_cells.shape[0]}, \nMiddle row: {df_cells_fut.name} = {df_cells_fut.shape[0]}', fontsize=20)

cax1 = plt.subplot(gs[0,3])
cax2 = plt.subplot(gs[2,3])

ax=plt.subplot(gs[0,0])
s= plot_2dhist_quantiles(df_cells[metrics[0]],y, df=df_cells,
                         ytitle=True, 
                         v_max=v_max, 
                         quantile_cutoff = quantile_cutoff,
                         bin_width = bin_width,
                         plot_legend = True,
                         bin_height = bin_height)

ax=plt.subplot(gs[0,1])
s=plot_2dhist_quantiles(df_cells[metrics[1]],y,  df=df_cells,
                        xtitle=False, 
                        ytitle=False, 
                        v_max=v_max, 
                        quantile_cutoff = quantile_cutoff,
                        bin_width = bin_width,
                        plot_legend = True,
                        bin_height = bin_height)

ax=plt.subplot(gs[0,2])
s=plot_2dhist_quantiles(df_cells[metrics[2]],y,  df=df_cells,
                        v_max=v_max, 
                        quantile_cutoff = quantile_cutoff,
                        bin_width = bin_width,
                        plot_legend = True,
                        bin_height = bin_height)

ax=plt.subplot(gs[2,0])
s=plot_2dhist_quantiles(df_cells_fut[metrics[0]],y_fut, df=df_cells_fut,
                        cax=cax1, 
                        ytitle=True, 
                        v_max=v_max, 
                        quantile_cutoff = quantile_cutoff,
                        bin_width = bin_width,
                        plot_legend = True,
                        bin_height = bin_height)

ax=plt.subplot(gs[2,1])
s=plot_2dhist_quantiles(df_cells_fut[metrics[1]],y_fut,  df=df_cells_fut,
                        v_max=v_max, 
                        quantile_cutoff = quantile_cutoff,
                        bin_width = bin_width,
                        plot_legend = True,
                        bin_height = bin_height)

ax=plt.subplot(gs[2,2])
s=plot_2dhist_quantiles(df_cells_fut[metrics[2]],y_fut, df=df_cells_fut,
                        cax=cax2, 
                        xtitle=False, 
                        v_max=v_max, 
                        quantile_cutoff = quantile_cutoff,
                        bin_width = bin_width,
                        bin_height = bin_height,
                        plot_legend = True,
                        plot_dist = True
                        )

##-----------------------------------------------------##
## Plot the qunatile lins of hsit adn future in 1 plot ##
##-----------------------------------------------------##
ax=plt.subplot(gs[4,0])
s= plot_2dhist_quantiles(df_cells[metrics[0]], y, 
                         df=df_cells,
                         xtitle=True, ytitle=True, v_max=v_max, 
                         quantile_cutoff = quantile_cutoff,
                         bin_width = bin_width, bin_height = bin_height,
                         y_range = 37,
                         plot_legend = True,
                         plot_dist = False, color='k')

ax=plt.subplot(gs[4,0])
s= plot_2dhist_quantiles(df_cells_fut[metrics[0]], y_fut, 
                         df=df_cells_fut, 
                         ytitle=True, v_max=v_max, 
                         quantile_cutoff = quantile_cutoff,
                         bin_width = bin_width, bin_height = bin_height , 
                         y_range = 37,
                         plot_dist = False, color='g')


ax=plt.subplot(gs[4,1])
s=plot_2dhist_quantiles(df_cells[metrics[1]], y, 
                        df=df_cells, 
                        xtitle=True,  v_max=v_max,  
                        quantile_cutoff = quantile_cutoff, 
                        bin_width = bin_width, bin_height = bin_height, 
                        y_range = 37,
                        plot_legend = True,
                        plot_dist = False, color='k')
ax=plt.subplot(gs[4,1])
s=plot_2dhist_quantiles(df_cells_fut[metrics[1]], y_fut, 
                        df=df_cells_fut, 
                        xtitle=True,  v_max=v_max,  
                        quantile_cutoff = quantile_cutoff, 
                        bin_width = bin_width, bin_height = bin_height, 
                        y_range = 37,
                        plot_dist = False, color='g')


ax=plt.subplot(gs[4,2])
s=plot_2dhist_quantiles(df_cells[metrics[2]], y, 
                        df=df_cells, 
                        xtitle=True,  v_max=v_max,  
                        quantile_cutoff = quantile_cutoff, 
                        bin_width = bin_width, bin_height = bin_height, 
                        y_range = 37,
                        plot_legend = True,
                        plot_dist = False , color='k')

ax=plt.subplot(gs[4,2])
s=plot_2dhist_quantiles(df_cells_fut[metrics[2]], y_fut, 
                        df=df_cells_fut, 
                        xtitle=True,  v_max=v_max,  
                        quantile_cutoff = quantile_cutoff, 
                        bin_width = bin_width, bin_height = bin_height, 
                        y_range = 37,
                        plot_dist = False, color='g' )


# save_title = df.name.replace(' ', '_').replace('(','_______').replace(')','')
plt.savefig(f'/usr/people/frei/MaxFrei/Max_figures/general_analysis/GustFront_TempDrop/TASdrop_2h_fut_hist.png', dpi=450)



#%%

##########################################################################
# Plot PDF distribution of temperature drop for different sizes of cells #
##########################################################################

fig = plt.subplots( figsize=(15,10) )
gs = GridSpec(nrows=3, ncols=3, height_ratios=[20, 1, 20], width_ratios=[20,20,20] )


#--------------------------------------------------------Plot top left -------------------------------------------------------
ax = plt.subplot(gs[0,0])
ax.invert_xaxis()
ax.set_title(r'PDF of 10th pct $\Delta$TAS drop for all cells', fontsize=10)
ax.set_xlabel('Temperature drop (2h)', fontsize=10)
s= ax.hist(df_cells['cell_q10_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label=f'Hist ({df_cells.shape[0]} Cells)')

s= ax.hist(df_cells_fut['cell_q10_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label=f'Fut ({df_cells_fut.shape[0]} Cells)')
ax.legend(loc='upper right')

ax.set_ylim(0,0.3)
#-----------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------Plot Bottom left -------------------------------------------------------
ax=plt.subplot(gs[2,0])
ax.invert_xaxis()
ax.set_title(r'PDF of mean $\Delta$TAS drop for all cells', fontsize=10)
ax.set_xlabel('Temperature drop (2h)', fontsize=10)
s= ax.hist(df_cells['cell_mean_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label='Historic run')

s= ax.hist(df_cells_fut['cell_mean_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label='Future run')

ax.set_ylim(0,0.3)

#-----------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------Plot top Middle -------------------------------------------------------
ax=plt.subplot(gs[0,1])
ax.invert_xaxis()
ax.set_title(f'PDF of 10th pct $\Delta$TAS for cells <= {threshold} grd_area', fontsize=10)
ax.set_xlabel('Temperature drop (2h)', fontsize=10)
s= ax.hist(df_cells_small['cell_q10_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label=f'Hist ({df_cells_small.shape[0]} Cells)')

s= ax.hist(df_cells_fut_small['cell_q10_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label=f'Fut ({df_cells_fut_small.shape[0]} Cells)')
ax.legend(loc='upper right')

ax.set_ylim(0,0.3)

#-----------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------Plot Bottom Middle -------------------------------------------------------
ax=plt.subplot(gs[2,1])
ax.invert_xaxis()
ax.set_title(f'PDF of mean $\Delta$TAS for cells <= {threshold} grd_area', fontsize=10)
ax.set_xlabel('Temperature drop (2h)', fontsize=10)
s= ax.hist(df_cells_small['cell_q50_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label='Historic run')

s= ax.hist(df_cells_fut_small['cell_mean_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label='Future run')


ax.set_ylim(0,0.3)
#-----------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------Plot top Right -------------------------------------------------------
ax=plt.subplot(gs[0,2])
ax.invert_xaxis()
ax.set_title(f'PDF of 10th pct $\Delta$TAS for cells > {threshold} grd_area', fontsize=9)
ax.set_xlabel('Temperature drop (2h)', fontsize=10)
s= ax.hist(df_cells_big['cell_q10_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label=f'Hist ({df_cells_big.shape[0]} Cells)')

s= ax.hist(df_cells_fut_big['cell_q10_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label=f'Fut ({df_cells_fut_big.shape[0]} Cells)')
ax.legend(loc='upper left')

ax.set_ylim(0,0.3)
#-----------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------Plot Bottom Right -------------------------------------------------------
ax=plt.subplot(gs[2,2])
ax.invert_xaxis()
ax.set_title(f'PDF of mean $\Delta$TAS for cells > {threshold} grd_area', fontsize=10)
ax.set_xlabel('Temperature drop (2h)', fontsize=10)
s= ax.hist(df_cells_big['cell_mean_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label='Historic run')

s= ax.hist(df_cells_fut_big['cell_q50_2h'], histtype='step', bins=np.arange(-15,5.5,0.5), density=True, label='Future run')


ax.set_ylim(0,0.3)
#-----------------------------------------------------------------------------------------------------------------------------

plt.savefig(f'/usr/people/frei/MaxFrei/Max_figures/general_analysis/GustFront_TempDrop/TASdrop_distribution_different_cellsizes.png', dpi=250)







#%%
#----------------------------------------------
# Plot size distribution of cells, fut & hist
#----------------------------------------------
fig = plt.subplots( figsize=(12,6) )
gs = GridSpec(nrows=1, ncols=3, height_ratios=[1], width_ratios=[1,1,1] )


#--------------------------------------------------------Plot left -------------------------------------------------------
ax=plt.subplot(gs[0,0])
bins = 10**(np.arange(0,3,0.01))
ax.set_xscale('log')
ax.set_title('Cell size distribution',fontsize=10)
ax.set_xlabel('Cell Area in gridpoints',fontsize=10)
ax.set_xscale('log')
s = ax.hist(df_cells['grd_clarea'], cumulative=True, density=True, histtype='step', bins=bins, log=False, label=f'Hist ({df_cells.shape[0]} Cells)')

s = ax.hist(df_cells_fut['grd_clarea'], cumulative=True, density=True, histtype='step', bins=bins, log=False, label=f'Fut ({df_cells_fut.shape[0]} Cells)')
ax.legend(loc='lower right')

# textstr = ''.join((
#     'Historic Cells: {}'.format(df_cells.shape[0]),
#     '\nFuture Cells: {}'.format(df_cells_fut.shape[0])
#     ))
# props = dict(boxstyle='round', facecolor='wheat',alpha=0.5)
# ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
#-----------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------Plot middle -------------------------------------------------------
ax=plt.subplot(gs[0,1])
bins = 10**(np.arange(0,3,0.01))
ax.set_xscale('log')
ax.set_title(f'PDF of 10th pct $\Delta$TAS for cells <= {threshold} grd_area',fontsize=10)
ax.set_xlabel('Cell Area in gridpoints',fontsize=10)
ax.set_xscale('log')
s = ax.hist(df_cells_small['grd_clarea'], cumulative=True, density=True, histtype='step', bins=bins, log=False, label=f'Hist ({df_cells_big.shape[0]} Cells)')

s = ax.hist(df_cells_fut_small['grd_clarea'], cumulative=True, density=True, histtype='step', bins=bins, log=False, label=f'Fut ({df_cells_fut_big.shape[0]} Cells)')

ax.legend(loc='lower right')
#-----------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------Plot right -------------------------------------------------------
ax=plt.subplot(gs[0,2])
bins = 10**(np.arange(0,3,0.01))
ax.set_xscale('log')
ax.set_title(f'PDF of 10th pct $\Delta$TAS for cells > {threshold} grd_area',fontsize=10)
ax.set_xlabel('Cell Area in gridpoints',fontsize=10)
ax.set_xscale('log')
s = ax.hist(df_cells_big['grd_clarea'], cumulative=True, density=True, histtype='step', bins=bins, log=False, label=f'Hist ({df_cells_big.shape[0]} Cells)')

s = ax.hist(df_cells_fut_big['grd_clarea'], cumulative=True, density=True, histtype='step', bins=bins, log=False, label=f'Fut ({df_cells_fut_big.shape[0]} Cells)')

ax.legend(loc='upper left')
#-----------------------------------------------------------------------------------------------------------------------------
plt.savefig(f'/usr/people/frei/MaxFrei/Max_figures/general_analysis/GustFront_TempDrop/cells_size_distribution', dpi=150)



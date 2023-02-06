#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:01:22 2022

@author: frei
"""


#%%

# LIBS
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import os
import imageio

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


from data_processing import sel_month, add_datetime64_column, sel_month

#%%
#----------
# Load Data
#----------
rootdir = '/net/pc200057/nobackup_1/users/frei/CPMs/'


# Dataframe with cells (excluding ocean and boundary touching)
df_cells = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cell.csv')
df_cells = add_datetime64_column(df_cells)

# Dataframe with cells (excluding ocean and boundary touching)
df_cells = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells.csv')
df_cells.name = 'Historic cells (1995-2005)'
df_cells = add_datetime64_column(df_cells)

df_cells_fut = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut.csv')
df_cells_fut.name = 'Future cells (1995-2005)'
df_cells_fut = add_datetime64_column(df_cells_fut)

#%%
def plot_cellfield_tas_precip_div(df,
                                  x_var,
                                  y_var,
                                  x_value,
                                  y_value,
                                  x_sel_width = 1/4,
                                  y_sel_width = 1,
                                  # contourf = ds_tas,
                                  pathfig = 'interactive_cell_field_plots',
                                  **kwargs
                                  ):
    """

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    x_var : TYPE
        DESCRIPTION.
    y_var : TYPE
        DESCRIPTION.
    x_value : TYPE
        DESCRIPTION.
    y_value : TYPE
        DESCRIPTION.
    x_sel_width : TYPE, optional
        DESCRIPTION. The default is 1/4.
    y_sel_width : TYPE, optional
        DESCRIPTION. The default is 1.
    # contourf : TYPE, optional
        DESCRIPTION. The default is ds_tas.
    pathfig : TYPE, optional
        DESCRIPTION. The default is 'interactive_cell_field_plots'.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    #---------------------------------
    # Define where figs will be stored
    #---------------------------------
    basic_path= '/usr/people/frei/MaxFrei/Max_figures/case_studies/'
    
    #-----------------------------------------
    # Select cases based on Area and PeakValue
    #-----------------------------------------
    peak_val = y_value
    # Define Tolerances around wg and area
    area_tol = x_sel_width * x_value
    #area_tolerance = kwargs.get('tolerance_wg')
    df = df
  
        
    if 'Historic' in df.name:
        #----------------
        ## Historic Data
        #----------------

        rootdir = '/net/pc200057/nobackup_1/users/frei/CPMs/'

        ## Load cell files
        ds_cells =  xr.open_dataset(f'/net/pc200057/nobackup_1/users/frei/track_cluster/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist/allyears_merged/cells.nc').drop_vars('lev')
        ## Windgusts dataset
        ds_wsgsmax = xr.open_dataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_wsgsmax_merged_lonlatbox/wsgsmax.his.CXFPS025.HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist.1hr_lonlatbox_merged.nc')
        ds_wsgsmax = sel_month(ds_wsgsmax)
        ## Load vertical and horizontal wind fields
        ds_ugs = xr.open_mfdataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_ugs_lonlatbox/*.nc')
        ds_ugs = sel_month(ds_ugs)
        ds_vgs = xr.open_mfdataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_vgs_lonlatbox/*.nc')
        ds_vgs = sel_month(ds_vgs)    
        ## Load temperature files
        ds_tas = xr.open_mfdataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_tas_lonlatbox/*.nc')
        ds_tas = sel_month(ds_tas)
        ## Load Precipitation files
        ds_pr = xr.open_mfdataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_pr_lonlatbox/*.nc')
        ds_pr = sel_month(ds_pr)

    if 'Future' in df.name:
        #----------------
        ## Future Data ##
        #----------------        
        ds_cells =  xr.open_dataset(f'/net/pc200057/nobackup_1/users/frei/CPM_future/track_cluster_fut/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r04_rcp85/cells_merged.nc').drop_vars('lev')
        data_dir = '/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/fields_lonlatbox_selmonth'
        ## Windgusts dataset
        ds_wsgsmax = xr.open_mfdataset(f'{data_dir}/wsgsmax_lonlatbox_selmonth/*.nc')
        ## Load vertical and horizontal wind fields
        ds_ugs = xr.open_mfdataset(f'{data_dir}/ugs_lonlatbox_selmonth/*.nc')
        #ds_ugs = sel_month(ds_ugs)
        ds_vgs = xr.open_mfdataset(f'{data_dir}/vgs_lonlatbox_selmonth/*.nc')
        ## Load temperature files
        ds_tas = xr.open_mfdataset(f'{data_dir}/tas_lonlatbox_selmonth/*.nc')
        ## Load Precipitation files
        ds_pr = xr.open_mfdataset(f'{data_dir}/pr_lonlatbox_selmonth/*.nc')
    
    
    
    
  
    #---------------------------------------------------------------
    # ScatterPlot: Area vs PeakVal + Rectangle around selected cases
    #---------------------------------------------------------------
    fig,ax = plt.subplots()
    img=ax.scatter(df[x_var], 
                   df[y_var], 
                   s=5,
                   edgecolors = 'k', 
                   alpha=0.8 )
    ax.set_xlabel(df[x_var].name)
    ax.set_ylabel(df[y_var].name)
    # ax.set_title('WG-cells PeakVal')
    if x_var == 'grd_clarea':
        ax.set_xscale('log')
    
    ax.set_ylim(np.floor(df[y_var].min()/10)*10 , np.ceil(df[y_var].max()/10)*10)
    ## Add rectangl around selected cases
    ax.add_patch(Rectangle(
        xy=( x_value - x_sel_width , y_value - y_sel_width) , 
        width=x_sel_width*2, height=y_sel_width*2,
        linewidth=1, color='red', fill=False))
    
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # PATH SELECTION for saving figure
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    if 'Historic' in df.name:    
        Selection = f'{df[x_var].name}_{int(x_value)}_{df[y_var].name}_{y_value}'
        path_fig = basic_path + pathfig + '/historic/' + Selection
        print(f'\nPath of scatterplot:\t\t{path_fig}\n')
        # Create outpath if it doesn't exist
        if not os.path.exists(path_fig):
            os.makedirs(path_fig)
        plt.savefig(f'{path_fig}/scatter_{Selection}.png' , dpi=500)    


    if 'Future' in df.name:    
        Selection = f'{df[x_var].name}_{int(x_value)}_{df[y_var].name}_{y_value}'
        path_fig = basic_path + pathfig + '/future/' + Selection
        print(f'\nPath of scatterplot:\t\t{path_fig}\n')
        # Create outpath if it doesn't exist
        if not os.path.exists(path_fig):
            os.makedirs(path_fig)
        plt.savefig(f'{path_fig}/scatter_{Selection}.png' , dpi=500)    

    #-------------------------------------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------------------------------------    
    # Select the cells that fulfill the condtions 
    mask = (df[x_var]>x_value - x_sel_width) & (df[x_var] < (x_value + x_sel_width)) & (df[y_var]> (y_value - y_sel_width)) & (df[y_var] < (y_value + y_sel_width))
    df_case_sel = df[mask]

    #------- Drop duplicates in df --------#
    df_case_sel.drop_duplicates(inplace=True)
    df_case_sel.reset_index(inplace=True, drop=True)
    #-------------------------------------------#

    
    ##################################################################################################################################################
    #================================================================================================================================================#
    ########                                                          MAIN LOOP                                                               ########
    #================================================================================================================================================#
    ##################################################################################################################################################

    
    #=================================================================================================================================================
    # Loop through all selected cases (in Rectangle)
    #=================================================================================================================================================
    # Count the number of different cases in rectangle selction
    case_counter = 0    
    for grd_clarea, peak_val in zip(df_case_sel.loc[0:5,'grd_clarea'] , df_case_sel.loc[0:5 , 'peakVal']):
        
        # Select the indivdual cases from df_case_sel
        df_case = df[ (df['grd_clarea']==grd_clarea) & (df['peakVal']==peak_val) ] 
        # reset index of dataframe
        df_case.reset_index(inplace=True, drop=True)
        
        # Ceck if a case occured in selected values, otherwise return error
        assert df_case.shape[0] != 0 , 'No case occured close to this selection of WG and Area, select different values!!'
    
        #====================================================== LOOP Through Timesteps before event =====================================================
        backwards = 4
        timesteps = []
        for i in range(backwards, 0, -1):
            timesteps.append(df_case['datetime'].iloc[0] - np.timedelta64( i , 'h'))
        forwards = 2
        for i in range(forwards+1):
            timesteps.append(df_case['datetime'].iloc[0] + np.timedelta64( i , 'h'))
        # Get total number of timestpes
        total_dt = len(timesteps)
        #=================================================================================================================================================

        
        """ 
        Loop trough all the timesteps around the event of interest and plot the fields
            - it is undetermined if the cell contours will be plotted and if they will be the respective cells at that particular time
            - it would be nice to have the cell of interest as a different color and show how this cell evolves
        """
        
        imagename=[]
        # fig,ax = plt.subplots()
        fig = plt.subplots(figsize=(10,10))
              
        # Set count_variables for titles, paths and progress messages
        img_nr, dt_counter = 0, 0
        case_counter += 1
        #------------------------------------------------------------ LOOP THROUGH TIMESTEPS BEFORE (AND AFTER) EVENT ---------------------------------------
        for dt in timesteps:     
            # Get upper and lower timestep for hourly output fields
            time_ceil =  dt + np.timedelta64(30,'m')
            time_floor = dt - np.timedelta64(30,'m')
            #------------------------------------------------------------
            
            
            
            if str(df_case.loc[0,'datetime'])[0:4] != '1995':
     
                ###############################################################################################################
                #########################                DATA SELECTION AND PROCESSING                #########################
                ###############################################################################################################
                #---------------------------------------------------------------------------
                ### TAS DATA
                # Floor hour
                tas_array_before = ds_tas.sel(time=time_floor).tas.data
                title_time_before = str(time_floor)[0:16]
                title_before = f'TAS-field at {title_time_before} (before)'
                # Ceiling hour
                tas_array_after = ds_tas.sel(time=time_ceil).tas.data
                tas_array_celsius = tas_array_after - 273
                title_time_after = str(time_ceil)[0:16]
                title_after = f'TAS-field at {title_time_after} (after)'
        
                #da_floor = ds_tas.sel(time=df_case['datetime_floor'][0]).tas
                #da_ceil = ds_tas.sel(time=df_case['datetime_ceil'][0]).tas
                #da = da_ceil - da_floor
                tas_array_diff = tas_array_after - tas_array_before
                title_time = str(dt)[0:16]
                title_diff = f'Difference of TAS around {title_time}'
                #------------------------------------------------------------------------------
        
                #---------------------------------------------------------------------------
                ### PRECIP DATA
                pr_array_after = ds_pr.sel(time=dt).pr.data
                #------------------------------------------------------------------------------
        
                #---------------------------------------------------------------------------
                ### DIVERGENCE DATA
                wsgsmax_array = ds_wsgsmax.sel(time=dt).wsgsmax.data
                div_array = np.gradient(wsgsmax_array)[0] + np.gradient(wsgsmax_array)[1]
                #------------------------------------------------------------------------------
        
                #------------------------------------------------------------------------------
                ## U and V components of wind gust for drawing vectors
                ugs_array = ds_ugs.sel(time=dt).ugs.data
                vgs_array = ds_vgs.sel(time=dt).vgs.data
                skip = (slice(None, None, 10), slice(None, None, 10))    
                #------------------------------------------------------------------------------
                
                #------------------------------------------------------------------------------
                # Plot all cell at this hour as contour lines
                #--------------------------------------------
                ## Cell-fields for contour plots
                ds_cell_case = ds_cells.sel(time=dt)
                cell_contour_array = (ds_cell_case.cellID[0,:,:] == df_case['clID'][0] ).data 
                  
                allcells_array = (ds_cell_case.cellID[0,:,:].isin(ds_cell_case.cellID[0,:,:]) ).data        
                ###############################################################################################################
                ###############################################################################################################
                
                ## SELECT PLOTWINDOW
                da = ds_tas.sel(time=df_case['datetime_floor'][0]).tas
                df_select_domain = da.to_dataframe().dropna()
                lonmin = df_select_domain.lon.min() - (df_select_domain.lon.max() - df_select_domain.lon.min()) * 1/40
                lonmax = df_select_domain.lon.max() + (df_select_domain.lon.max() - df_select_domain.lon.min()) * 1/40          
                latmin = df_select_domain.lat.min() - (df_select_domain.lat.max() - df_select_domain.lat.min()) * 1/40           
                latmax = df_select_domain.lat.max() + (df_select_domain.lat.max() - df_select_domain.lat.min()) * 1/40            
        
                ## DEFINE MESHGRID
                X,Y = ds_tas.lon, ds_tas.lat
                
                ############################################################################################################################################
                # START PLOTTING
                ############################################################################################################################################
                ### Open Plot ###
                fig = plt.subplots(figsize=(18,6))
        
                # Define grid for plots
                # gs  = GridSpec(nrows=6, ncols=2, height_ratios=[20, -2.4, 1.2, 20, -2.4, 1.2])  # arange plot windows
                gs = GridSpec(nrows=3,ncols=3,  height_ratios=[1, -0.11, 0.05], width_ratios=[1,1,1])
                
                
                # Define coloar abr axes
                cax1 = plt.subplot(gs[2,0])
                # cax2 = plt.subplot(gs[5,0])
                # cax3 = plt.subplot(gs[2,1])
                # cax4 = plt.subplot(gs[5,1])    
                
                cax2 = plt.subplot(gs[2,1])
                cax4 = plt.subplot(gs[2,2])
                
                ## General layout things (title and space between figures)
                plt.suptitle(f"Time to gust-event t{-backwards + dt_counter}", fontsize=25)
                plt.subplots_adjust(top=0.95)

                ############################################# UPPER LEFT (TAS) #########################################################################
                ax1=plt.subplot(gs[0,0], projection=ccrs.PlateCarree())
        
                cmap = plt.cm.nipy_spectral     # mpl.cm.viridis    # plt.cm.RdYlBu   
                levs = np.round( np.arange( 0, 41, 2 ))                     #np.round(delta_levs/20,1)))
                norm = mpl.colors.BoundaryNorm(levs,cmap.N)
        
                # map
                ax1.coastlines(linewidth=1)
                ax1.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
                # set the ticks
                ax1.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
                ax1.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
                # format the ticks as e.g 60302260Wtemp_ceil_array
                ax1.xaxis.set_major_formatter(LongitudeFormatter())
                ax1.yaxis.set_major_formatter(LatitudeFormatter())   
                # Title
                ax1.set_title(title_after)
                # Plot outlines of cells
                E=ax1.contour(X,Y , allcells_array , linewidths=0.8, linestyles ='--', colors='beige', zorder=3, alpha=0.8 ,transform=ccrs.PlateCarree() )
                D=ax1.contour(X,Y ,  cell_contour_array ,  linewidths=1, linestyles ='-', colors='white', zorder=4 , transform=ccrs.PlateCarree() )
                # Plot Temperature field
                img1=ax1.contourf(X,Y, tas_array_celsius, cmap=cmap , levels=levs , norm=norm, extend='both'  , transform=ccrs.PlateCarree() )
                plt.colorbar(img1,label=r'$TAS_{2m} \, [^{\circ}C]$',ax=ax1,orientation="horizontal" ,cax=cax1)      ### Took out formatter: ,format='%3i'
                # Plot Wind vectors
                vect=ax1.quiver(X[skip],Y[skip] ,  ugs_array[skip] ,vgs_array[skip], zorder=2  , alpha=0.8)         #, scale=1e3,   headwidth=4, headlength=5)
                ax1.set_extent([lonmin,lonmax,latmin,latmax] , ccrs.PlateCarree() )
        
        
                #################################################################### LOWER LEFT (TAS DIFFERENCE) ################################################################################
                # ax2=plt.subplot(gs[3,0], projection=ccrs.PlateCarree())
                ax2=plt.subplot(gs[0,1], projection=ccrs.PlateCarree())
                
                cmap = plt.cm.coolwarm  #bwr
                levs = np.round( np.arange( -8, 9, 1 ))                     #np.round(delta_levs/20,1)))
                norm = mpl.colors.BoundaryNorm(levs,cmap.N)
                #map
                ax2.coastlines(linewidth=1)
                ax2.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
                # set the ticks
                ax2.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
                ax2.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
                # format the ticks as e.g 60302260Wtemp_ceil_array
                ax2.xaxis.set_major_formatter(LongitudeFormatter())
                ax2.yaxis.set_major_formatter(LatitudeFormatter())      
                # Title
                ax2.set_title(title_diff)
        
                E=ax2.contour(      X,Y , allcells_array , linewidths=0.8, linestyles ='--', colors='beige', zorder=3, alpha=0.8 ,transform=ccrs.PlateCarree() )
                D=ax2.contour(      X,Y ,  cell_contour_array ,  linewidths=1, linestyles ='-', colors='white', zorder=4 , transform=ccrs.PlateCarree() )
                img2=ax2.contourf(  X,Y, tas_array_diff, cmap=cmap, levels=levs ,  norm=norm, extend='both', zorder=1 , transform=ccrs.PlateCarree() )
                plt.colorbar(img2,label=r'$\Delta \, TAS_{2m} \, [^{\circ}C]$',ax=ax2,orientation="horizontal" , cax=cax2)      ### Took out formatter: ,format='%3i'
                vect=ax2.quiver(    X[skip],Y[skip] ,  ugs_array[skip] ,vgs_array[skip], zorder=2 , alpha=0.8 )         #, scale=1e3,   headwidth=4, headlength=5)
                ax2.set_extent([lonmin,lonmax,latmin,latmax] , ccrs.PlateCarree() )
        
                ################################################################## UPPER RIGHT (PRECIPITATION ) ################################################################################
                # ax3=plt.subplot(gs[0,1], projection=ccrs.PlateCarree())
                # cmap = plt.cm.YlGnBu                                        #plt.cm.viridis
                # #levs = np.round( np.arange( 0, 0.035, 0.005 ), 3)                     #np.round(delta_levs/20,1)))
                # levs = np.round( np.arange( 0, 10, 1 ) )                     #np.round(delta_levs/20,1)))        
                # norm = mpl.colors.BoundaryNorm(levs,cmap.N)
                # #map
                # ax3.coastlines(linewidth=1)
                # ax3.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
                # # set the ticks
                # ax3.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
                # ax3.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
                # # format the ticks as e.g 60302260Wtemp_ceil_array
                # ax3.xaxis.set_major_formatter(LongitudeFormatter())
                # ax3.yaxis.set_major_formatter(LatitudeFormatter())      
                # # Title
                # ax3.set_title(f'Precipitation {title_time}')
        
                # E=ax3.contour(      X,Y , allcells_array , linewidths=0.8, linestyles ='--', colors='dimgray', zorder=3, alpha=0.99 ,transform=ccrs.PlateCarree() )
                # D=ax3.contour(      X,Y ,  cell_contour_array ,  linewidths=1, linestyles ='-', colors='white', zorder=4 , transform=ccrs.PlateCarree() )
                # img3=ax3.contourf(  X,Y, pr_array_after*1000, cmap=cmap,levels=levs ,  norm=norm, extend='both', zorder=1 , transform=ccrs.PlateCarree() )
                # plt.colorbar(img3,label=r'Precipitation [$g\,m^2\,s^{-1}$]',ax=ax3,orientation="horizontal" , cax=cax3)      ### Took out formatter: ,format='%3i'
                # vect=ax3.quiver(    X[skip],Y[skip] ,  ugs_array[skip] ,vgs_array[skip], zorder=2 , alpha=0.8 )         #, scale=1e3,   headwidth=4, headlength=5)
        
                # ax3.set_extent([lonmin,lonmax,latmin,latmax] , ccrs.PlateCarree() )
                
                ################################################################## LOWER RIGHT (DIVERGENCE PLOT) ################################################################################
                # ax4=plt.subplot(gs[3,1], projection=ccrs.PlateCarree())
                ax4=plt.subplot(gs[0,2], projection=ccrs.PlateCarree())
                
                cmap = plt.cm.RdBu
                levs = np.round( np.arange(-4, 5, 0.5), 1 )                     #np.round(delta_levs/20,1)))
                norm = mpl.colors.BoundaryNorm(levs,cmap.N)
                #map
                ax4.coastlines(linewidth=1)
                ax4.gridlines(ylocs=np.arange(-90, 91, 5), xlocs=np.arange(-180, 181, 5))
                # set the ticks
                ax4.set_xticks(np.arange(-180, 181, 5), crs=ccrs.PlateCarree());
                ax4.set_yticks(np.arange(-90, 91, 5), crs=ccrs.PlateCarree());
                # format the ticks as e.g 60302260Wtemp_ceil_array
                ax4.xaxis.set_major_formatter(LongitudeFormatter())
                ax4.yaxis.set_major_formatter(LatitudeFormatter())      
                # Title
                ax4.set_title(f'Divergence field at {title_time}')
        
                E=ax4.contour(      X,Y ,    allcells_array , linewidths=0.8 , linestyles ='--' , colors='dimgray', zorder=3, alpha=0.99 ,transform=ccrs.PlateCarree() )
                D=ax4.contour(      X,Y ,cell_contour_array ,  linewidths=1  , linestyles ='-'  , colors='white', zorder=4 ,           transform=ccrs.PlateCarree() )
                img4=ax4.contourf(  X,Y, div_array , cmap=cmap , levels=levs ,  norm=norm , extend='both' , zorder=1 , transform=ccrs.PlateCarree() )
                plt.colorbar(img4,label='Divergence',ax=ax4,orientation="horizontal" , cax=cax4)      ### Took out formatter: ,format='%3i'
                vect=ax4.quiver(    X[skip],Y[skip] ,  ugs_array[skip] ,vgs_array[skip], zorder=2 , alpha=0.8 )         #, scale=1e3,   headwidth=4, headlength=5)
        
                ax4.set_extent([lonmin,lonmax,latmin,latmax] , ccrs.PlateCarree() )
                
                ############################################################################################################################################################################
                #------------------------------------------------------ Progress metrics -----------------------------------------------------------
                ## Set title with relative timestep to mainevent
                img_nr+=1
                dt_counter +=1
                print(f'\n Plotting case: \t {case_counter} of {df_case_sel.shape[0]} \t\t Done with timestep: \t {dt_counter} of {total_dt} \n')
                #------------------------------------------------------------------------------------------------------------------------------------
    
                #------------------------------------------------------ Create new folder for each selected case --------------------------------------------------
                EventDate = str(df_case['datetime'])[5:21]        
                subpath_fig = path_fig + '/' + EventDate
                
                print(f'\nEventDate:\t{EventDate}\nPath of figures:\t\t{subpath_fig}\n')
                # Create outpath if it doesn't exist
                if not os.path.exists(subpath_fig):
                    os.makedirs(subpath_fig)
                #--------------------------------------------------------------------------------------------------------------------------------------------------

                #-------------------------------------------------------------------------------------------------------------------------------------------------
                # SAVE FIGURE
                #-------------------------------------------------------------------------------------------------------------------------------------------------
                plt.savefig(     f'{subpath_fig}/contourf_area{int(grd_clarea)}_peakVal{int(peak_val)}_dt{dt_counter}.png' , dpi=500)
                imagename.append(f'{subpath_fig}/contourf_area{int(grd_clarea)}_peakVal{int(peak_val)}_dt{dt_counter}.png' )
            
            
            EventDate = str(df_case['datetime'])[5:21]   
            subpath_fig = path_fig + '/' + EventDate
            if not os.path.exists(subpath_fig):
                    os.makedirs(subpath_fig)
            with imageio.get_writer(f'{subpath_fig}/contourf{int(grd_clarea)}_peakVal{int(peak_val)}.gif', mode='I',duration=1.5) as writer:
                for filename in imagename:
                    image = imageio.imread(filename)
                    writer.append_data(image)


#%%

# if __name__ == '__main__':    
#     wsgsmax       = 35
#     grid_cellarea = 1000
    
#     plot_cellfield_tas_precip_div(wsgsmax,grid_cellarea, 
#                               dfs = dfs, 
#                               contourf=ds_tas, 
#                               contour1=ds_wsgsmax, 
#                               area_tolerance = 1/4, 
#                               wg_tolerance = 1,
#                               pathfig='tas_precip_div')


#     plot_cellfield_tas_precip_div(
#         df = df_cells_fut,
#         x_var = 'cell_q10_2h',
#         y_var = 'peakVal',
#         x_value = -16,
#         y_value = 40,
#         x_sel_width=1,
#         y_sel_width=1)

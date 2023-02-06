#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:19:12 2022

@author: frei


This script calculates statistics of the temperature drop that occurs within the cells. 


Algoritmn to compute TAS-drop stats (used in function: get_tas_stats2h):
    
1) Get a mask field for the each cellID (clID in df_cell)
    
2) Apply mask field to deltaTAS filed to get the temperature difference within the cell

3) Calculate statistical values (max, min, 90pctl, median, mean) within the cells deltaTAS-field
    
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl

from data_processing import sel_month, add_datetime64_column, read_df_cell

from joblib import Parallel, delayed
from tqdm import tqdm


#%%
def remove_cells_end_of_time(df):
    """
    Parameters
    ----------
    df : pandas.data.frame
        Input either df_cells or df_cells_fut.

    Returns
    -------
    df : pandas.data.frame
        Same df_cells/df_cells_fut as inputed, but removed timestep at the end of September (i.e. 30-09 22:30) 
        
    TAS-field that is taken at that time is:
        ceiling time of wsgsmax cell -> 23:00
        + 1 hours                    -> 00:00 OCTOBER 1st

    """
    error_list=[]
    df_cell_old_length = df.shape[0]
    print(f'\nLenght of df_cells before cleaning: \t\t{df_cell_old_length}\n')
    for i in range(df_cell_old_length):
        if (df.loc[i,'datetime_ceil']+np.timedelta64(1,'h')).month == 10:
            error_list.append(i)
            df.drop(i, inplace=True)

    print(f'Number of cells occuring in september 30th at 22:30: \t{len(error_list)} \t(dropped these values, since they cause bugs)')
    df_cells.reset_index(inplace=True)
    
    ## Assert that the original elnght of the dataset (64871) minus the critical time values equals the new dataset   
    assert df_cell_old_length - len(error_list) == df_cells.shape[0]                                                  
    
    return df
#%%


def get_tas_stats2h(i):

    #-----------------------------------------------------------------------------------------------------------------------
    # 1) Get the cell-mask for a corresponding Cell-ID
    da_cell_mask = ds_cells.sel(time=df_cells.loc[i, 'datetime'])['cellID'] == df_cells.loc[i, 'clID']
    da_cell_mask = da_cell_mask.where(da_cell_mask==True)    
    
    da_dtas_2h = ds_tas.sel( time=df_cells.loc[i, 'datetime_ceil'] + np.timedelta64(1,'h') ).tas - ds_tas.sel(time=df_cells.loc[i, 'datetime_floor']).tas 
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    # 2) Get TAS-difference within the contour with the corresponding cell-ID
    da_tas_cell_2h = da_dtas_2h.where(da_cell_mask==1)
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    # 3) Compute statistics of the TAS-drop of the cell
    mean_2h = np.nanmean(da_tas_cell_2h[:,:,0].values)
    q90_2h  = np.nanquantile(da_tas_cell_2h[:,:,0].values, 0.9)
    q75_2h  = np.nanquantile(da_tas_cell_2h[:,:,0].values, 0.75)
    q50_2h  = np.nanquantile(da_tas_cell_2h[:,:,0].values, 0.50)
    q25_2h  = np.nanquantile(da_tas_cell_2h[:,:,0].values, 0.25)
    q10_2h  = np.nanquantile(da_tas_cell_2h[:,:,0].values, 0.10)
    #-----------------------------------------------------------------------------------------------------------------------
   
    return mean_2h, q90_2h, q75_2h, q50_2h, q25_2h, q10_2h

#%%

if __name__ == '__main__':
    
    #====================================================================================================================================================================================#
    #------                Input Parameters                             ---------#
    #----------------------------------------------------------------------------#
    # Which df_cells is used (after cleaning (order of 10k cells) or after pre-processing (order of 100k cells))?
    # version_df_cells     = 'df_cells'                # Preprocessed AND filtered
    version_df_cells     = 'df_cells_preprocessed'     # ONLY preprocessed (larger, since synoptic events still contained)
    #=====================================================================================================================================================================================#


   
    #-----------------------------------------------------------------------------------------------------
    # Load hist data
    print('\n================================================================================================================')
    print('\t\t\t Computation Historic Temperature-drop stats')
    print('================================================================================================================')
    # Read in DataSets (cell-outline, wsgsmax-field, TAS-field)
    rootdir = '/net/pc200057/nobackup_1/users/frei/CPMs/'
    ds_cells =  xr.open_dataset('/net/pc200057/nobackup_1/users/frei/track_cluster/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist/allyears_merged/cells.nc').drop_vars('lev')
    ds_tas = xr.open_mfdataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_tas_lonlatbox/*.nc')
    ds_tas = sel_month(ds_tas)
    ds_wsgsmax = xr.open_dataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_wsgsmax_merged_lonlatbox/wsgsmax.his.CXFPS025.HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist.1hr_lonlatbox_merged.nc')
    # Select months MJJAS (note in future wsgsmax sets, these months are already selected)
    ds_wsgsmax = sel_month(ds_wsgsmax)
    
        
    path_df_cells = f'/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/{version_df_cells}.csv'
    # Read cell data with own function read_df_cell (-> this fct adds the datetime floor and ceiling column and converts them into correct dtype)
    df_cells = read_df_cell(path_df_cells)
    #-----------------------------------------------------------------------------------------------------


    ###################### Perfrom cell cleaning ################################################################
    # Some cells occur at sep 30th at 22:30, these cells yield an error when computing the 2h TAS difference    #
    # because their upper TAS field is at OCTOBER 1st.                                                          #
    # --> All cells that occur at that time most be dropped            (done by fct: remove_cells_end_of_time)  #
    #                                                                                                           #
    df_cells = remove_cells_end_of_time(df_cells)                                                               #
    #############################################################################################################
    
    
    
    ########################################################################################
    #### Compute temp drop in parallel #######
    ## Start parallel region
    ## ---------------------
    df_iter = df_cells.shape[0]
    print('Size dataset: ',df_iter)
    
    # results = Parallel(n_jobs=-1)(delayed(get_tas_stats2h)(i) for i in tqdm(range(df_iter)))
    results = Parallel(n_jobs=-1)(delayed(get_tas_stats2h)(i) for i in tqdm(range(df_iter)))
    ########################################################################################
    results = np.array(results, dtype=float)
    
    print('\nHistoric cells:')
    print('----------------------------------------------------------------------------------')
    print(f'Cell_mean:\n{results[:,0]}\n\nQ90:\n{results[:,1]}\n')
    print(f'q75: \n{results[:,2]}\n\nQ50:\n{results[:,3]}\n')
    print(f'q25: \n{results[:,4]}\n\nQ10:\n{results[:,5]}')
    print('----------------------------------------------------------------------------------\n\n\n')

    print(f'SHAPE OF Historic-temperature stats:\t\t{results.shape}')
    #-----------------------------------------------------------#
    ## Convert arrays to pandas dataframes ##
    df_cells_tempstats = pd.DataFrame(results, columns = ['cell_mean_2h','cell_q90_2h','cell_q75_2h','cell_q50_2h','cell_q25_2h','cell_q10_2h'])

    #-----------------------------------------------------------#
    ## Save the temperature statistics-dataframes as csv files ##  
    if 'preprocess' in version_df_cells:
        print('Save output file as df_cells_preprocessed_tempstats.csv')
        df_cells_tempstats.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/cell_temp_drop_stats/df_cells_preprocessed_tempstats.csv', index=False)
    else:
        print('Save output file as df_cells_tempstats.csv')
        df_cells_tempstats.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/cell_temp_drop_stats/df_cells_tempstats.csv', index=False)


    print('\n================================================================================================================')
    print('\t\t\t Computation Future Temperature-drop stats')
    print('================================================================================================================')

    # #-----------------------------------------------------------------------------------------------------
    # Load future data   
    # Read in DataSets (cell-outline, wsgsmax-field, TAS-field)
    ds_cells = xr.open_dataset('/net/pc200057/nobackup_1/users/frei/CPM_future/track_cluster_fut/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r04_rcp85/cells_merged.nc').drop_vars('lev')
    ds_wsgsmax = xr.open_mfdataset('/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/fields_lonlatbox_selmonth/wsgsmax_lonlatbox_selmonth/wsgsmax.*.nc')
    ds_tas = xr.open_mfdataset('/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/fields_lonlatbox_selmonth/tas_lonlatbox_selmonth/tas.*.nc')
    
    path_df_cells_fut = f'/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/{version_df_cells}_fut.csv'
    
    # Read cell data with own function read_df_cell (-> this fct adds the datetime floor and ceiling column and converts them into correct dtype)
    df_cells = read_df_cell(path_df_cells_fut)
    #-----------------------------------------------------------------------------------------------------

    ###################### Perfrom cell cleaning ################################################################
    # Some cells occur at sep 30th at 22:30, these cells yield an error when computing the 2h TAS difference    #
    # because their upper TAS field is at OCTOBER 1st.                                                          #
    # --> All cells that occur at that time most be dropped            (done by fct: remove_cells_end_of_time)  #
    #                                                                                                           #
    df_cells = remove_cells_end_of_time(df_cells)                                                               #
    #############################################################################################################


    ########################################################################################
    #### Compute temp drop in parallel #######
    ## Start parallel region
    ## ---------------------
    df_iter = df_cells.shape[0]
    print('Size dataset: ',df_iter)
    results_fut = Parallel(n_jobs=-1)(delayed(get_tas_stats2h)(i) for i in tqdm(range(df_iter)))
    ########################################################################################
    results_fut = np.array(results_fut, dtype=float)
    
    print(f'\nSHAPE OF Future-temp drop stats:\t\t{results_fut.shape}\n')

    
    print('Future cells:')
    print('----------------------------------------------------------------------------------')
    print(f'Cell_mean:\n{results_fut[:,0]}\n\nQ90:\n{results_fut[:,1]}\n')
    print(f'q75: \n{results_fut[:,2]}\n\nQ50:\n{results_fut[:,3]}\n')
    print(f'q25: \n{results_fut[:,4]}\n\nQ10:\n{results_fut[:,5]}')
    print('----------------------------------------------------------------------------------\n\n')

    


    #-----------------------------------------------------------#
    ## Convert arrays to pandas dataframes ##
    df_cells_tempstats_fut = pd.DataFrame(results_fut, columns = ['cell_mean_2h','cell_q90_2h','cell_q75_2h','cell_q50_2h','cell_q25_2h','cell_q10_2h'])

    #-----------------------------------------------------------#
    ## Save the temperature statistics-dataframes as csv files ##  
    if 'preprocess' in version_df_cells:
        print('Save output file as df_cells_preprocessed_tempstats_fut.csv')
        df_cells_tempstats_fut.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/cell_temp_drop_stats/df_cells_preprocessed_tempstats_fut.csv', index=False)
    else:
        print('Save output file as df_cells_tempstats_fut.csv')
        df_cells_tempstats_fut.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/cell_temp_drop_stats/df_cells_tempstats_fut.csv', index=False)



    print('END OF SCRIPT')
############################################################ END OF USED SCRIPT #############################################################################################################
#############################################################################################################################################################################################



#%%


##### Tests why i ran into errors with the only preprocessed dataset

# # 1) Get the cell-mask for a corresponding Cell-ID




# i = 5834   # This index does not work!!!!!
# i = 5833



# da_cell_mask = ds_cells.sel(time=df_cells.loc[i, 'datetime'])['cellID'] == df_cells.loc[i, 'clID']
# da_cell_mask = da_cell_mask.where(da_cell_mask==True)    


# try:
#     da_dtas_2h = ds_tas.sel(time=df_cells.loc[i,'datetime_ceil']+np.timedelta64(1,'h')).tas  -   ds_tas.sel(time=df_cells.loc[i, 'datetime_floor']).tas 
# except:
#     da_tas_2h = np.nan
        
# #-----------------------------------------------------------------------------------------------------------------------

# #-----------------------------------------------------------------------------------------------------------------------
# # 2) Get TAS-difference within the contour with the corresponding cell-ID
# da_tas_cell_2h = da_dtas_2h.where(da_cell_mask==1)
# #-----------------------------------------------------------------------------------------------------------------------

# #-----------------------------------------------------------------------------------------------------------------------
# # 3) Compute statistics of the TAS-drop of the cell
# mean_2h = np.nanmean(da_tas_cell_2h[:,:,0].values)
# q90_2h  = np.nanquantile(da_tas_cell_2h[:,:,0].values, 0.9)
# q75_2h  = np.nanquantile(da_tas_cell_2h[:,:,0].values, 0.75)


#%%


#%%
"""
NOTE WHY THE CODE BELOW IS NOT USED ANYMORE:
    
    The Part below computes the temperature difference field for the ceiling time and floor time of the wsgsmax-event itself 
    (which is given at half-hour intervals, i.e. 13:30, 14:30 and so on...).
    But since we figured out that the wsgsmax field is given at the end of the hour (i.e., wsgsmax field at 13:30 shows the wind gust fields at 13:59), 
    we have to take a 2 hour difference.
    
    As an example: A windgust event given at 13:30 takes the temperature difference between 15:00 and 13:00 (becuase the wsgsmax field is given at ~14:00).

"""
# """
# 1) Get a mask field for the each cellID (clID in df_cell)
    
# 2) Apply mask field to deltaTAS filed to get the temperature difference within the cell

# 3) Calculate statistical values (max, min, 90pctl, median, mean) within the cells deltaTAS-field
    
# """


# def get_tas_stats(i):
#     #---------------------------------------------
#     # 1) Isolate the cell with the correct Cell-ID
#     #---------------------------------------------
#     da_cell_mask = ds_cells.sel(time=df_cell.loc[i, 'datetime'])['cellID'] == df_cell.loc[i, 'clID']
#     da_cell_mask = da_cell_mask.where(da_cell_mask==True)          
#     da_dtas = ds_tas.sel(time=df_cell.loc[i,'datetime_ceil']).tas - ds_tas.sel(time=df_cell.loc[i, 'datetime_floor']).tas 

#     #---------------------------------------------
#     # 2) Get TAS-difference value for that cell-ID
#     #---------------------------------------------
#     da_tas_cell = da_dtas.where(da_cell_mask==1)
    
#     # Progress Metrics
#     #count += 1
#     #if count % 500 == 0:
#     #    print(f'\ni = {count} \nLoop done with {np.round(count*100/df_cell.shape[0], 3)} % of dataset')
    
#     #---------------------------------------------
#     # 3) Extract the TAS-diff values of the cell
#     #---------------------------------------------
#     #Create a DataFrame to fill with all the stats for the temperature values of the wsgsmax-cells
#     mean = np.nanmean(da_tas_cell[:,:,0].values)
#     #cell_mean.append(np.nanmean(da_tas_cell[:,:,0].values))
    
#     q90  = np.nanquantile(da_tas_cell[:,:,0].values, 0.9)
#     #cell_q90.append(np.nanquantile(da_tas_cell[:,:,0].values, 0.9))

#     q75  = np.nanquantile(da_tas_cell[:,:,0].values, 0.75)
#     #cell_q75.append(np.nanquantile(da_tas_cell[:,:,0].values, 0.75))

#     q50  = np.nanquantile(da_tas_cell[:,:,0].values, 0.50)
#     #cell_q50.append(np.nanquantile(da_tas_cell[:,:,0].values, 0.50))

#     q25  = np.nanquantile(da_tas_cell[:,:,0].values, 0.25)
#     #cell_q25.append(np.nanquantile(da_tas_cell[:,:,0].values, 0.25))

#     q10  = np.nanquantile(da_tas_cell[:,:,0].values, 0.10)
#     #cell_q10.append(np.nanquantile(da_tas_cell[:,:,0].values, 0.10))

#     return mean, q90, q75, q50, q25, q10
#     #return cell_mean, cell_q90, cell_q75, cell_q50, cell_q25, cell_q10





# ## Start parallel region
# df_iter = df_cell.shape[0]

# results = Parallel(n_jobs=-2)(delayed(get_tas_stats)(i) for i in tqdm(range(df_iter)))






# results = np.array(results, dtype=float)
# print(f'SHAPE OF NP ARRAY FOO:\t\t{results.shape}\n\n')

# print('----------------------------------------------------------------------------------')
# print(f'Cell_mean:\n{results[:,0]}\n\nQ90:\n{results[:,1]}\n')
# print(f'q75: \n{results[:,2]}\n\nQ50:\n{results[:,3]}\n')
# print(f'q25: \n{results[:,4]}\n\nQ10:\n{results[:,5]}')
# print('----------------------------------------------------------------------------------\n\n\n')






# df_cells_tempstats = pd.DataFrame(results, columns = ['cell_mean','cell_q90','cell_q75','cell_q50','cell_q25','cell_q10'])

# print(df_cells_tempstats.shape)

# df_cells_tempstats.to_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_tempstats.csv', index=False)




#===========================================================================================================================================================================================
#===========================================================================================================================================================================================
#===========================================================================================================================================================================================
#===========================================================================================================================================================================================


#============================================================================================================================
####################################  OLD WWAY TO LOAD DATA #################################################################
#============================================================================================================================



# #====================================================================================================================================================================================#
# #------                Input Parameters                             ---------#
# #----------------------------------------------------------------------------#
# # Which model is used (historic/future)?
# model='hist'
# # model='fut'


# # Which df_cells is used (after cleaning (order of 10k cells) or after pre-processing (order of 100k cells))?
# # version_df_cells     = 'df_cells.csv'
# version_df_cells     = 'df_cells_preprocessed.csv'

# #=====================================================================================================================================================================================#







# #######################   Historic   ##############################
# if 'hist' in model:
#     print('Computation of historic temp. drop stats')
#     # Read in DataSets (cell-outline, wsgsmax-field, TAS-field)
#     rootdir = '/net/pc200057/nobackup_1/users/frei/CPMs/'
#     ds_cells =  xr.open_dataset('/net/pc200057/nobackup_1/users/frei/track_cluster/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist/allyears_merged/cells.nc').drop_vars('lev')
#     ds_tas = xr.open_mfdataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist_tas_lonlatbox/*.nc')
#     ds_tas = sel_month(ds_tas)
#     ds_wsgsmax = xr.open_dataset(f'{rootdir}/fields_lonlatbox/HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_wsgsmax_merged_lonlatbox/wsgsmax.his.CXFPS025.HCLIM38h1_CXFPS_fRACMOfECEARTH_r14_hist.1hr_lonlatbox_merged.nc')
#     # Select months MJJAS (note in future wsgsmax sets, these months are already selected)
#     ds_wsgsmax = sel_month(ds_wsgsmax)
    
    
#     path_df_cells = f'/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/{version_df_cells}'
#     # Read cell data with own function read_df_cell (-> this fct adds the datetime floor and ceiling column and converts them into correct dtype)
#     df_cells = read_df_cell(path_df_cells)

# #######################    Future    ##############################
# elif 'fut' in model:    
#     print('Computation of future temp. drop stats')
#     # Read in DataSets (cell-outline, wsgsmax-field, TAS-field)
#     ds_cells = xr.open_dataset('/net/pc200057/nobackup_1/users/frei/CPM_future/track_cluster_fut/output_tracking_sdomain/celltrack/HCLIM38h1_CXFPS_fRACMOfECEARTH_r04_rcp85/cells_merged.nc').drop_vars('lev')
#     ds_wsgsmax = xr.open_mfdataset('/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/fields_lonlatbox_selmonth/wsgsmax_lonlatbox_selmonth/wsgsmax.*.nc')
#     ds_tas = xr.open_mfdataset('/net/pc200057/nobackup_1/users/frei/CPM_future/fields_lonlatbox/fields_lonlatbox_selmonth/tas_lonlatbox_selmonth/tas.*.nc')
    
    
#     path_df_cells_fut = f'/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/{version_df_cells}'
#     # Read cell data with own function read_df_cell (-> this fct adds the datetime floor and ceiling column and converts them into correct dtype)
#     df_cells = read_df_cell(path_df_cells_fut)

# #######    Raise exception if no of the two dataframes is selected     ########
# else:
#     raise Exception("Sorry, no valid model chosen. Choose either: 'historic' or 'future' model.") 
    


#----------------------------------------------------------------------------------------------------------------
# Deprecated due to fucntion read_df_cell                                      # This is what the function read_df_cell does:
#----------------------------------------
# # Dataframe with cells (excluding ocean and boundary touching)
# df_cells = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut.csv')
# df_cells['datetime'] = pd.to_datetime(df_cells['datetime'])
# df_cells['datetime_ceil'] = pd.to_datetime(df_cells['datetime_ceil'])
# df_cells['datetime_floor'] = pd.to_datetime(df_cells['datetime_floor'])
#----------------------------------------------------------------------------------------------------------------



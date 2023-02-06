#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:48:03 2022

@author: frei
"""
import pandas as pd
import numpy as np

from plot_case_studies_annimation import plot_cellfield_tas_precip_div
from data_processing import sel_month, add_datetime64_column, sel_month


#%%
## Load data

df_cells_fut = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut.csv')
df_cells_fut.name = 'Future cells (1995-2005)'
df_cells_fut = add_datetime64_column(df_cells_fut)

df_cells = pd.read_csv('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells.csv')
df_cells.name = 'Historic cells (1995-2005)'
df_cells = add_datetime64_column(df_cells)




#%%
#=============================================================================#
# Example 1:                                                                  #
#                                                                             #
#   Future: TempDrop (-18C +/- 2                                              #
#                    30m/s +/- 2)                                             #
#=============================================================================#
plot_cellfield_tas_precip_div(
    df = df_cells_fut,
    x_var = 'cell_q10_2h',
    y_var = 'peakVal',
    x_value = -12,
    y_value = 30,
    x_sel_width=0.5,
    y_sel_width=0.5
    )

#%%

#=============================================================================#
# Example 2:                                                                  #
#                                                                             #
#   Hist: TempDrop (-18C +/- 2                                                #
#                    30m/s +/- 2)                                             #
#=============================================================================#
plot_cellfield_tas_precip_div(
    df = df_cells,
    x_var = 'cell_q10_2h',
    y_var = 'peakVal',
    x_value = -15,
    y_value = 33,
    x_sel_width=2,
    y_sel_width=2
    )

#%%
# Find case of interest
filt = (df_cells['datetime'] > np.datetime64('1997-08-03T16:00:00.000000000')) & (df_cells['datetime'] < np.datetime64('1997-08-03T17:00:00.000000000'))
df_case = df_cells[filt]

# Select biggest cell
df_case = df_case[df_case['grd_clarea']>10]
#=============================================================================#
# Example 3:                                                                  #
#                                                                             #
#   Hist: Case at 09-03-1997                                                  #
#         Tempdrop q10_2h: -12.19
#         PeakVal        :   27.3 m/s
#=============================================================================#

df_case.name='Historic'
plot_cellfield_tas_precip_div(
    df = df_case,
    x_var = 'cell_q10_2h',
    y_var = 'peakVal',
    x_value = -12,
    y_value = 27,
    x_sel_width=2,
    y_sel_width=2
    )















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:38:08 2022

@author: frei
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_processing import sel_month, add_datetime64_column, read_df_cell


#%%

## Load Data

df_cells = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells.csv')
df_cells_fut = read_df_cell('/net/pc200057/nobackup_1/users/frei/track_cluster/cluster_stats_df/df_cells_fut.csv')

#%%

#-----------------------------------------
# Probability of exceedance calculations #
#-----------------------------------------
def calc_proba_of_exceedance(df):
    # Create and array with thresholds
    wsgsmax_thr = np.arange(20, math.ceil(df['peakVal'].max()+1), 0.1)
    list_exc =[]
    for thr in wsgsmax_thr:
        nr_exc = 0
        for val in df['peakVal']:
            if thr < val:
                nr_exc += 1
        list_exc.append(nr_exc)
    exce_array = np.asarray(list_exc) / df.shape[0]
    return exce_array, wsgsmax_thr
#%%

hist_excedance = calc_proba_of_exceedance(df_cells)
fut_excedance  = calc_proba_of_exceedance(df_cells_fut)

#%%
# fig = plt.subplots( figsize=(15,10) )
# gs = GridSpec(nrows=1, ncols=2, height_ratios=[20], width_ratios=[20,20] )

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(hist_excedance[1], hist_excedance[0], label='Historic run')
ax.plot(fut_excedance[1], fut_excedance[0], label='Future run')
ax.set_yscale('log')
ax.set_title('Probability of exceedance')
ax.set_ylabel('Log of probability')
ax.set_xlabel(f'wsgsmax $[m/s]$')
ax.legend()

plt.savefig('/usr/people/frei/MaxFrei/Max_figures/general_analysis/CellTrack_Analysis/Prob_exceedance_wsgsmax/probab_of_exceedance_futhist', dpi=100)


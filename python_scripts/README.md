Here I try to list the python files used for the project in chronological order (as it appears in the report) and with a short description of its main functionality.

__domain_selection.py:__
  - Plots the overview over the whole domain and the subdomain around the Netherlands. 
  - Produced Plots include the maximal windgust, the 99th percentile windgust and these two fields divided by each other.
  
  
__windgust_overview.py:__
  - Plots the windgust dailymax data against the wind at 1000hPa (dailymax) on a scatter plot. 
  - Note: The 1000hPa data is not the the origin of the windgust data. If you're interested in such a plot, find surface wind dataset.
  
  
__wsgsmax_elevation_dependence.py:__
  - Main functionality is to asses the height dependance of wsgsmax-values
  - Two plotting methods are included in one function:
      - Plot a 2d histogramm of elevation (y-axis) and maximal wsgsmax value (x-axis)
      - Bin the wsgsmax data into 4 elevation groups and plot a 1-CDF plot (probability of exceedance)

  All plots include statistical measures to test robustness. (P25, P50, P75, P90)
  
*Note: At this point a celltracking algortighm was applied to wsgsmax datasets (future and present)*


__CellTracking_DataProcessing_Cleaning.py:__
  - Preprocess the data (-> drop cells that touch boundary or when their centre of mass is located over the ocean)
  - Add a datetime64 column to cell-track output dataframe (IMPORTANT for further steps, such as adding temperature drop statisitcs).
  - Compute ration of max value within cell and the do main avergae (inhomogeneity measure)
  (- Add numbers of cells that occured on one day. This metric was not futher used.)
  

__compute_TAS_quantiles_parallel.py:__
  - Drop all cells that occur on last hour of the year (i.e. september 30th at 22:30), becuase the corresponding temperature field already lies in october
  - Compute the TAS-drop value (90th percentil to 10th percentile) for every cell (parallel -> takes ~20 hours for 65'000 cells with parallel computing).
  - Safe the temperature drop as two seperate dataframes


__CellTracking_Filtering.py:__
  - Filter cells with two filters (field-inhomogeneity and convective-time filter)
  - Convective time filter only keeps the cells that occur between early afternoon and early morning.
  - Field inhomogeneity only keeps cells with a homogeneity ratio heigher than a threshold of 3 (threshold can be changed for sensitivity studies).
  - Add the teperature drop stats to the cell-track dataframes (df_cells and df_cells_future).


__CellTracking_Plotting.py__
  - Plots the distirbution of cells (cell_area vs wsgsmax peakValue within cell)


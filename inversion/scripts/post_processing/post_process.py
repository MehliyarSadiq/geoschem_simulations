#!/usr/bin/env python
# coding: utf-8

# import all the necessary modules 
# import all functions needed to post process
from tools import *
import sys

args = sys.argv
print(args[1])
# In[ ]:


# %%time
### this cell could do all post-processing for a case, over a few months, or one month

year = 2016
assim_month = int(args[1]) # month of flux tagged
lag_window = 3 #
case_name = 'CO2-' + str(year) 
mod_dir = '/geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs/' + str(year) + '/' + case_name + '/nd51/' + short_name_of_month(assim_month) + '/'

for imonth in range(assim_month-lag_window+1, assim_month+1): # months of tagged runs

    # convert bpch files to netcdf
    bpch_to_nc_mass(data_dir = mod_dir,
                            year = year,
                            month = imonth)
    # combine daily outputs into monthly
    combine_daily_to_one(data_dir = mod_dir,
                                 year = year,
                                 month = imonth,
                                 str1 = 'ts_satellite.',
                                 str2 = '.nc')
    # flatten 4d to 2d array
    flatten_4d_to_2d(data_dir = mod_dir,
                                 year = year,
                                 month = imonth,
                                 str1 = 'ts_satellite.',
                                 str2 = '.nc')

    # interpolate to satellite grid
    interpolate_model_to_satellite(mod_dir = mod_dir, 
                                               sat_dir = '/geos/u73/msadiq/satellite/oco-2/', 
                                               year = year, 
                                               month = imonth, 
                                               str1_mod = '2d_ts_satellite.', 
                                               str1_sat = '2d_OCO2_extract_')
    # calculate XCO2 difference
    delta_Y(mod_dir = mod_dir, 
                    sat_dir = '/geos/u73/msadiq/satellite/oco-2/', 
                    year = year, 
                    month = imonth, 
                    str1_mod = 'interpolated_2d_ts_satellite.', 
                    str1_sat = '2d_OCO2_extract_',
                    mode = 'N')

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# collection of functions
# simple functions could be tested within the cell it is defined
# more complex ones uses the cell below to do the testing
# rules: 
# 1, each function should not take longer than 1min to run
# 2, not too long... 20 lines?


# In[2]:


# for plots
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')
from matplotlib import rcParams
rcParams["savefig.dpi"] = 300
rcParams["font.size"] = 12
import warnings
warnings.filterwarnings('ignore')
from gamap_colormap import WhGrYlRd
# packages
import math
import matplotlib.pyplot as plt # load plotting libraries
import cartopy.crs as ccrs
import numpy as np
import numpy.ma as ma
import xarray as xr
import regionmask
import re
from bpch2nc import bpch_2_netcdf
import scipy.linalg as sp
import pandas as pd
from util.functions import create_masks
# numpy precision
np.set_printoptions(suppress=True)


# In[3]:


# same as above, only long names
def long_name_of_month(month): # returns long name of month
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    name_month = month_names[month-1]
    return name_month
#long_name_of_month(5)


# In[4]:


# input: 3
# output: 'Mar'
def short_name_of_month(month): # returns short name of month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    name_month = month_names[month-1]
    return name_month
#short_name_of_month(12)


# In[6]:


# input: number of month: 1-12
# output: a string '01', '02', '03', ..., '10', '11', '12'
def month_string(month): # string for a month, input 3 output '03'
    if(month >= 10):
        mm_str = str(month)
    else:
        mm_str = '0' + str(month)
    return mm_str
#month_string(3)


# In[7]:


# input: 2018, 2
# output: array(['2018-02-01', '2018-02-02', '2018-02-03', '2018-02-04', ..., '2018-02-27', '2018-02-28'], dtype='datetime64[D]')
# needs some work
def days_str_in_a_month(year, month): # returns days in a month, in numpy datetime64[D] format
    month_str = month_string(month)
    if month <= 11:
        first_day = str(year) + '-' + month_str + '-01'
        month_str_p1 = month_string(month+1)
        last_day_p1 = str(year) + '-' + month_str_p1 + '-01'
    else:
        first_day = str(year) + '-' + month_str + '-01'
        month_str_p1 = month_string(1)
        last_day_p1 = str(year+1) + '-' + month_str_p1 + '-01'
    
    return np.arange(first_day, last_day_p1, dtype='datetime64[D]')
#days_str_in_a_month(2018,1)


# In[9]:


# a correct leap year function! might not needed at all
# input: year in integer
# output: True if leap year, False if not
def leap_year(year):
    ans = False
    if year%4 == 0 and year%100 !=0:
        ans = True
    if year%400 == 0:
        ans = True
    return ans
#leap_year(2012)


# In[10]:


def days_in_month(year, month):
    leap_flag = leap_year(year)
    days_in_a_year = [31,28,31,30,31,30,31,31,30,31,30,31]
    if leap_flag == True and month == 2: return 29
    else: return days_in_a_year[month-1]

#days_in_month(2010,2)


# In[12]:


# Approximate the area of a spatial grid square from the latitudes and longitudes of the diagonal vertices
def area_latlon(lat1, lon1, lat2, lon2):
    # This function calculates the area (in km^2) of a spatial grid square, given the latitudes and longitudes of the two diagonal vertices of the grid square.
    # lat/lon is in angle; lat: [-90:90]; lon:[-180:180].
    # lat1/lon1 and lat2/lon2 are thus the diagonal vertices of the square grid.
    lat1 = lat1/180*np.pi
    lat2 = lat2/180*np.pi
    lon1 = lon1/180*np.pi
    lon2 = lon2/180*np.pi
    A = np.absolute(6371.009**2*(np.sin(lat2)-np.sin(lat1))*(lon2-lon1))
    return A
#area_latlon(10,0,11,1)


# In[14]:


# input: an xarray dataarray or dataset with lon and lat coords
# output: [[lat1,lon1], [lat2,lon2]], smallest box that covers dataset
def lat_lon_bounds(ds):
    minlat = math.floor(ds.lat.min())
    maxlat = math.ceil(ds.lat.max())
    minlon = math.floor(ds.lon.min())
    maxlon = math.ceil(ds.lon.max())
    mins = [minlat, minlon] 
    maxs = [maxlat, maxlon]
    
    # round up to nearest number divisible by 5, GEOS-Chem grids keep these grid points even (?)
    for i in range(len(mins)):
        if mins[i]%5 != 0: mins[i] = mins[i] - mins[i]%5
    for i in range(len(maxs)):
        if maxs[i]%5 != 0: maxs[i] = maxs[i] + 5 - maxs[i]%5
    ans = [mins,maxs]   
    return ans


# In[17]:


#fname = '/geos/d21/msadiq/TNO-GHGco/gridded/TNO_2018_0.05x0.1.nc'
#ds = xr.open_dataset(fname)
#lat_lon_bounds(ds)


# In[19]:


# get country total of a variable
def country_totals(co2_xarray, countries, varnames):
    # create masks using regionmask
    mask1 = regionmask.defined_regions.natural_earth.countries_50.mask(co2_xarray)
    # sum up total emissions for each country
    co2_countries = co2_xarray.groupby(mask1).sum('stacked_lat_lon')
    abbrevs = regionmask.defined_regions.natural_earth.countries_50[co2_countries.region.values].abbrevs
    names = regionmask.defined_regions.natural_earth.countries_50[co2_countries.region.values].names
    co2_countries.coords['abbrevs'] = ('region', abbrevs)
    co2_countries.coords['names'] = ('region', names)

    country_totals = np.zeros(len(countries))
    for i in range(len(countries)):
        tmp = co2_countries.isel(region=(co2_countries.names == countries[i]))
        country_totals[i] = tmp[varnames[0]].values
    return country_totals


# In[21]:


#fname = '/geos/d21/msadiq/TNO-GHGco/gridded/TNO_2018_0.05x0.1.nc'
#ds = xr.open_dataset(fname)

#countries = ['Germany', 'United Kingdom', 'Italy', 'France', 'Poland', 'Spain']
#var    = ['co2_ff']
#before = country_totals(ds[var], countries, var)

#bar_pos = np.arange(len(countries)) + 1 # position of the bars
#fig = plt.figure(figsize=[8, 4])
#width = 0.3
#bars_before = plt.bar(bar_pos, before*1e-9, width=width, color = 'b', label='before')
#plt.xticks(bar_pos, countries)
#plt.title('Annual total emission (Tg/yr) of ' + var[0], loc='left')
#plt.legend()


# In[ ]:


# regrid an xarray dataarray from finer resolution to coarser resolution (res)
# dr is the dataarray
def regrid_fine_to_coarse_sum_dr(dr, target, res_lat, res_lon):
    target_copy = target.copy()
    for ilon, lon in enumerate(target['lon'].values):
        for ilat, lat in enumerate(target['lat'].values):
            subset = dr.sel(lat=slice(lat-res_lat/2,lat+res_lat/2), lon = slice(lon-res_lon/2,lon+res_lon/2))
            target_copy[ilat,ilon] = subset.sum().values
    
    return target_copy


# In[ ]:


# regrid a 3-d (time, lat, lon) xarray dataarray from finer resolution to coarser resolution (res_lat, res_lon)
def regrid_fine_to_coarse_sum_dr_monthly(dr, target, res_lat, res_lon):
    target_copy = target.copy()
    for ilon, lon in enumerate(target['lon'].values):
        for ilat, lat in enumerate(target['lat'].values):
            subset = dr.sel(lat=slice(lat-res_lat/2,lat+res_lat/2), lon = slice(lon-res_lon/2,lon+res_lon/2))
            target_copy[:,ilat,ilon] = subset.sum(dim=('lat','lon')).values
            target_copy.attrs = dr.attrs
    return target_copy


# In[23]:


# regrid an xarray dataarray from finer resolution to coarser resolution (res)
# ds is the dataset, varnames are variables need to be regridded
def regrid_fine_to_coarse_sum(ds, varnames, res_lat, res_lon):
    # get outer bounds of input data, [[lat1,lon1], [lat2,lon2]] 
    bounds = lat_lon_bounds(ds)
    # target grid
    target = xr.DataArray(0, dims=('lat', 'lon'), 
                            coords={'lat': np.arange(bounds[0][0], bounds[1][0] + res_lat, res_lat), # larger than CHE domain
                                    'lon': np.arange(bounds[0][1], bounds[1][1] + res_lon, res_lon)}) # slightly smaller than TNO domain
    target = target.astype(dtype='float64')
    output = target.to_dataset(name = varnames[0])
    for ivar in varnames: output[ivar] = target.copy()
    # regridding
    for ivar in varnames:
        dr = ds[ivar]
        for ilon, lon in enumerate(target['lon'].values):
            for ilat, lat in enumerate(target['lat'].values):
                subset = dr.sel(lat=slice(lat-res_lat/2,lat+res_lat/2), lon = slice(lon-res_lon/2,lon+res_lon/2))
                target[ilat,ilon] = subset.sum().values
                target.attrs = dr.attrs
        output[ivar] = target.copy()
    return output


# In[25]:


#%%time
#fname = '/geos/d21/msadiq/TNO-GHGco/gridded/TNO_2018_0.05x0.1.nc'
#ds = xr.open_dataset(fname)
#ds_regrid = regrid_fine_to_coarse_sum(ds, ['co2_ff'], 0.25, 0.3125) # to coarse resolution

#countries = ['Germany', 'United Kingdom', 'Italy', 'France', 'Poland', 'Spain']
#var    = ['co2_ff']
#before = country_totals(ds[var], countries, var)
#after  = country_totals(ds_regrid[var], countries, var)

#bar_pos = np.arange(len(countries)) + 1 # position of the bars
#fig = plt.figure(figsize=[9, 4])
#width = 0.2
#bars_before = plt.bar(bar_pos-0.1, before*1e-9, width=width, color = 'b', label='before')
#bars_after  = plt.bar(bar_pos+0.1, after*1e-9, width=width, color = 'g', label='after')
#plt.xticks(bar_pos, countries)
#plt.title('Annual total emission (Tg/yr) of ' + var[0], loc='left')
#plt.legend()


# In[26]:


# convert kg/year to kg/m2/s
# inputs: dataset (xarray), lat and lon, variable names in dataset
# outputs: dataset
def unit_convert_ds_yearly(ds, varnames, res_lat, res_lon):
    for ivar in varnames:
        dr = ds[ivar]
        # calculate grid area (using the area_latlon) and compute flux
        for ilat, lat in enumerate(dr['lat'].values):
            area = 1e6 * area_latlon(lat1=lat, lat2=lat+res_lat, 
                                     lon1=10, lon2=10+res_lon) # m^2, longitude doesn't matter
            dr[ilat,:] = dr[ilat,:]/(area*3600*24*365) # kg/m2/s
        ds[ivar] = dr.copy()
        ds[ivar].attrs['units'] = 'kg/m2/s'
        ds[ivar].attrs['long_units'] = 'kg(' + ivar + ')/m2/s'
    return ds


# In[30]:


#fname = '/geos/d21/msadiq/TNO-GHGco/gridded/TNO_2018_0.05x0.1.nc'
#ds = xr.open_dataset(fname)
##ds['co2_ff'].plot()
#ds_flux = unit_convert_ds_yearly(ds, ['co2_ff'], 0.05, 0.1)
#ds_flux['co2_ff'].plot()


# In[42]:


# convert kg/year to kg/m2/s
# inputs: dataarray (xarray), lat and lon, variable names in dataset
# outputs: dataarray
def unit_convert_dr_yearly(dr, res_lat, res_lon):
    dr_copy = dr.copy()
    # calculate grid area (using the area_latlon) and compute flux
    for ilat, lat in enumerate(dr_copy['lat'].values):
        area = 1e6 * area_latlon(lat1=lat, lat2=lat+res_lat, 
                                    lon1=10, lon2=10+res_lon) # m^2, longitude doesn't matter
        dr_copy[ilat,:] = dr_copy[ilat,:]/(area*3600*24*365) # kg/m2/s
    dr_copy.attrs['units'] = 'kg/month'
    return dr_copy


# In[37]:


#fname = '/geos/d21/msadiq/TNO-GHGco/gridded/TNO_2018_0.05x0.1.nc'
#ds = xr.open_dataset(fname)
#dr = ds['co2_ff']
##ds['co2_ff'].plot()
#dr_flux = unit_convert_dr_yearly(dr, 0.05, 0.1)
#dr_flux.plot()


# In[39]:


#convert annual mean kg/m2/s to kg/year
def unit_convert2_ds_yearly(ds, varnames, res_lat, res_lon):
    for ivar in varnames:
        dr = ds[ivar]
        # use grid area function
        for ilat, lat in enumerate(dr.lat.values):
            area = 1e6 * area_latlon(lat1 = lat, lat2 = lat + res_lat,
                                     lon1 = 10, lon2 = 10 + res_lon) # m^2, longitude doesn't matter
            dr[ilat,:] = dr[ilat,:] * area * (3600*24*365) # kg/year
        ds[ivar] = dr.copy()
        ds[ivar].attrs['units'] = 'kg/year'
        ds[ivar].attrs['long_units'] = 'kg(' + ivar + ')/year'
    return ds


# In[41]:


#fname = '/geos/d21/msadiq/TNO-GHGco/gridded/TNO_2018_0.25x0.3125.nc' # unit: kg/m2/s
#ds = xr.open_dataset(fname)
#ds_flux = unit_convert2_ds_yearly(ds, ['co2_ff'], 0.25, 0.3125)
#ds_flux['co2_ff'].plot()


# In[44]:


#convert annual mean kg/m2/s to kg/year
def unit_convert2_dr_yearly(dr, res_lat, res_lon):
    dr_copy = dr.copy()
        # use grid area function
    for ilat, lat in enumerate(dr_copy.lat.values):
        area = 1e6 * area_latlon(lat1 = lat, lat2 = lat + res_lat,
                                    lon1 = 10, lon2 = 10 + res_lon) # m^2, longitude doesn't matter
        dr_copy[ilat,:] = dr_copy[ilat,:] * area * (3600*24*365) # kg/year
    dr_copy.attrs['units'] = 'kg/year'
    return dr_copy


# In[49]:


#fname = '/geos/d21/msadiq/TNO-GHGco/gridded/TNO_2018_0.25x0.3125.nc' # unit: kg/m2/s
#ds = xr.open_dataset(fname)
#dr = ds['co2_ff']
#dr_flux = unit_convert2_dr_yearly(dr, 0.25, 0.3125)
#dr_flux.plot()


# In[ ]:


# convert annual mean kg/month to kg/m2/s
# works for leap years
def unit_convert_dr_monthly(dr, year, res_lat, res_lon):
    dr_copy = dr.copy()
    days_list = []
    for imonth in range(12): days_list.append(days_in_month(year, imonth+1))
    # use grid area function
    for imonth in range(12):
        for ilat, lat in enumerate(dr_copy.lat.values):
            area = 1e6 * area_latlon(lat1 = lat, lat2 = lat + res_lat,
                                     lon1 = 10, lon2 = 10 + res_lon) # m^2, longitude doesn't matter
            dr_copy[imonth, ilat, :] = dr_copy[imonth, ilat, :] / (area * days_list[imonth] * (3600*24)) # kg/year
    dr_copy.attrs['units'] = 'kg/month'
    return dr_copy


# In[50]:


# convert annual mean kg/m2/s to kg/month
# works for leap years
def unit_convert2_dr_monthly(dr, year, res_lat, res_lon):
    dr_copy = dr.copy()
    days_list = []
    for imonth in range(12): days_list.append(days_in_month(year, imonth+1))
    # use grid area function
    for imonth in range(12):
        for ilat, lat in enumerate(dr_copy.lat.values):
            area = 1e6 * area_latlon(lat1 = lat, lat2 = lat + res_lat,
                                     lon1 = 10, lon2 = 10 + res_lon) # m^2, longitude doesn't matter
            dr_copy[imonth, ilat, :] = dr_copy[imonth, ilat, :] * area * days_list[imonth] * (3600*24) # kg/year
    dr_copy.attrs['units'] = 'kg/month'
    return dr_copy


# In[51]:


# convert kg/m2/s to kg/(number of hours)
def unit_convert2_hours(dr, res_lat, res_lon, hours):
    dr_copy = dr.copy()
    # use grid area function
    for ilat, lat in enumerate(dr_copy.lat.values):
        area = 1e6 * area_latlon(lat1 = lat, lat2 = lat + res_lat,
                                 lon1 = 10, lon2 = 10 + res_lon) # m^2, longitude doesn't matter
        dr_copy[:,ilat,:] = dr_copy[:,ilat,:] * area * (3600*hours) # kg/(#hours)
    return dr_copy


# In[21]:


# convert bpch files to netcdf format, 
# given input directory, a year and month
# naming convention is ts_satellite.yyyymmdd.bpch
# output format is ts_satellite.yyymmdd.nc
# need tracerinfo.dat and diaginfo.dat in the same directory
# uses days_str_in_a_month function
def bpch_to_nc_mass(data_dir, year, month):
    
    name_bpch1 = 'ts_satellite.'
    
    tinfo_file = data_dir + 'tracerinfo.dat'
    dinfo_file = data_dir + 'diaginfo.dat'
    
    days = days_str_in_a_month(year, month)
    
    for iday in np.arange(len(days)):
        day_string = days[iday] # format not right for the following function
        #print('converting bpch to netcdf on day: ', day_string)
        new_day_string = re.sub("[^0-9]", "", str(day_string)) # strip off '-'s

        bpchfile = data_dir + name_bpch1 + new_day_string + '.bpch'
        ncfile = data_dir + name_bpch1 + new_day_string + '.nc'

        bpch_2_netcdf(bpchfile=bpchfile, 
                      tinfo_file=tinfo_file, 
                      dinfo_file=dinfo_file, 
                      ncfile=ncfile)
    print('converted daily bpch outputs to netcdf format')
    return

#bpch_to_nc_mass(data_dir = '/geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs/CO2-2018/nd51/',
#               year = 2018,
#               month = 1)


# In[22]:


# combine a month of daily netcdf files into one file
# input file name format has to be str1 + yyyymmdd + str2
# output file name will be str1 + yyyymm + str2

def combine_daily_to_one(data_dir, year, month, str1, str2):
    days = days_str_in_a_month(year, month)
    
    # get first file, copy attributes, dimensions from it
    # prepare output file format
    first_day = days[0]
    new_day_string = re.sub("[^0-9]", "", str(first_day)) # strip off '-'s

    first_file = xr.open_dataset(data_dir + str1 + new_day_string + str2)
    varnames = list(first_file.data_vars.keys())  # a list of variable names

    lon = first_file.lon
    lat = first_file.lat
    lev = first_file.lev
    time = days
    target = xr.DataArray(np.nan, coords=[time, lev, lat, lon], dims=['time', 'lev', 'lat', 'lon'])

    
    output = target.to_dataset(name = 'null')
    output.attrs = first_file.attrs
    for ivar in varnames: output[ivar] = target.copy()

    
    # combine the netcdf files into one, monthly
    for iday in np.arange(len(days)):
        day_string = days[iday]
        #print(day_string)
        new_day_string = re.sub("[^0-9]", "", str(day_string)) # strip off '-'s
        
        ncfile = data_dir + str1 + new_day_string + str2

        ds_tmp = xr.open_dataset(ncfile)
    
        for ivar in varnames:
            output[ivar][iday,:,:,:] = ds_tmp[ivar][0,:,:,:].copy()
            output[ivar].attrs = ds_tmp[ivar].attrs
    
    # output file name
    first_day_string = re.sub("[^0-9]", "", str(first_day)) # strip off '-'s
    monthly_string = first_day_string[0:6]
    output.to_netcdf(data_dir + str1 + monthly_string + str2)
    print('created ' + str1 + monthly_string + str2)


# In[23]:


# flatten 4d arrays to 2d,
# input data file name has to be str1 + yyyymm + str2
# output file name will be 2d_ + str1 + yyyymm + str2
def flatten_4d_to_2d(data_dir, year, month, str1, str2):
    input_file = data_dir + str1 + str(year) + month_string(month) + str2
    ds = xr.open_dataset(input_file)
    varnames = list(ds.data_vars.keys())  # a list of variable names
    record = range(len(ds.lon) * len(ds.lat) * len(ds.time)) # length of array
    # output data format
    target = xr.DataArray(np.nan, coords=[record, ds.lev], dims=['record', 'levels'])
    ds_output = target.to_dataset(name = 'null')
    
    for ivar in varnames: ds_output[ivar] = target.copy()
    
    flat = ds[ivar][:,0,:,:].to_dataframe() # flatten a variable at one level
    flat.reset_index(inplace=True) # get indices to prepare output coordinates
    
    lat = xr.DataArray(0, coords=[record], dims=['record'])
    lon = xr.DataArray(0, coords=[record], dims=['record'])
    date = xr.DataArray(0, coords=[record], dims=['record'])
    lat.values = flat['lat']
    lon.values = flat['lon']
    date.values = flat['time']
    ds_output['lat'] = lat
    ds_output['lon'] = lon
    ds_output['date'] = date
    ds_output

    for ivar in varnames:
        target = xr.DataArray(np.nan, coords=[record, ds.lev], dims=['record', 'levels'])
        for ilev in range(len(ds.lev)):
            flat = ds[ivar][:,ilev,:,:].to_dataframe() # flatten a variable at one level
            target[:,ilev] = flat[ivar] # store output to a dataarray
        ds_output[ivar] = target.copy() # store dataarray to dataset
        #print(ivar + ' done!')

    output_file = '2d_' + str1 + str(year) + month_string(month) + str2
    ds_output.to_netcdf(data_dir + output_file)
    print('created ' + output_file)


# In[24]:


# quite fast for one variable (<1m), but rather slow for 20+ tagged species (>15m)
def interpolate_model_to_satellite(mod_dir, sat_dir, year, month, str1_mod, str1_sat):
    
    # read model and satellite files
    mod_file = str1_mod + str(year) + month_string(month) + '.nc'
    ds_mod = xr.open_dataset(mod_dir + mod_file)
    sat_file = str1_sat + str(year) + month_string(month) + '.nc'
    ds_sat = xr.open_dataset(sat_dir + sat_file)
    # read in variables and compute
    varnames = list(ds_mod.data_vars.keys())  # list of variable name
    needed_vars = [i for i in varnames if i.startswith('SpeciesConc_CO2')] # species var names
    record            = ds_mod['record'].values
    levels_model      = ds_mod['levels']
    surf_press_mod    = ds_mod['PEDGE_S_PSURF']
    profile_press_mod = surf_press_mod * levels_model # model pressure at different levels
    profile_press_sat = ds_sat['pressure'] # satellite pressure profile of different levels
    # find records where measurements are available
    surf_press = profile_press_sat[:,19].values
    nonzero_record = np.where(surf_press != 0)[0] # loop over these records only
    # prepare output dataset
    interpolated = ds_sat['pressure'].to_dataset() # output dataset
    # variables need not to be interpolated
    noneed_interp = ['lat', 'lon', 'date']
    for ivar in noneed_interp: interpolated[ivar] = ds_mod[ivar].copy()
    # tmp dataarray to store interpolated output
    tmp = xr.DataArray(data = np.nan,
                       dims=('record', 'levels'),
                       coords=[record, ds_mod.levels])
    tmp = tmp.astype(dtype = 'float32')
    for ivar in needed_vars: interpolated[ivar] = tmp.copy()
    
    print('interpolation began')
    # interpolation
    for ivar in needed_vars:
        before = ds_mod[ivar] # co2 before interpolation
        for irecord in nonzero_record:
            var_before  = before[irecord,:].values # a co2 profile
            pres_before = np.log(profile_press_mod[irecord].values) # log space
            pres_after  = np.log(profile_press_sat[irecord])
            # linear interpolation on log space    
            interpolated[ivar][irecord,:] = np.interp(x  = pres_after, 
                                                      xp = np.flip(pres_before), # increasing order
                                                      fp = np.flip(var_before))
        print(ivar, 'done')
    
    output_file = mod_dir + 'interpolated_' + str1_mod + str(year) + month_string(month) + '.nc'
    interpolated.to_netcdf(output_file)
    
    print('created ' + 'interpolated_' + str1_mod + str(year) + month_string(month) + '.nc')

#interpolate_model_to_satellite(mod_dir = '/geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs/CO2-2018-03/nd51/', 
#                                   sat_dir = '/geos/u73/msadiq/satellite/oco-2/', 
#                                   year = 2018, 
#                                   month = 3, 
#                                   str1_mod = '2d_ts_satellite.', 
#                                   str1_sat = '2d_OCO2_extract_')


# In[ ]:


# quite fast for one variable (<1m), but rather slow for 20+ tagged species (>15m)
def interpolate_model_to_satellite2(mod_dir, sat_dir, year, month, str1_mod, str1_sat, background):
    
    # read model and satellite files
    mod_file = str1_mod + str(year) + month_string(month) + '.nc'
    ds_mod = xr.open_dataset(mod_dir + mod_file)
    sat_file = str1_sat + str(year) + month_string(month) + '.nc'
    ds_sat = xr.open_dataset(sat_dir + sat_file)
    # read in variables and compute
    varnames = list(ds_mod.data_vars.keys())  # list of variable name
    needed_vars = [i for i in varnames if i.startswith('SpeciesConc_CO2Tag')] # species var names
    record            = ds_mod['record'].values
    levels_model      = ds_mod['levels']
    surf_press_mod    = ds_mod['PEDGE_S_PSURF']
    profile_press_mod = surf_press_mod * levels_model # model pressure at different levels
    profile_press_sat = ds_sat['pressure'] # satellite pressure profile of different levels
    # find records where measurements are available
    surf_press = profile_press_sat[:,19].values
    nonzero_record = np.where(surf_press != 0)[0] # loop over these records only
    # prepare output dataset
    interpolated = ds_sat['pressure'].to_dataset() # output dataset
    # variables need not to be interpolated
    noneed_interp = ['lat', 'lon', 'date']
    for ivar in noneed_interp: interpolated[ivar] = ds_mod[ivar].copy()
    # tmp dataarray to store interpolated output
    tmp = xr.DataArray(data = np.nan,
                       dims=('record', 'levels'),
                       coords=[record, ds_mod.levels])
    tmp = tmp.astype(dtype = 'float32')
    for ivar in needed_vars: interpolated[ivar] = tmp.copy()
    
    print('interpolation began')
    # interpolation
    for ivar in needed_vars:
        before = ds_mod[ivar] # co2 before interpolation
        for irecord in nonzero_record:
            var_before  = before[irecord,:].values - background*1e3 # a co2 profile, subtract background (ppm to ppb)
            pres_before = np.log(profile_press_mod[irecord].values) # log space
            pres_after  = np.log(profile_press_sat[irecord])
            # linear interpolation on log space    
            interpolated[ivar][irecord,:] = np.interp(x  = pres_after, 
                                                      xp = np.flip(pres_before), # increasing order
                                                      fp = np.flip(var_before))
        print(ivar, 'done')
    
    output_file = mod_dir + 'interpolated_' + str1_mod + str(year) + month_string(month) + '.nc'
    interpolated.to_netcdf(output_file)
    
    print('created ' + 'interpolated_' + str1_mod + str(year) + month_string(month) + '.nc')


# In[54]:


#
def delta_Y(mod_dir, sat_dir, year, month, str1_mod, str1_sat, mode):
    mod_file = str1_mod + str(year) + month_string(month) + '.nc'
    sat_file = str1_sat + str(year) + month_string(month) + '.nc'
    ds_mod = xr.open_dataset(mod_dir + mod_file)
    ds_sat = xr.open_dataset(sat_dir + sat_file)
    
    # variables needed 
    varnames = list(ds_mod.data_vars.keys())  # list of variable name
    tag_species = [i for i in varnames if i.startswith('SpeciesConc_CO2Tag')] # species var names
    
    # mask data: predefined region, land vs. ocean, latitudinal bands
    mask_directory = '/geos/u73/msadiq/GEOS-Chem/analysis/inversion/data/'
    mask_name     = 'flatten_mask.nc'
    ds_mask = xr.open_dataset(mask_directory + mask_name)

    avg_kern = ds_sat['xco2_averaging_kernel']
    co2_pr   = ds_sat['co2_profile_apriori']
    pres_wgt = ds_sat['pressure_weight']
    op_mode  = ds_sat['operation_mode'] # 0=Nadir, 1=Glint
    mode_mask= (op_mode-1)*-1    # mask to mask out glint, 1=Nadir, 0=Glint

    # new dataset to store all model xco2
    lat = ds_mod['lat']
    delta_y_mod = lat.to_dataset()
    delta_y_mod['lon'] = ds_mod['lon']
    delta_y_mod['date'] = ds_mod['date']
    
    # loop over tag species to compute delta y0
    for ivar in tag_species:   
        co2_model = ds_mod[ivar]*1e-3 # unit: ppbv to ppm
        #xco2_tmp = pres_wgt * (1 - avg_kern) * co2_pr + pres_wgt * avg_kern * co2_model
        xco2_tmp =  pres_wgt * avg_kern * co2_model
        xco2 = xco2_tmp.sum(dim = 'levels') # sum along vertical axis, unit: ppm
        xco2_land = xco2 * ds_mask['land'][0:len(xco2)]  # exclude ocean
        if mode == 'N':
            xco2_mode = xco2_land * mode_mask                # select observation mode
        else:
            xco2_mode = xco2_land 

        tmp_name = 'X_' + ivar

        delta_y_mod[tmp_name] = xco2_mode.copy()
        #print(tmp_name + ' done!')
    
    if mode == 'N':
        output_file = mod_dir + 'delta_y0_model_N_' + str(month) + '.nc'
    else:
        output_file = mod_dir + 'delta_y0_model_' + str(month) + '.nc'
    
    delta_y_mod.to_netcdf(output_file)
    print('created ' + 'delta_y0_model_N_' + str(month) + '.nc')


# In[62]:


#imonth = 3
#if __name__ == '__main__':
#    year = 2018
#    for imonth in range(1,13):
#        case_name = 'CO2-TC67-2018-' + month_string(imonth)
#        mod_dir = '/geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs/' + case_name + '/nd51/'

#        delta_Y(mod_dir = mod_dir, sat_dir = '/geos/u73/msadiq/satellite/oco-2/', 
#             year = year, month = imonth, 
#             str1_mod = 'interpolated_2d_ts_satellite.', 
#             str1_sat = '2d_OCO2_extract_',
#             mode = 'N')
    #mod_dir = '/geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs/CO2-2018-03/nd51/'
    #fname = 'delta_y0_model_N_3.nc'
    #fname = '2d_ts_satellite.201803.nc'
    #ds = xr.open_dataset(mod_dir + fname)
    #ds['SpeciesConc_CO2Tag10'].plot()


# In[59]:


def r_rmse(obs_series, model_series):
    R = ma.corrcoef(ma.masked_invalid(obs_series), ma.masked_invalid(model_series))
    x = obs_series[~np.isnan(obs_series)]
    y = model_series[~np.isnan(model_series)]
    rmse = np.sqrt(((y - x) ** 2).mean())
    format_R = float("{0:.2f}".format(R[0,1]))
    format_rmse = float("{0:.2f}".format(rmse))
    return format_R, format_rmse


# In[60]:


def plot_xco2_diff(diff_before, diff_after, lat_slice, lon_slice, imonth, mod_dir):
    '''Produce plots of XCO2 differences
    inputs (1d arrays): 
        diff in XCO2, before and after
        lat_slice, lon_slice, lat and lon for each data point
        imonth, month number, for plot title
    outputs: plots
        '''
    nobs = len(diff_before)
    # Creat target dataarray and dataset
    lat_res = 2.    # horizontal resolution of lat and lon you would want
    lon_res = 2.5
    lat = np.linspace(-90, 90, int(180/lat_res + 1)) # grid
    lon = np.linspace(-180, 177.5, int(360/lon_res))
    diff_1 = xr.DataArray(data = np.nan, 
                        dims = ('lat', 'lon'), 
                        coords = {'lat': lat,
                                'lon': lon},
                        name = 'diff')
    diff_2 = xr.DataArray(data = np.nan, 
                        dims = ('lat', 'lon'), 
                        coords = {'lat': lat,
                                'lon': lon},
                        name = 'diff')
    # map 1d data onto dataarray
    for i in range(nobs):
        lat = int((lat_slice[i].values + 90)/2) # lat index
        lon = int((lon_slice[i].values + 180)/2.5)
        diff_1[lat, lon] = -diff_before[i]
        diff_2[lat, lon] = -diff_after[i]
    print('y diff before:',"{:.2f}".format(diff_1.mean().values))
    print('y diff after:',"{:.2f}".format(diff_2.mean().values))

    # figure 1, distribution
    fig, axes = plt.subplots(1, 2, 
                             figsize=[14, 6], 
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             gridspec_kw={'hspace': 0.2, 'wspace': 0})
    # before
    diff_1.plot(ax=axes[0], vmax = 4, add_labels = False, cbar_kwargs={'shrink': 0.5})
    axes[0].set_title(short_name_of_month(imonth) + ' XCO2: a prior - OCO2', loc='left')
    axes[0].set_title('ppm', loc = 'right')
    axes[0].coastlines()
    axes[0].gridlines(linestyle = '--')
    # after
    diff_2.plot(ax=axes[1], vmax = 4, add_labels = False, cbar_kwargs={'shrink': 0.5})
    axes[1].set_title(short_name_of_month(imonth) + ' XCO2: a posterior - OCO2', loc='left')
    axes[1].set_title('ppm', loc = 'right')
    axes[1].coastlines()
    axes[1].gridlines(linestyle = '--')

    fig.savefig(mod_dir + 'bio_results_map_diff_' + str(imonth) + '.png', dpi=300)
    
    obs_series = xco2_oco_slice.values
    model_series = xco2_mod_slice.values
    format_R1, format_rmse1 = r_rmse(obs_series, model_series)
    #R = ma.corrcoef(ma.masked_invalid(obs_series), ma.masked_invalid(model_series))
    #x = obs_series[~np.isnan(obs_series)]
    #y = model_series[~np.isnan(model_series)]
    #rmse = np.sqrt(((y - x) ** 2).mean())
    #format_R1 = float("{0:.2f}".format(R[0,1]))
    #format_rmse1 = float("{0:.2f}".format(rmse))
    print('R1 is:', format_R1, ' RMSE1 is: ', format_rmse1)
    
    obs_series = xco2_oco_slice.values
    model_series = xco2_oco_slice.values - diff_after
    format_R2, format_rmse2 = r_rmse(obs_series, model_series)
    print('R2 is:', format_R2, ' RMSE2 is: ', format_rmse2)
    
    # figure 2, scatter plot
    fig = plt.figure(figsize=[5,5])
    plt.plot([300,450],[300,450], c='black')
    plt.scatter(xco2_oco_slice, xco2_mod_slice, s=0.7, label = 'A prior')
    plt.scatter(xco2_oco_slice, xco2_oco_slice - diff_after, s=0.7, label = 'A posterior')
    plt.ylim(top   = 420,bottom = 395)
    plt.xlim(right = 420,left   = 395)
    plt.text(x=405, y=397.5, s='R1: ' + str(format_R1) + ' RMSE1: ' + str(format_rmse1), size = 12)
    plt.text(x=405, y=396, s='R2: ' + str(format_R2) + ' RMSE2: ' + str(format_rmse2), size = 12)
    plt.title(name_month + ' XCO2 (ppm)')
    plt.ylabel('GEOS-Chem')
    plt.xlabel('OCO2')
    plt.legend(markerscale = 4)
    
    fig.savefig(mod_dir + 'bio_results_scatter_diff_' + str(imonth) + '.png', dpi=300)
    


# In[61]:


# split masks (dataarray) into separate dataarrays, as one dataset, 
# with 1 for masked area, 0 for elsewhere
def split_mask(input_mask):
    target = input_mask.copy()
    target[:,:] = 0.
    ds_masks = target.to_dataset(name = 'MASK1')
    nm_masks = int(input_mask.max().values) # number of masks
    
    for count in range(1,nm_masks+1): # + ocean
        target = input_mask.copy()
        target[:,:] = 0.
        mask_TF = input_mask == count # True or False map
        target = mask_TF.where(True)
        name_tmp = 'MASK' + str(count)
        ds_masks[name_tmp] = target
    name_tmp = 'MASK' + str(count+1) # last mask for all nan values, ocean in giorgi mask
    ds_masks[name_tmp] = np.isnan(input_mask).where(True)
    return ds_masks


# In[ ]:


# 
def compare_XCO2(mod_dir, sat_dir, year, month, str1_mod, str1_sat, mode):
    mod_file = str1_mod + str(year) + month_string(month) + '.nc'
    sat_file = str1_sat + str(year) + month_string(month) + '.nc'
    ds_mod = xr.open_dataset(mod_dir + mod_file)
    ds_sat = xr.open_dataset(sat_dir + sat_file)
    
    # mask data: predefined region, land vs. ocean, latitudinal bands
    mask_directory = '/geos/u73/msadiq/GEOS-Chem/analysis/inversion/data/'
    mask_name     = 'flatten_mask.nc'
    ds_mask = xr.open_dataset(mask_directory + mask_name)

    avg_kern = ds_sat['xco2_averaging_kernel']
    co2_pr   = ds_sat['co2_profile_apriori']
    pres_wgt = ds_sat['pressure_weight']
    op_mode  = ds_sat['operation_mode'] # 0=Nadir, 1=Glint
    mode_mask= (op_mode-1)*-1    # mask to mask out glint, 1=Nadir, 0=Glint

    # model simulated CO2 concentration
    co2_model = ds_mod['SpeciesConc_CO2']*1e-3 # unit: ppbv to ppmv 
    co2_profile = pres_wgt * (1 - avg_kern) * co2_pr + pres_wgt * avg_kern * co2_model
    # sum up to get column CO2
    xco2_model = co2_profile.sum(dim = 'levels')      # sum along vertical axis, unit: ppbv to ppm

    xco2_model_mode = xco2_model * mode_mask # extract desired mode of observation: Nadir
    xco2_oco2_mode = ds_sat['xco2'] * mode_mask

    obs_series = xco2_oco2_mode.values
    model_series = xco2_model_mode.values
    #obs_series[obs_series==0] = 'nan'
    #model_series[model_series==0] = 'nan'
    
    format_R, format_rmse = r_rmse(obs_series, model_series)
    #R = ma.corrcoef(ma.masked_invalid(obs_series), ma.masked_invalid(model_series))
    #x = obs_series[~np.isnan(obs_series)]
    #y = model_series[~np.isnan(model_series)]
    #rmse = np.sqrt(((y - x) ** 2).mean())
    #format_R = float("{0:.2f}".format(R[0,1]))
    #format_rmse = float("{0:.2f}".format(rmse))
    print('R is:', format_R, ' RMSE is: ', format_rmse)

    xco2_model_land = xco2_model_mode * ds_mask['land'][0:len(xco2_model)]
    xco2_model_land[xco2_model_land==0] = 'nan'
    xco2_oco2_land = xco2_oco2_mode * ds_mask['land'][0:len(xco2_model)]
    xco2_oco2_land[xco2_oco2_land==0] = 'nan'

    fig = plt.figure(figsize=[5,5])
    name_month = short_name_of_month(month)
    for region in ['high_lat', 'mid_lat', 'low_lat']:
        xco2_model_mask = xco2_model_land * ds_mask[region][0:len(xco2_model_land)]
        xco2_model_mask[xco2_model_mask==0] = 'nan'
        xco2_oco2_mask = xco2_oco2_land * ds_mask[region][0:len(xco2_model_land)]
        xco2_oco2_mask[xco2_oco2_mask==0] = 'nan'

        plt.scatter(xco2_oco2_mask, xco2_model_mask, s=0.7, label = region)
        plt.plot([300,450],[300,450], c='r')
        plt.ylim(top   = 420,bottom = 395)
        plt.xlim(right = 420,left   = 395)
        plt.title(name_month + ' XCO2 (ppm)')
        plt.ylabel('GEOS-Chem')
        plt.xlabel('OCO2')
        plt.legend(markerscale = 4)

        plt.text(x=410, y=399, s='R: ' + str(format_R), size = 12)
        plt.text(x=410, y=398, s='RMSE: ' + str(format_rmse), size = 12)
        fig.savefig(mod_dir + '/mod_vs_obs_XCO2_latitudinal_'+ mode + '_' + name_month + '.png', dpi=300)


    diff = xco2_oco2_land - xco2_model_land   # diff to calculate a posteriori
    new_data = diff.to_dataset(name = 'diff')
    new_data['xco2_oco2'] = xco2_oco2_land
    new_data['xco2_model'] = xco2_model_land
    new_data['xco2_error'] = ds_sat['xco2_uncertainty']
    new_data['lat'] = ds_mod['lat']
    new_data['lon'] = ds_mod['lon']
    new_data['date'] = ds_mod['date']
    new_data.to_netcdf(mod_dir + 'XCO2_mod_and_oco2_' + mode + '_' + name_month + '.nc')

    # Creat target dataarray and dataset
    lat_res = 2    # horizontal resolution of lat and lon you would want
    lon_res = 2.5
    lat = np.linspace(-90, 90, int(180/lat_res + 1)) # grid
    lon = np.linspace(-180, 177.5, int(360/lon_res))
    days = len(diff)/(len(lat)*len(lon))

    var_3d = xr.DataArray(data = np.nan, 
                          dims = ('days', 'lat', 'lon'), 
                          coords = {'days': range(int(days)),
                                    'lat': lat,
                                    'lon': lon},
                          name = 'diff')
    var_3d = var_3d.astype(dtype='float32')

    diff2 = xco2_model_land - xco2_oco2_land # diff to map onto global map
    var_3d.values = diff2.values.reshape((int(days),len(lat),len(lon)))
    
    var_2d = var_3d.mean(dim='days')
    # plot after mapping
    fig = plt.figure(figsize=[8, 8])
    proj=ccrs.PlateCarree()
    ax = plt.subplot(111, projection=proj)
    # 
    var_2d.plot(ax=ax, vmax = 3, add_labels = False, cbar_kwargs={'shrink': 0.4})
    ax.set_title(name_month + ' XCO2: a posterior - OCO2', loc = 'left');
    ax.set_title('ppm', loc = 'right')
    ax.coastlines()
    ax.gridlines(linestyle = '--')
    

    fig.savefig(mod_dir + 'diff_map_' + name_month + '.png', dpi=300)

    ds_output = var_3d.to_dataset()
    var_3d.values = xco2_model_land.values.reshape((int(days),len(lat),len(lon)))
    ds_output['mod'] = var_3d.copy()
    var_3d.values = xco2_oco2_land.values.reshape((int(days),len(lat),len(lon)))
    ds_output['obs'] = var_3d.copy()
    ds_output.to_netcdf(mod_dir + 'XCO2_diff_' + str(month) + '.nc')


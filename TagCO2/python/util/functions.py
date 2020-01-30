"""
    Some useful functions
        Authors: Mehliyar Sadiq
        History: 2019-11-15, added area_latlon
                 2020-01-29, added create_masks
                 2020-01-29, split_masks
                 2020-01-29, mask_plus_times
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # plots
import cartopy.crs as ccrs      # map projections
from pathlib import Path        # check if directory exists, if not, create

import regionmask               # package to create masks, https://regionmask.readthedocs.io/en/stable/

# Function #1
# Approximate the area of a spatial grid square from the latitudes and longitudes of the diagonal vertices
def area_latlon(lat1, lon1, lat2, lon2):
    """
    This function calculates the area (in km^2) of a spatial grid square, given the latitudes and longitudes of the two diagonal vertices of the grid square.
    lat/lon is in angle; lat: [-90:90]; lon:[-180:180].
    lat1/lon1 and lat2/lon2 are thus the diagonal vertices of the square grid.
    """
    lat1 = lat1/180*np.pi
    lat2 = lat2/180*np.pi
    lon1 = lon1/180*np.pi
    lon2 = lon2/180*np.pi
    A = np.absolute(6371.009**2*(np.sin(lat2)-np.sin(lat1))*(lon2-lon1))
    return A

# Function #2
# Create corresponding region/country masks for an input file
def create_masks(input_file, 
                 input_file_type='dataarray', 
                 var_name='emi_co2', 
                 mask_type='giorgi', 
                 output_format='xarray', 
                 output_path='.', 
                 figure_flag = False, 
                 figure_path='.',
                 debug=False):
    """
    This function creates mask files according to the resolution of input file.
    	input_file could be NetCDF file or xarray dataarray, needs to be specified in input_file_type, 
	input_file_type could be 'netcdf' or 'dataarray'
        var_name is the name of the variable in the file, preferably a two dimensional variable (lonxlat)
	mask_type could be 'giorgi', 'srex', 'countries_110', 'countries_50', 'us_states_50', 'us_states_10'
    	output_format could be 'netcdf' or 'xarray'
	output_path is directory to store output, if output_format is netcdf
        figure_flag, if True, make mask plots and save to figure_path
        figure_path is directory to store the output figures
    """
    if(input_file_type=='netcdf'):
        ds = xr.open_dataset(input_file) # read in netcdf file
        dr = ds[var_name]
        #print(dr)
    elif(input_file_type=='dataarray'):
        dr=input_file
    
    # get resolution of input data
    resolution_input_lon = dr.coords['lon'][3] - dr.coords['lon'][2]
    resolution_input_lat = dr.coords['lat'][3] - dr.coords['lat'][2]
    resolution_input = str(resolution_input_lon.values) + 'x' + str(resolution_input_lat.values)
    
    if(mask_type in ['giorgi', 'srex']): 
        mask = getattr(regionmask.defined_regions,mask_type).mask(dr)
    elif(mask_type in ['countries_110', 'countries_50', 'us_states_50', 'us_states_10']):
        mask = getattr(regionmask.defined_regions.natural_earth,mask_type).mask(dr)
    else:
        print('mask_type not supported, stopped')
        return

    if(figure_flag == True):
        # make mask plots
        fig = plt.figure(figsize=[8, 4])

        proj=ccrs.PlateCarree()
        ax = plt.subplot(111, projection=proj)
    
        low = mask.min()
        high = mask.max()
        levels = np.arange(low - 0.5, high + 1)

        mask.plot(ax=ax, transform=ccrs.PlateCarree(), levels=levels, cmap='tab20', vmax = 21.5, cbar_kwargs={'shrink': 0.8,})
        ax.set_title(mask_type + ' ' + str(int(high.values)) + " masks " + resolution_input)
        ax.coastlines();
        # save this plot to output path
        fig.savefig(figure_path + '/mask_' + mask_type + '_' + resolution_input + '.png', dpi=300)
        plt.close()
        if(debug == True):
            print('Mask figure is saved to: ' + figure_path)
    
    print('finished creating masks for ' + mask_type)
    
    if(output_format == 'xarray'):
        return mask
    elif(output_format == 'netcdf'):
        mask.to_netcdf(output_path+ '/mask_' + mask_type + '_' + resolution_input +  '.nc')
        if(debug == True):
            print(mask_type + ' netcdf output file is saved to: ' + output_path)
    
    return 

# for testint
#for mask_type in ['giorgi', 'srex', 'countries_110', 'countries_50', 'us_states_50', 'us_states_10']:
#    create_masks(input_file='ODIAC_2015_1x1.nc', input_file_type='netcdf', mask_type=mask_type, output_format='netcdf', output_path='./nc', figure_path='./plots')

def split_masks(input_file, 
                input_file_type='dataarray',
                output_format='netcdf', 
                output_path='./MASKS', 
                figure_flag = True, 
                figure_path='./figures',
                debug=False):
    """
    This function splits a mask created by above function into separate mask files
        input_file could be NetCDF file or xarray dataarray, needs to be specified in input_file_type, 
        input_file_type could be 'netcdf' or 'dataarray'
        output_format could be 'netcdf' or 'xarray'
        output_path is directory to store output (if format is netcdf)
        figure_flag, if True, make figures for each mask file and save to figure_path
        figure_path is directory to store the output figures
    """
    
    nm_masks = input_file.max().values # number of masks
    
    # open a target file that you want to map your masks to
    fname = '~/Desktop/co2/data/emission/GC/SE_Asia_mask.generic.1x1.nc'
    ds = xr.open_dataset(fname)
    
    lon_len = len(ds.lon)
    lat_len = len(ds.lat)
    time_len = len(ds.time)
    
    resolution_output_lon = ds.coords['lon'][3] - ds.coords['lon'][2]
    resolution_output_lat = ds.coords['lat'][3] - ds.coords['lat'][2]
    resolution_output = str(resolution_output_lon.values) + 'x' + str(resolution_output_lat.values)
    
    for count in np.arange(nm_masks)+1:
        
        target = xr.Dataset({"MASK": (("time", "lat", "lon"), np.zeros(lon_len*lat_len*time_len).reshape(time_len,lat_len,lon_len))},
                            coords=ds.coords)
        target = target.astype(dtype='float32')
        
        mask_TF = input_file == count # True or False map
        target['MASK'][0,:,:] = mask_TF.where(True)
        
        target['MASK'].attrs = ds['MASK'].attrs
        target['lon'].attrs = ds['lon'].attrs
        target['lat'].attrs = ds['lat'].attrs
        target['time'].attrs = ds['time'].attrs
        
        target.attrs = ds.attrs
        target.attrs['comment'] = 'region masks, by m.sadiq 2020'
        target.attrs['history'] = 'made by m.sadiq 2020'
        
        if(output_format == 'netcdf'):
            name_file = 'MASK' + str(count.astype(int)) + '_' + resolution_output + '.nc'
            target.to_netcdf(output_path + '/' + name_file)
            if(debug == True):
                print(name_file +' NetCDf file is saved to: ' + output_path)
            
        if(figure_flag == True):
            # plot the last mask
            fig = plt.figure(figsize=[8, 4])

            proj=ccrs.PlateCarree()
            ax = plt.subplot(111, projection=proj)

            target['MASK'].plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={'shrink': 0.8})
            ax.set_title("MASK " + str(count.astype(int)))
            ax.set_label(' ')
            ax.coastlines()
            
            fig.savefig(figure_path + '/mask_' + str(count.astype(int)) + '_' + resolution_output + '.png', dpi=300)
            plt.close()
            
            if(debug == True):
                print(name_file + ' figure is saved to: ' + figure_path)
    
    # last mask for places not belonging to any mask above
    target = xr.Dataset({"MASK": (("time", "lat", "lon"), np.zeros(lon_len*lat_len*time_len).reshape(time_len,lat_len,lon_len))},
                            coords=ds.coords)
    target = target.astype(dtype='float32')
    
    mask_nan = np.isnan(input_file)
    target['MASK'][0,:,:] = mask_nan.where(True)
        
    target['MASK'].attrs = ds['MASK'].attrs
    target['lon'].attrs = ds['lon'].attrs
    target['lat'].attrs = ds['lat'].attrs
    target['time'].attrs = ds['time'].attrs
        
    target.attrs = ds.attrs
    target.attrs['comment'] = 'region masks, by m.sadiq 2020'
    target.attrs['history'] = 'made by m.sadiq 2020'
        
    if(output_format == 'netcdf'):
        name_file = 'MASK' + str(nm_masks.astype(int)+1) + '_' + resolution_output + '.nc'
        target.to_netcdf(output_path + '/' + name_file)
        if(debug == True):
            print(name_file +' NetCDf file is saved to: ' + output_path)
            
    if(figure_flag == True):
        # plot the last mask
        fig = plt.figure(figsize=[8, 4])

        proj=ccrs.PlateCarree()
        ax = plt.subplot(111, projection=proj)

        target['MASK'].plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={'shrink': 0.8})
        ax.set_title("MASK " + str(nm_masks.astype(int)+1))
        ax.set_label(' ')
        ax.coastlines()
        plt.close()
            
        fig.savefig(figure_path + '/mask_' + str(nm_masks.astype(int)+1) + '_' + resolution_output + '.png', dpi=300)
        
        if(debug == True): 
            print(name_file + ' figure is saved to: ' + figure_path)

    print('finished spliting masks')
    
    return


def mask_plus_times(input_file,
                    year = '2017',
                    month = '2017-04',
                    output_name = 'MASK.nc',
                    output_path='./MASKS/'):
    """
    This function adds time dimension to masks made up above function
        input_file is NetCDF file
        year is the year of the emission, daily
        month is the month of mask to have, rest is all zero
        output_name is the name of the output file
        output_path is directory to store output (if format is netcdf)
    """
    
    ds = xr.open_dataset(input_file) # open the input_file

    year_sim = np.array(year,dtype='datetime64[Y]')
    month_emission = np.array(month,dtype='datetime64[M]')

    # days in year 
    days = np.arange(year_sim, year_sim + 1, dtype='datetime64[D]')
    days_in_ns = np.array(days, dtype='M8[ns]') # unit conversion to match source

    lon_len = len(ds.lon) # length of longitude
    lat_len = len(ds.lat) # length of latitude
    time_len = len(days_in_ns) # length of time dimension, days in a year

    emi_range = np.arange(month_emission,month_emission+1,dtype='datetime64[D]') # range of time to have the mask, a month
    
    target = xr.Dataset({"MASK": (("time", "lat", "lon"), 
                              np.zeros(lon_len*lat_len*time_len).reshape(time_len,lat_len,lon_len))},)
    target.coords['lon'] = ds.coords['lon']
    target.coords['lat'] = ds.coords['lat']
    target.coords['time'] = days_in_ns
    target = target.astype(dtype='float32')
    
    target['MASK'].loc[emi_range,:,:] = ds['MASK'].values # only assign mask to the month of concern
    
    Path(output_path).mkdir(parents=True, exist_ok=True)

    target.to_netcdf(output_path + output_name)

    return

#### test
mask_plus_times(input_file='~/Desktop/co2/data/emission/MASKS/nc/giorgi/MASK11_1.0x1.0.nc',
                    year = '2017',
                    month = '2017-04',
                    output_path='~/Desktop/co2/data/emission/MASKS/nc/giorgi/')

    

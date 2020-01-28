"""
    Some useful functions
        Authors: Mehliyar Sadiq
        History: 2019-11-15, added area_latlon
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # plots
import cartopy.crs as ccrs      # map projections

import regionmask               # package to create masks

def create_masks(input_file, input_file_type='xarray', var_name='emi_co2', mask_type='giorgi', output_format='xarray', output_path='.'):
    """
    This function creates mask files according to the resolution of input file.
    	input_file could be NetCDF file, xarray or numpy array, needs to be specified in input_file_type, 
	input_file_type could be 'netcdf', 'xarray' or 'numpy'
	mask_type could be 'giorgi', 'srex', 'countries_110', 'countries_50', 'us_states_50', 'us_states_10'
    	output_format could be 'netcdf', 'xarray' or 'numpy'
	output_path is directory to store output, if output_format is netcdf
    """
    if(input_file_type=='netcdf'):
        ds = xr.open_dataset(input_file) # read in netcdf file
        dr = ds[var_name]
        #print(dr)
    
    if(mask_type=='giorgi'):
        mask = regionmask.defined_regions.giorgi.mask(dr)
    elif(mask_type=='srex'):
        mask = regionmask.defined_regions.srex.mask(dr)
    elif(mask_type=='countries_110'):
        mask = regionmask.defined_regions.natural_earth.countries_110.mask(dr)
    elif(mask_type=='countries_50'):
        mask = regionmask.defined_regions.natural_earth.countries_50.mask(dr)
    elif(mask_type=='us_states_50'):
        mask = regionmask.defined_regions.natural_earth.us_states_50.mask(dr)
    elif(mask_type=='us_states_10'):
        mask = regionmask.defined_regions.natural_earth.us_states_10.mask(dr)
    else:
        print('mask_type not supported, stopped')
        return

    # plot the mask
    fig = plt.figure(figsize=[8, 4])

    proj=ccrs.PlateCarree()
    ax = plt.subplot(111, projection=proj)
    
    low = mask.min()
    high = mask.max()
    levels = np.arange(low - 0.5, high + 1)

    mask.plot(ax=ax, transform=ccrs.PlateCarree(), levels=levels, cmap='tab20', vmax = 21.5, cbar_kwargs={'shrink': 0.8,})
    ax.set_title(mask_type + " masks")
    ax.coastlines();

    fig.savefig(output_path + '/masks_' + mask_type + '.png', dpi=300)
    
    return 

create_masks(input_file='ODIAC_2015_1x1.nc', input_file_type='netcdf', mask_type='giorgi', output_format='xarray', output_path='./plots')

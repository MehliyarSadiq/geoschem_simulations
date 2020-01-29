"""
    Some useful functions
        Authors: Mehliyar Sadiq
        History: 2019-11-15, added area_latlon
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # plots
import cartopy.crs as ccrs      # map projections

import regionmask               # package to create masks, https://regionmask.readthedocs.io/en/stable/

def create_masks(input_file, input_file_type='xarray', var_name='emi_co2', mask_type='giorgi', output_format='xarray', output_path='.', plot_path='.'):
    """
    This function creates mask files according to the resolution of input file.
    	input_file could be NetCDF file or xarray, needs to be specified in input_file_type, 
	input_file_type could be 'netcdf' or 'xarray'
        var_name is the name of the variable in the file, preferably a two dimensional variable (lonxlat)
	mask_type could be 'giorgi', 'srex', 'countries_110', 'countries_50', 'us_states_50', 'us_states_10'
    	output_format could be 'netcdf' or 'xarray'
	output_path is directory to store output, if output_format is netcdf
        plot_path is directory to store the output figures
    """
    if(input_file_type=='netcdf'):
        ds = xr.open_dataset(input_file) # read in netcdf file
        dr = ds[var_name]
        #print(dr)
    elif(input_file_type=='xarray'):
        dr=input_file[var_name]
    
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
    fig.savefig(plot_path + '/mask_' + mask_type + '_' + resolution_input + '.png', dpi=300)
    
    if(output_format == 'xarray'):
        return mask
    elif(output_format == 'netcdf'):
        mask.to_netcdf(output_path+ '/mask_' + mask_type + '_' + resolution_input +  '.nc')
    return 

for mask_type in ['giorgi', 'srex', 'countries_110', 'countries_50', 'us_states_50', 'us_states_10']:
    create_masks(input_file='ODIAC_2015_1x1.nc', input_file_type='netcdf', mask_type=mask_type, output_format='netcdf', plot_path='./plots', output_path='./nc')

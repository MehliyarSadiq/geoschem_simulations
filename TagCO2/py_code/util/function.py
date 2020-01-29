"""
    Some useful functions
        Authors: Mehliyar Sadiq
        History: 2019-11-15, added area_latlon
                 2020-01-29, added create_masks
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt # plots
import cartopy.crs as ccrs      # map projections

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
def create_masks(input_file, input_file_type='xarray', var_name='emi_co2', mask_type='giorgi', output_format='xarray', output_path='.', figure_flag = False, figure_path='.'):
    """
    This function creates mask files according to the resolution of input file.
    	input_file could be NetCDF file or xarray, needs to be specified in input_file_type, 
	input_file_type could be 'netcdf' or 'xarray'
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
        print('Mask figure is saved to: ' + figure_path)
    
    if(output_format == 'xarray'):
        return mask
    elif(output_format == 'netcdf'):
        mask.to_netcdf(output_path+ '/mask_' + mask_type + '_' + resolution_input +  '.nc')
        print(mask_type + ' netcdf output file is saved to: ' + output_path)
    return 

# for testint
#for mask_type in ['giorgi', 'srex', 'countries_110', 'countries_50', 'us_states_50', 'us_states_10']:
#    create_masks(input_file='ODIAC_2015_1x1.nc', input_file_type='netcdf', mask_type=mask_type, output_format='netcdf', output_path='./nc', figure_path='./plots')







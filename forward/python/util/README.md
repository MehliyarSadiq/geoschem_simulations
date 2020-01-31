# Modules I need often

## 1. gamap_colormap.py

Copied from GEOS-Chem tutorial: [Chapter01_NetCDF_xarray](https://github.com/MehliyarSadiq/tutorials/blob/master/GEOSChem-python/Chapter01_NetCDF_xarray.ipynb)

It is a python script containing IDL/gamap's default WhGrYlRd color scheme. Looks much better than default matplotlib.plot() color scheme, when plotting absolute value of a variable.

**Usage:** 

    from gamap_colormap import WhGrYlRd
    dr.plot(cmap=WhGrYlRd); # dr is a 2-d dataarray you want to plot

## 2. functions.py

- **area_latlon**

Rewritten from TGABI tools: get_geo.R: [http://www.cuhk.edu.hk/sci/essc/tgabi/tools.html](http://www.cuhk.edu.hk/sci/essc/tgabi/tools.html)

Approximate the area of a spatial grid square from the latitudes and longitudes of the diagonal vertices.

**Usage:**

    from util.functions import area_latlon
    area_latlon(lat1, lon1, lat2, lon2):
        """
        This function calculates the area (in km^2) of a spatial grid square, given the latitudes and longitudes of the two diagonal vertices of the grid square.
        lat/lon is in angle; lat: [-90:90]; lon:[-180:180].
        lat1/lon1 and lat2/lon2 are thus the diagonal vertices of the square grid.
        """

- **create_masks**

This function creates mask files according to the resolution of input file. Six different kinds of masks could be made: 'giorgi', 'srex', 'countries_110', 'countries_50', 'us_states_50', 'us_states_10', using [regionmask](http://regionmask.py) module: [https://regionmask.readthedocs.io/en/stable/](https://regionmask.readthedocs.io/en/stable/)

**Usage:**

    create_masks(input_file, 
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
    			debug=False, dose not print progress
        """

- **split_masks**

Split one map of masks into separate mask files

**Usage:**

    split_masks(input_file, 
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

- **mask_plus_times**

Add time dimension to splitted masks

**Usage:**

    mask_plus_times(input_file,
                        year = '2017',
                        month = '2017-04',
                        output_name = 'MASK.nc',
                        output_path='./MASKS/'):
        """
        This function adds time dimension to masks made with above function
            input_file is a NetCDF file
            year is the year of the emission
            month is the month of mask to have, rest is all zero
            output_name is the name of the output file
            output_path is directory to store output (if format is netcdf)
        """

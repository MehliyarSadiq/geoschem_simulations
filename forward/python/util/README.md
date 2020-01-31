# Modules I need often

### 1. gamap_colormap.py

Copied from GEOS-Chem tutorial: [Chapter01__NetCDF_xarray.ipynb]()

[Chapter01_NetCDF_xarray](https://github.com/MehliyarSadiq/tutorials/blob/master/GEOSChem-python/Chapter01_NetCDF_xarray.ipynb)

It is a python script containing IDL/gamap's default WhGrYlRd color scheme. Looks much better than default matplotlib.plot() color scheme, when plotting absolute value of a variable.

**Usage:** 

    from gamap_colormap import WhGrYlRd
    dr.plot(cmap=WhGrYlRd); # dr is a 2-d dataarray you want to plot

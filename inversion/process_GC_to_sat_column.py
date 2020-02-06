#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:56:18 2020

Convolve GEOS-Chem outputs to TROPOMI pressure levels

Calculate equivalent column, applying the TROPOMI averaging kernel

@author: mlunt
"""
import numpy as np
import test_flib as flb
import time as run_time
import datetime
import xarray
import glob
import matplotlib.pyplot as plt
import pandas as pd
import re
import GC_output_mod as process
from scipy import interpolate

def get_pressure_weight(pres, p_surf):
    """
    pres = 4d array of shape (ntime,nlat,nlon,nlev)
    p_surf = 3d array of shape (ntime,nlat,nlon)
    Outputs:
        pres_wgt - weights fror each pressure level in forming the columns
    """
    pres_wgt = pres.copy()*0.
    
    p_surf2 = p_surf[:,:,:,np.newaxis]
    
    pdiff = pres[:,:,:,1:] - pres[:,:,:,:-1] # length n-1
    log_pdiff = np.log(pres[:,:,:,1:]/pres[:,:,:,:-1])
    
    # Lowest level
    pres_wgt[:,:,:,0] = np.abs(-1.*pres[:,:,:,0] + (pdiff[:,:,:,0]/log_pdiff[:,:,:,0]))* 1./p_surf
    
    # Highest level
    pres_wgt[:,:,:,-1] = np.abs(pres[:,:,:,-1] - (pdiff[:,:,:,-1]/log_pdiff[:,:,:,-1]))* 1./p_surf
    
    # Middle levels
    pres_wgt[:,:,:,1:-1] = np.abs( (pdiff[:,:,:,1:]/log_pdiff[:,:,:,1:]) - (pdiff[:,:,:,:-1]/log_pdiff[:,:,:,:-1]))*1./p_surf2
    
    return pres_wgt

run_name = "test_run_CH4" 


satellite = "TROPOMI"
#satellite = None

domain = "SSA"

file_type='nc'

#varnames_short = ["CH4",     "CH4BC",   "CH4_OIL", "CH4_GAS", 
#                  "CH4_COL", "CH4_LIV", "CH4_LDF", "CH4_WST",
#                  "CH4_RIC", "CH4_OTA", "CH4_BBN", "CH4_WTL",
#                  "CH4_SEE", "CH4_LAK", "CH4_TER", "CH4_E1"]  

varnames_short = ["CH4"]  
        
ch4_mod_new={}
ch4_mod={}
xch4_column={}


varnames_long=[]
for varname in varnames_short:
        varnames_long.append("IJ_AVG_S_" + varname)

############################################################################
#%%

# Define directories for each run
        
run_dir  = "/path/to/run_dir/" + run_name + "/" 

#output_dir  = run_dir + "OutputDir/" 
output_dir  = "/path/to/outputs/" + run_name + "/nc_files/"
        
column_out_dir = output_dir + "sat_columns/"  
fname_out= column_out_dir + "XCH4_Model_" + satellite + ".nc" 

if satellite == "GOSAT":
    obs_dir = "/path/to/GOSAt/obs/"
    obs_str = "GOSAT-L2-ACOS_GCPR-CH4-avg-CA-merra2-"
elif satellite == "TROPOMI":
    
    obs_dir = "/path/to TROPOMI/obs/"
    obs_str = "TROPOMI_XCH4_025x03125_daily_" + domain + "_"
    
else:
    obs_dir = None

tracer_file = run_dir + "tracerinfo.dat"
diag_file = run_dir + "diaginfo.dat"

#gc_file_str = "ts_satellite."
gc_file_str = "sat_output_"
#%%
startt = run_time.time()
files = process.filenames(output_dir, gc_file_str, file_type=file_type)

if len(files) < 1:
    raise ValueError("No files found in " + output_dir )

m = re.search(gc_file_str +'(.+?)' + '.' + file_type, files[0])
if m:
    start_date = m.group(1)
else:
    raise ValueError("START_DATE: GC output file string does not fit expected format")
    
m2 = re.search(gc_file_str +'(.+?)' + '.' + file_type, files[-1])
if m:
    end_date = m2.group(1)
else:
    raise ValueError("END_DATE: GC output file string does not fit expected format")


ds_gc = process.read_GC_out_files(output_dir, gc_file_str, start_date=start_date, 
                                  end_date=end_date, tracer_file=tracer_file,
                                  diag_file=diag_file, irregular_vars=False,
                                  file_type='nc')

# In this instance ds_gc is a list of datasets
# Each one has a time dimension applied
gc_time = ds_gc.time

time_mod_int = []
for ti_day in gc_time:
    time_int = int(pd.to_datetime(ti_day.values).strftime('%Y%m%d'))
    time_mod_int.append(time_int)

lat_mod = ds_gc.lat.values
lon_mod = ds_gc.lon.values



#%% 
# Read satellite data
if satellite in(["GOSAT", "TROPOMI"]):
    latmin = lat_mod.min()
    latmax = lat_mod.max()
    lonmin = lon_mod.min()
    lonmax = lon_mod.max()
    
    ds_obs = process.read_satellite(obs_dir, obs_str, start_date=start_date, 
                                    end_date=end_date, region=[latmin,latmax,lonmin,lonmax])
    
if satellite == "TROPOMI":
    
    ds_temp = ds_gc.copy()
    ds_gc = ds_temp.reindex_like(ds_obs, method="nearest")
    
#%%        
# Calculate pressure weight matrix
# Can probably do as array operation in Python rather than using f2py.

#ds_gc  = ds_gc.transpose("time", "lat", "lon", "lev")
pres_mod = ds_gc["PEDGE_S_PSURF"].values

#ch4_mod = ds_gc.IJ_AVG_S_CH4
nlev_mod = len(ds_gc.lev)
# Pressure weighting function from Connor et al 2008 https://doi.org/10.1029/2006JD008336
#xch4_field = {}
if satellite == None:
    p_surf = pres_mod[:,:,:,0]
    pres_wgt = get_pressure_weight(pres_mod, p_surf)
    
    for varname in varnames_short:
        xch4_column[varname] = (pres_wgt * ds_gc["IJ_AVG_S_" + varname]).sum(dim="lev")
        
elif satellite == "TROPOMI":
    
    ntime = len(ds_obs.time)
    nlev_sat=12
    dp_sat = ds_obs.pressure_interval
    psurf_sat = ds_obs.surface_pressure
    
    for ti in range(ntime):
        if np.max(psurf_sat[ti,:,:]) > 5000:
            dp_sat[ti,:,:] = dp_sat[ti,:,:]/100.
            psurf_sat[ti,:,:] = psurf_sat[ti,:,:]/100.
    

    # Satellite files go from top of atmsphere to bottom
    # GC goes from bottom to top.
    # Need to be consistent!!!!!!!
    air_subcolumn = ds_obs.dry_air_subcolumn[:,:,:,::-1]
    ch4_ap = ds_obs.ch4_profile_apriori[:,:,:,::-1]   # mol/m2
    averaging_kernel = ds_obs.ch4_averaging_kernel[:,:,:,::-1]
    
    pedge_mod = pres_mod.copy()
    pcent_mod = pres_mod.copy()*0.

    # Calculate pressure at centre of model grid cells
    pcent_mod[:,:,:,:-1] = (pedge_mod[:,:,:,:-1] + pedge_mod[:,:,:,1:])/2.
    pcent_mod[:,:,:,-1] = pedge_mod[:,:,:,-1]/2.
    
    
    #ntime = len(ds_obs.time)
    nlat= len(ds_obs.lat)
    nlon=len(ds_obs.lon)
    intmap = np.zeros((ntime, nlat, nlon,nlev_mod,nlev_sat))
    intmap2 = np.zeros((ntime, nlat, nlon,nlev_mod,nlev_sat))
    success=0
    
    ############################################################
    # I don't really want to loop through all obs but maybe it's the only way. 
    # Do this in f2py if so. 
    p_sat_cent = np.zeros((ntime,nlat,nlon,nlev_sat))
    #p_sat_edge = np.zeros((ntime,nlat,nlon,nlev_sat))
    
    # Need to loop through all species
    # Separate interpolation function for each one.
    
#            ch4_mod_new={}
#            ch4_mod={}
#            xch4_column={}
 
    
    for varname in varnames_short:
    
        ch4_mod_new[varname] = np.zeros((ntime,nlat,nlon,nlev_sat))
        ch4_mod[varname]  = ds_gc["IJ_AVG_S_" + varname]
    
    for ti in range(ntime):
        for lati in range(nlat):
            for loni in range(nlon):
                
                p_sat_cent_i = (psurf_sat.values[ti,lati,loni]  - dp_sat.values[ti,lati,loni]*(np.arange(12)+0.5))
                #p_sat_edge_i = (psurf_sat.values[ti,lati,loni]  - dp_sat.values[ti,lati,loni]*(np.arange(12)))
                # Are these edges or centres?
                
                if np.isfinite(p_sat_cent_i[0]):
                
                    p_mod_i = pcent_mod[ti,lati,loni,:]
                    p_sat_cent[ti,lati,loni,:]  = p_sat_cent_i.copy()
                
                    for varname in varnames_short:
                        
                        ch4_mod_i = ch4_mod[varname].values[ti,lati,loni,:]
                        
                        fill_values = (ch4_mod_i[-1], ch4_mod_i[0])
                        interp_func = interpolate.interp1d(p_mod_i, ch4_mod_i, bounds_error=False, fill_value=fill_values)
                    
                        # No longer calls error if SAT max pressure > GC max pressure - fills with maximum CH4 value.
                        ch4_mod_new[varname][ti,lati,loni,:] = interp_func(p_sat_cent_i)
                   
                    #p_sat_edge[ti,lati,loni,:]  = p_sat_edge_i.copy()
                else:
                    for varname in varnames_short:
                        ch4_mod_new[varname][ti,lati,loni,:] = np.nan
    
   
    # Can this be sped up? The interp function might make that impossible. 
    # Think I can skip straight to CH4 new as don't need the interpolation map itself.
    ##############################################################
    
    # Now need to apply the pressure weighting as above, but using satellite pressure levels instead. 
    # Create a function to calculate weight so this doesn't get repeated continuously.
    
    total_air_column = np.sum(air_subcolumn.values,axis=3)
    p_wgt_sat = air_subcolumn.values/total_air_column[:,:,:,None]
    

    # Convert ch4_ap subcolumn to mol/mol dry air
    ch4_ap_subcolumn = ch4_ap.values/air_subcolumn.values
    xch4_ap = np.sum((p_wgt_sat * ch4_ap_subcolumn), axis=3)*1.e9
    
    
    #xch4_obs_no_ap = xch4_obs - 
    I_minus_A_xap = xch4_ap - np.sum((p_wgt_sat*averaging_kernel.values *ch4_ap_subcolumn*1.e9),axis=3)
    
    for varname in varnames_short:
        if varname in (["CH4", "CH4BC"]):
            xch4_column[varname] = xch4_ap + np.sum(p_wgt_sat*averaging_kernel.values * 
                                   (ch4_mod_new[varname] - ch4_ap_subcolumn*1.e9), axis=3)
            
        else:
            xch4_column[varname] = np.sum(p_wgt_sat*averaging_kernel.values*ch4_mod_new[varname], axis=3)
    

#%% 

    xch4_obs = ds_obs.XCH4
    xch4_precision = ds_obs.XCH4_precision
    dates_out = ds_obs.time

    ds_out = xarray.Dataset({"XCH4_obs": (["time", "lat", "lon"],xch4_obs),
                             "IminusA_XCH4_ap": (["time", "lat", "lon"],I_minus_A_xap),
                             "XCH4_ap": (["time", "lat", "lon"],xch4_ap),
                             "XCH4_precision": (["time", "lat", "lon"],xch4_precision),
                             "surface_pressure":(["time", "lat", "lon"],psurf_sat),
                             "pressure_interval":(["time", "lat", "lon"],dp_sat),
                             "ch4_profile_apriori":(["time", "lat", "lon", "layer"],ch4_ap),
                             "ch4_averaging_kernel":(["time", "lat", "lon","layer"],averaging_kernel),
                             "dry_air_subcolumn":(["time", "lat", "lon","layer"],air_subcolumn),
                             },
                             coords={"lon":lon_mod, "lat": lat_mod, "time":dates_out})
    ds_out.attrs["Comments"]='XCH4 mean at 0.25x0.3125'
    
    
    for varname in varnames_short:
        ds_out[varname]  = (["time", "lat", "lon"],xch4_column[varname])
    
    ds_out.attrs["qa_threshold"] = 0.5
    #ds_out.attrs["Comments"]='Monthly mean XCH4 at 0.1x0.1'
    
    ds_out["XCH4_obs"].attrs = xch4_obs.attrs
    ds_out["XCH4_precision"].attrs = xch4_precision.attrs
    ds_out["surface_pressure"].attrs = psurf_sat.attrs
    ds_out["pressure_interval"].attrs = dp_sat.attrs
    ds_out["ch4_profile_apriori"].attrs = ch4_ap.attrs
    ds_out["ch4_averaging_kernel"].attrs = averaging_kernel.attrs
    ds_out["dry_air_subcolumn"].attrs = air_subcolumn.attrs
    
    ds_out["surface_pressure"].attrs["units"] = "hPa"
    ds_out["pressure_interval"].attrs["units"] = "hPa"
    
    for key in list(ds_out.keys()):
            ds_out[key].encoding['zlib'] = True                    
    ds_out.to_netcdf(path=fname_out, mode='w')
#!/usr/bin/env python
# coding: utf-8

# todo:
# 1, update matrix -> fluxes (NetCDF)

from tools import *
from bio_flux_functions_TransCom import *
import sys

args = sys.argv

# inversion configurations
year        = 2016
assim_month = int(args[1]) # first month to assimilate observation, start from 3, Mar
lag_window  = 3    # months
nx          = 67   # number of tagged tracers, same as number of masks
mode        = 'N' # OCO-2 Nadir only
mod_err     = 1. #1.5 # model/transport error, unit: ppm
land_prior_err  = 0.7 #0.5 # intial prior error, unitless, multiply biospheric flux for actual prior error
ocean_prior_err = 0.3
snow_prior_err = 0.1
tag_case_name = 'CO2-TC67-'

# full CO2 simulation directory
top_dir = '/geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs/' + str(year) + '/'
mod_dir = top_dir + 'CO2-' + str(year) + '/nd51/'

## todo: download as .py and load it correctly
## done, begining of this script

# get monthly flux of each region

input_dir = '/geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/'
f1 = input_dir + 'NEE.Liang.2016.nc'
f2 = input_dir + 'Taka2009_CO2_Monthly.nc'
   
monthly_bio_flux = regional_monthly_sink(f1, f2)
#monthly_bio_flux = regional_monthly_bio(f1)


# In[ ]:


# prior error
del_X_f = np.diag(np.repeat(land_prior_err,nx))
for i in range(55, nx-1): del_X_f[i,i] = ocean_prior_err # ocean and low emission regions
del_X_f[nx-1,nx-1] = snow_prior_err # snow region
#print('initial: ', np.diag(del_X_f))

# prior error, read in from previous assimilation (if any)
if assim_month > 3:
    del_X_lag = np.loadtxt(mod_dir + "prior_error_" + short_name_of_month(assim_month-1) + ".txt")
else:
    del_X_lag = np.zeros((nx*lag_window,nx*lag_window))
    
    for i in range(lag_window): 
        del_X_lag[i*nx:(i+1)*nx,i*nx:(i+1)*nx] = del_X_f #[nx*nlag, nx*nlag], diag,
        
print('actual:', np.diag(del_X_lag))
del_X_updated = del_X_lag


# In[ ]:


update_array = np.zeros(12)
# start of inversion calculation:
for assim_month in range(3,13):
    # 1, read in from full CO2 simulations
    name_month = short_name_of_month(assim_month) # Jan, Feb, Mar, ...
    xco2_file = 'XCO2_mod_and_oco2_N_' + name_month + '.nc'
    ds_xco2 = xr.open_dataset(mod_dir + xco2_file)
    xco2_oco2  = ds_xco2['xco2_oco2']  # XCO2 from OCO-2
    xco2_model = ds_xco2['xco2_model'] # XCO2 from model
    xco2_error = ds_xco2['xco2_error'] # measurement error from OCO-2
    diff = ds_xco2['xco2_oco2'] - ds_xco2['xco2_model']
    # reduce the size of above arrays and matrices, from ~400k to <3k
    x = xco2_oco2.copy()
    x = x[np.logical_not(np.isnan(x))]
    ind = x.record.values   # index for slicing
    nobs = len(ind) # number of obs in this month
    print('number of observation this month: ', nobs)
    # get slices of arrays
    diff_slice = diff[ind].values            # [nobs], 1-3k per month
    lat_slice = ds_xco2.lat[ind]
    lon_slice = ds_xco2.lon[ind]
    xco2_mod_slice = xco2_model[ind]
    xco2_oco_slice = xco2_oco2[ind]
    xco2_error_slice = xco2_error[ind].values 
    # observation error
    obs_error = np.zeros((nobs,nobs))  # [nobs,nobs], diagonally store obs error
    for idiag in range(nobs):
        obs_error[idiag, idiag] = (0.5*xco2_error_slice[idiag])**2 + mod_err**2
        # measurment error from oco2
        # model error and representation error = 2.5 for land

    # delta y0
    del_Y = np.empty((nobs,nx*lag_window))   # [nobs, nx*lag_window]
    del_Y[:] = np.nan

    # 2, read in tag runs
    for itmp in range(lag_window):
        ilag_month = assim_month - lag_window + itmp + 1
        ens_dir = top_dir + tag_case_name + str(year) + '-' + month_string(ilag_month) + '/nd51/'
        delta_y0_file = 'delta_y0_model_N_' + str(assim_month) + '.nc'
        # open datasets
        ds_delta_y0 = xr.open_dataset(ens_dir + delta_y0_file)
        varnames = list(ds_delta_y0.data_vars.keys())  # list of variable name
        needed_vars = [i for i in varnames if i.startswith('X_SpeciesConc_CO2Tag')] # species var names
        # read variables
        for itag, ivar in enumerate(needed_vars):
            del_Y[:,itag+nx*itmp] = ds_delta_y0[ivar][ind].values # column order: assim_month - 2, assim_month - 1, assim_month

    del_Y0 = del_Y
    del_X_lag = del_X_updated
    del_Y = np.dot(del_Y, del_X_lag)
    
    # calculation of posterior
    del_Y_tran = np.matrix.transpose(del_Y)     # del_y transpose [nx*lag_window,nobs]
    first      = np.matmul(del_Y,del_Y_tran)    # del_y dot del_y_tran [nobs,nobs]
    second     = np.linalg.inv(first+obs_error) # (Y*Yt + R)^-1 [nobs,nobs], dominated by second term, issue???
    third      = np.matmul(del_Y_tran,second)   # Yt*(Y*Yt + R)^-1 [nx*lag_window,nobs]
    k_e        = np.matmul(del_X_lag,third)     # kalman gain, k_e = X_f*Yt*(Y*Yt + R)^-1 [nx*lag_window,nobs]
    adjust     = np.matmul(k_e, diff_slice)     # adjustment to prior, k_e*(yobs - ym)  [nx*lag_window]

    update = adjust * monthly_bio_flux[assim_month-lag_window:assim_month].flatten()
    # monthly updates
    for i in range(lag_window):
        print(short_name_of_month(assim_month - lag_window + i + 1), 'adjustment: {:.2f}'.format(sum(update[i*nx:(i+1)*nx])))
        update_array[assim_month - lag_window + i] +=  sum(update[i*nx:(i+1)*nx])
    print(update_array)
    # update Y matrix
    diff_new = diff_slice - np.dot(del_Y0,adjust) # [nobs], ppm
    plot_xco2_diff(diff_slice, diff_new, lat_slice, lon_slice, assim_month)

    # transformation matrix to update prior
    fourth = np.identity(nx*lag_window) - np.matmul(third, del_Y)
    transform_mat = sp.sqrtm(fourth)
    del_X_lag = np.matmul(del_X_lag, transform_mat)

    # update del_X_lag and use it in next assimilation
    del_X_updated = np.zeros((nx*lag_window,nx*lag_window))
    del_X_updated[:(lag_window-1)*nx, :(lag_window-1)*nx] = del_X_lag[nx:,nx:]
    del_X_updated[(lag_window-1)*nx:, (lag_window-1)*nx:] = del_X_f


# In[ ]:


# all fluxes used in simulations
fname = '/geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2019-12/monthly_emission.nc'
ds_monthly = xr.open_dataset(fname)
dr_monthly = ds_monthly['ff'] # just for making a mask

# create masks for this resolution
dr = dr_monthly[0,:,:]
# create masks for input file
mask = create_masks(input_file=dr, 
                    input_file_type='dataarray', 
                    var_name='', 
                    mask_type='giorgi', 
                    output_format='xarray', 
                    output_path='/geos/u73/msadiq/GEOS-Chem/MASKS/nc/giorgi/', 
                    figure_flag = False, 
                    figure_path='/home/msadiq/Desktop/co2/data/emission/MASKS/figures/')

# split masks into separate dataarrays, with 1 for masked area, 0 for elsewhere
target = dr.copy()
target[:,:] = 0.
ds_masks = target.to_dataset(name = 'MASK1')

nm_masks = int(mask.max().values) # number of masks
for count in range(1,nm_masks+1): # + ocean
    target = dr.copy()
    target[:,:] = 0.
        
    mask_TF = mask == count # True or False map
    target = mask_TF.where(True)
    name_tmp = 'MASK' + str(count)
    ds_masks[name_tmp] = target

# monthly net fluxes for 22 tagged regions
monthly_net_flux_casa = np.zeros((12,nm_masks+1))
dr_monthly = ds_monthly['ff'] + ds_monthly['fire'] + ds_monthly['ocean'] + ds_monthly['nte'] + ds_monthly['casa_bio']

for i in range(12):
    for j in range(nm_masks):
        mask_name = 'MASK' + str(j+1)
        tmp = dr_monthly[i,:,:] * ds_masks[mask_name]
        monthly_net_flux_casa[i,j] = tmp.sum().values*1e-12*12/44
# last one
for i in range(12):
    tmp = dr_monthly[i,:,:]
    sum_tmp = tmp.sum().values*1e-12*12/44
    masked_sum = sum(monthly_net_flux_casa[i,:])
    monthly_net_flux_casa[i,-1] = sum_tmp - masked_sum

net_flux_2018 = np.sum(monthly_net_flux_casa, axis = 1) # global net flux
print('My annual total net flux w/ CASA (PgC/year): ', np.sum(net_flux_2018))




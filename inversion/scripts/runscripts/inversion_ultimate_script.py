#!/usr/bin/env python
# coding: utf-8
import sys
from tools import *
from bio_flux_functions_TransCom import *

print('-'*50)
print('load packages and functions')

# inversion configurations
year        = 2016
assim_month = int(sys.argv[1])      # first month to assimilate observation, start from 3, Mar
lag_window  = 3      # months
nx          = 67     # number of tagged tracers, same as number of masks
mode        = 'N'    # OCO-2 Nadir only
mod_err     = 1.5    #1.5 # model/transport error, unit: ppm
land_prior_err  = 0.5 #0.5 # intial prior error, unitless, multiply biospheric flux for actual prior error
ocean_prior_err = 0.3
snow_prior_err  = 0.1

tag_case_name = 'CO2-TC67-' + str(year) + '-'

# full CO2 simulation directory
top_dir = '/scratch/local/msadiq/rundir/' + str(year) + '/'
#top_dir = '/geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs/' + str(year) + '/'

name_month = short_name_of_month(assim_month) # Jan, Feb, Mar, ...
# read in data from current directory, analyze
curr_dir = top_dir + 'CO2-' + str(year) + '/nd51/' + name_month + '/'
# output in next directory
next_dir = top_dir + 'CO2-' + str(year) + '/nd51/' + short_name_of_month(assim_month+1) + '/'

print('-'*50)
print('started inversion calculation for ' + str(year) + ' ' + short_name_of_month(assim_month))
print('current working directory:')
print(curr_dir)

### this cell could do all post-processing for a case, over a few months, or a month

print('-'*50)
print('post-processing outputs from previous run')

for imonth in range(assim_month-lag_window+1,assim_month+1):

    bpch_to_nc_mass(data_dir = curr_dir,
                        year = year,
                        month = imonth)

    combine_daily_to_one(data_dir = curr_dir,
                         year = year,
                         month = imonth,
                         str1 = 'ts_satellite.',
                         str2 = '.nc')

    flatten_4d_to_2d(data_dir = curr_dir,
                         year = year,
                         month = imonth,
                         str1 = 'ts_satellite.',
                         str2 = '.nc')
    
    interpolate_model_to_satellite(mod_dir = curr_dir, 
                                       sat_dir = '/geos/u73/msadiq/satellite/oco-2/', 
                                       year = year, 
                                       month = imonth, 
                                       str1_mod = '2d_ts_satellite.', 
                                       str1_sat = '2d_OCO2_extract_')
    compare_XCO2(mod_dir = curr_dir, 
                 sat_dir = '/geos/u73/msadiq/satellite/oco-2/', 
                 year = year, 
                 month = imonth, 
                 str1_mod = 'interpolated_2d_ts_satellite.', 
                 str1_sat = '2d_OCO2_extract_',
                 mode = 'N')
    


print('-'*50)
print('calculating sinks used in previous run')

input_dir = '/geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/'

if year == 2016 and assim_month == 3:
    f1 = input_dir + 'NEE.Liang.2016.nc'
    f2 = input_dir + 'Taka2009_CO2_Monthly.nc'
else:
    f1 = input_dir + 'NEE.Liang.2016.updated.' + str(year) + '.' + short_name_of_month(assim_month-1) + '.nc'
    f2 = input_dir + 'Taka2009_CO2_Monthly.updated.' + str(year) + '.' + short_name_of_month(assim_month-1) + '.nc'
    
monthly_bio_flux = regional_monthly_sink(f1, f2)


print('-'*50)
print('prior error matrix')

# initial prior error
del_X_f = np.diag(np.repeat(land_prior_err,nx))
for i in range(55, nx-1): 
    del_X_f[i,i] = ocean_prior_err # ocean and low emission regions
del_X_f[nx-1,nx-1] = snow_prior_err # snow region

# prior error, read in from previous assimilation (if any)
if year == 2016 and assim_month == 3:
    del_X_lag = np.zeros((nx*lag_window,nx*lag_window))
    for i in range(lag_window): 
        del_X_lag[i*nx:(i+1)*nx,i*nx:(i+1)*nx] = del_X_f #[nx*nlag, nx*nlag], diag,
else:        
    del_X_lag = np.loadtxt(curr_dir + "prior_error_" + short_name_of_month(assim_month) + ".txt")
    
print('diagonal:', np.diag(del_X_lag))


print('-'*50)
print('inversion calculation')

# start of inversion calculation:
# 1, read in from full CO2 simulations

xco2_file = 'XCO2_mod_and_oco2_N_' + name_month + '.nc'
ds_xco2 = xr.open_dataset(curr_dir + xco2_file)
xco2_oco2  = ds_xco2['xco2_oco2']  # XCO2 from OCO-2
xco2_model = ds_xco2['xco2_model'] # XCO2 from model
xco2_error = ds_xco2['xco2_error'] # measurement error from OCO-2
diff = ds_xco2['xco2_oco2'] - ds_xco2['xco2_model']

# reduce the size of above arrays and matrices, from ~400k to <3k
x = xco2_oco2.copy()
x = x[np.logical_not(np.isnan(x))]
ind = x.record.values   # index for slicing
nobs = len(ind) # number of obs in this month
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
    ens_dir = top_dir + tag_case_name + month_string(ilag_month) + '/nd51/'
    delta_y0_file = 'delta_y0_model_N_' + str(assim_month) + '.nc'
    # open datasets
    ds_delta_y0 = xr.open_dataset(ens_dir + delta_y0_file)
    varnames = list(ds_delta_y0.data_vars.keys())  # list of variable name
    needed_vars = [i for i in varnames if i.startswith('X_SpeciesConc_CO2Tag')] # species var names
    # read variables
    for itag, ivar in enumerate(needed_vars):
        del_Y[:,itag+nx*itmp] = ds_delta_y0[ivar][ind].values # column order: assim_month - 2, assim_month - 1, assim_month

del_Y0 = del_Y
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

# update Y matrix
diff_new = diff_slice - np.dot(del_Y0,adjust) # [nobs], ppm
# weird... following line doesn't work
plot_xco2_diff(name_month, xco2_mod_slice, xco2_oco_slice, diff_slice, diff_new, lat_slice, lon_slice, assim_month, curr_dir)

# transformation matrix to update prior
fourth = np.identity(nx*lag_window) - np.matmul(third, del_Y)
transform_mat = sp.sqrtm(fourth)
del_X_lag = np.matmul(del_X_lag, transform_mat)

# update del_X_lag and use it in next assimilation
del_X_updated = np.zeros((nx*lag_window,nx*lag_window))
del_X_updated[:(lag_window-1)*nx, :(lag_window-1)*nx] = del_X_lag[nx:,nx:]
del_X_updated[(lag_window-1)*nx:, (lag_window-1)*nx:] = del_X_f


# cost function
# J = (x-x_b)_transpose * B * (x-x_b) + (y-h(x))_transpose * R * (y-h(x))
first_half = 0.
second_half = np.dot(np.dot(np.matrix.transpose(diff_slice),np.linalg.inv(obs_error)), diff_slice)
J = first_half + second_half
print('before inversion cost of x:', first_half)
print('before inversion cost of y:', second_half)

x_diff = update # x - prior
B = del_X_lag # prior error covariance matrix 
first_half = np.dot(np.dot(np.matrix.transpose(x_diff),np.linalg.inv(B)), x_diff)
second_half = np.dot(np.dot(np.matrix.transpose(diff_new),np.linalg.inv(obs_error)), diff_new)
J = first_half + second_half
print('after inversion cost of x:', first_half)
print('after inversion cost of y:', second_half)


# use this to update biospheric flux in total CO2 simulations
scale_array = 1+adjust
scale_lag = scale_array.reshape(lag_window,nx) # 2d

#scale_lag = np.flip(scale_lag, axis = 0) # flip?

for i in range(lag_window): 
    #plt.plot(scale_lag[i], label = short_name_of_month(assim_month - lag_window + i + 1))
    print(short_name_of_month(assim_month - lag_window + i + 1), ' scaling factors:')
    print(scale_lag[i])
#plt.legend()


print('-'*50)
print('update fluxes')

# monthly fluxes of sinks
dr_sink = monthly_sink(f1, f2) # (12,144,91)

# use TransCom masks
fname = '/geos/u73/msadiq/GEOS-Chem/MASKS/MASK_TC67_1x1.nc'
ds_tc = xr.open_dataset(fname)
dr_mask = ds_tc['transcom_regions']
    
dr_flux_mask = dr_sink[0,:,:].copy()
dr_flux_mask[:,:] = 0.
for ilon, lon in enumerate(dr_flux_mask['lon'].values):
    for ilat, lat in enumerate(dr_flux_mask['lat'].values):
        dr_flux_mask[ilat,ilon] = dr_mask.sel(lat = lat, lon = lon, method="nearest")
        
#dr_flux_mask.plot()


# In[ ]:


# make scaling maps for each month
scale_map = dr_sink[0:lag_window,:,:].copy()
scale_map[:,:,:] = 0. # scale map, [lag_window, lat, lon]

for imonth in range(lag_window):
    for ilat in range(len(scale_map.lat)):
        for ilon in range(len(scale_map.lon)):
            if np.isnan(dr_flux_mask[ilat,ilon].values): scale_map[imonth,ilat,ilon] = scale_lag[imonth,-1] # last one, ocean
            else: 
                mask_nm = int(dr_flux_mask[ilat,ilon].values)
                scale_map[imonth,ilat,ilon] = scale_lag[imonth,mask_nm-1] # mask number from 1 to 21


# In[ ]:


# multiply this scale map to bio flux
# plot updates during lag window
fig, axes = plt.subplots(3, 2, 
                         figsize=[14, 14], 
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         gridspec_kw={'hspace': 0.2, 'wspace': 0})

# monthly biospheric flux
for i in range(lag_window):
    
    dr_sink[assim_month - lag_window + i,:,:].plot(ax=axes[i,0], 
                    vmax = 5e9,
                    vmin = -5e9,
                    cmap = 'RdBu_r',
                    add_labels = False,
                    cbar_kwargs={'shrink': 0.8})
    axes[i,0].set_title(long_name_of_month(assim_month - lag_window + i + 1), loc='left')
    axes[i,0].set_title('prior flux', loc='right')
    axes[i,0].coastlines()
    axes[i,0].gridlines(linestyle = '--')
    
# adjustments
for i in range(lag_window):
    adjust_abs = dr_sink[assim_month - lag_window + i,:,:]*(scale_map[i,:,:]-1)
    adjust_abs.plot(ax = axes[i,1],
                            vmax = 5e9,
                            vmin = -5e9,
                            cmap = 'RdBu_r',
                            add_labels = False,
                            cbar_kwargs={'shrink': 0.8})

    axes[i,1].set_title(long_name_of_month(assim_month - lag_window + i + 1), loc='left')
    axes[i,1].set_title('adjustment', loc='right')
    axes[i,1].coastlines()
    axes[i,1].gridlines(linestyle = '--')
    
fig.savefig(curr_dir  + 'flux_scaling.' + short_name_of_month(assim_month) + '.update.png', dpi = 300)


# In[ ]:


# update oceanic flux
print('-'*50)
print('update oceanic fluxes')

ds = xr.open_dataset(f2)
data = ds['CO2'].copy()
months = np.arange('2000-01', '2001-02', dtype='datetime64[M]')
for i in range(lag_window):
    imonth = assim_month - lag_window + i
    before = data.sel(time = months[imonth])
    after = data.sel(time = months[imonth])*scale_map[i,:,:]
    # assign this new subset into dataarray
    time_dim = before.time
    print('updated time slices', time_dim.values)
    data.loc[dict(time = time_dim)] = after
ds_output = data.to_dataset(name = 'CO2')
ds_output.attrs = ds.attrs

ds_output.to_netcdf(input_dir + 'Taka2009_CO2_Monthly.updated.' + str(year) + '.' + name_month + '.nc')


# In[ ]:


# update biospheric flux
print('-'*50)
print('update biospheric fluxes')

ds = xr.open_dataset(f1)
data = ds['CO2']

scale_map_regrid = data[0:lag_window].copy()
scale_map_regrid[:,:,:] = 0.
for i in range(lag_window):
    for ilon, lon in enumerate(scale_map_regrid['lon'].values):
        for ilat, lat in enumerate(scale_map_regrid['lat'].values):
            scale_map_regrid[i,ilat,ilon] = scale_map[i,:,:].sel(lat = lat, lon = lon, method="nearest")


# In[ ]:


months = np.arange('2016-01', '2017-02', dtype='datetime64[M]')

for i in range(lag_window):
    imonth = assim_month - lag_window + i
    print('updated months ', months[imonth])
    before = data.sel(time = slice(months[imonth], months[imonth+1]))
    after  = data.sel(time = slice(months[imonth], months[imonth+1]))*scale_map_regrid[i,:,:]
    # assign this new subset into dataarray
    time_dim = before.time
    #print(time_dim.values)
    data.loc[dict(time = time_dim)] = after
ds_output = data.to_dataset(name = 'CO2')
ds_output.attrs = ds.attrs

ds_output.to_netcdf(input_dir + 'NEE.Liang.2016.updated.' + str(year) + '.' + name_month + '.nc')


# In[ ]:


# updated prior error, save it in current and next month directory
np.savetxt(next_dir + "prior_error_" + short_name_of_month(assim_month+1) + ".txt", del_X_updated)


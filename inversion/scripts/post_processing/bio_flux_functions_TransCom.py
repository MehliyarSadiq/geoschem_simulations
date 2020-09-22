#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().run_line_magic('run', 'tools.ipynb')


# In[5]:


# functions to calculation bio fluxes


# In[12]:


def regional_monthly_bio(fname): 
    ''' 
    input: file name of flux
    output: 2d numpy array of monthly fluxes from each region
        we are using TransCom 67 regions here, so output dim is (12,67)
    '''
    ds_original = xr.open_dataset(fname)

    dr_unit = ds_original['CO2'].copy()
    dr_output = unit_convert2_hours(dr_unit, 1., 1.25, 3.) # unit converted to kg/3hours per grid cell

    # compute regional total
    data = ds_original['CO2'].copy()
    # use TransCom masks
    fname = '/geos/u73/msadiq/GEOS-Chem/MASKS/MASK_TC67_1x1.nc'
    ds_tc = xr.open_dataset(fname)
    dr_mask = ds_tc['transcom_regions']
    
    dr_flux_mask = ds_original['CO2'][0,:,:]
    dr_flux_mask[:,:] = 0.
    for ilon, lon in enumerate(dr_flux_mask['lon'].values):
        for ilat, lat in enumerate(dr_flux_mask['lat'].values):
            dr_flux_mask[ilat,ilon] = dr_mask.sel(lat = lat, lon = lon, method="nearest")
        
    ds_mask_splits = split_mask(input_mask = dr_flux_mask)
    ds_mask = ds_mask_splits.drop('MASK68')
    
    # sum up monthly from 3 hourly data
    dr_monthly = ds_original['CO2'][:12,:,:].copy()
    dr_monthly[:,:,:] = 0.
    # fix time dimension
    months = np.arange('2016-01', '2017-02', dtype='datetime64[M]')
    dr_monthly['time'] = months[:12]
    
    for i in range(12):
        before = dr_output.sel(time = slice(months[i], months[i+1]))
        dr_monthly[i,:,:] = before.sum(axis=0) # unit: kgCO2/month
    
    # monthly regional total flux
    nx = 67
    monthly_bio_flux = np.zeros((12,nx))

    target = ds_mask['MASK3'].copy()
    target[:,:] = 0.
    
    for i in range(12):
        for j in range(nx):
            mask_name = 'MASK' + str(j+1)
            tmp = dr_monthly[i,:,:] * ds_mask[mask_name]
            monthly_bio_flux[i,j] = tmp.sum().values*1e-12*12/44 # unit: PgC/month
    print('Biospheric annual flux (PgC/year): ', np.sum(monthly_bio_flux))
    return monthly_bio_flux


# In[15]:


#%%time
#fname = '/geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/NEE.Liang.2016.nc'
#monthly_bio_flux = regional_monthly_bio(fname)
#monthly_bio_flux[0]


# In[11]:


#fname = '/geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/monthly_emission.nc'
#ds_tmp = xr.open_dataset(fname)
#dr_sink = ds_tmp['ocean'] + ds_tmp['casa_bio']
#dr_sink[10,:,:].plot()


# In[17]:


def regional_monthly_sink(fname_bio, fname_ocean):
    #fname = '/geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/monthly_emission.nc'
    ds_ocean = xr.open_dataset(fname_ocean)
    varname = 'CO2'
    dr_ocean = ds_ocean[varname]
    lat_res = ds_ocean.attrs['Delta_Lat']
    lon_res = ds_ocean.attrs['Delta_Lon']
    dr_ocean_monthly = unit_convert2_dr_monthly(dr_ocean, 2000, lat_res, lon_res)
    target = dr_ocean.copy()
    target[:,:,:] = np.nan
    
    ds_bio  = xr.open_dataset(fname_bio)
    lat_res = ds_bio.attrs['Delta_Lat']
    lon_res = ds_bio.attrs['Delta_Lon']
    varname = 'CO2'
    dr_bio  = ds_bio[varname].copy()

    for ilat, lat in enumerate(dr_bio.lat):
        area_tmp = 1e6* area_latlon(lat1 = lat, 
                                    lon1 = dr_bio.lon[10].values, 
                                    lat2 = lat + lat_res, 
                                    lon2 = dr_bio.lon[10].values + lon_res)
        dr_bio[:,ilat,:] = dr_bio[:,ilat,:] * area_tmp * 3 * 3600 # unit: kgCO2/3hours
    print('annual total of bio fluxes:', dr_bio.sum().values*1e-12*12/44)

    # monthly total
    dr_bio_monthly = dr_bio[0:12,:,:].copy()
    dr_bio_monthly[:,:,:] = 0.
    dr_bio_monthly['time'] = dr_ocean['time']
    
    data = dr_bio.copy()
    months = np.arange('2016-01', '2017-02', dtype='datetime64[M]')

    for i in range(12):
        s = data.sel(time = slice(months[i], months[i+1]))
        if i == 11: st = s
        else:
            st = s[:-1,:,:]
        dr_bio_monthly[i,:,:] = st.sum(dim='time')
    print('after summing up monthly: ', dr_bio_monthly.sum().values*1e-12*12/44)

    # regrid
    target = regrid_fine_to_coarse_sum_dr_monthly(dr_bio_monthly, target, 2., 2.5)
    print('after regridding: ', target.sum().values*1e-12*12/44) # GtC/yr
    dr_land = target.copy()

    ########################
    dr_sink = dr_land + dr_ocean_monthly
    
    # compute regional total
    data = ds_ocean[varname].copy()
    
    # use TransCom masks
    fname = '/geos/u73/msadiq/GEOS-Chem/MASKS/MASK_TC67_1x1.nc'
    ds_tc = xr.open_dataset(fname)
    dr_mask = ds_tc['transcom_regions']
    
    dr_flux_mask = ds_ocean[varname][0,:,:]
    dr_flux_mask[:,:] = 0.
    for ilon, lon in enumerate(dr_flux_mask['lon'].values):
        for ilat, lat in enumerate(dr_flux_mask['lat'].values):
            dr_flux_mask[ilat,ilon] = dr_mask.sel(lat = lat, lon = lon, method="nearest")
        
    ds_mask_splits = split_mask(input_mask = dr_flux_mask)
    ds_mask = ds_mask_splits.drop('MASK68')
    
    # monthly regional total flux
    nx = 67
    monthly_bio_flux = np.zeros((12,nx))

    target = ds_mask['MASK3'].copy()
    target[:,:] = 0.
    
    for i in range(12):
        for j in range(nx):
            mask_name = 'MASK' + str(j+1)
            tmp = dr_sink[i,:,:] * ds_mask[mask_name]
            monthly_bio_flux[i,j] = tmp.sum().values*1e-12*12/44 # unit: PgC/month
    print('Total annual sink (PgC/year): ', np.sum(monthly_bio_flux))
    return monthly_bio_flux


# In[1]:


#%%time
#file_dir = '/geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/'
#f1 = file_dir + 'NEE.Liang.2016.nc'
#f2 = file_dir + 'Taka2009_CO2_Monthly.nc'
#monthly_bio_flux = regional_monthly_sink(f1, f2)


# In[2]:


#for i in range(67): plt.plot(monthly_bio_flux[:,i])


# In[ ]:





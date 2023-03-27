###############################################################
# Veronica Torralba
# May 2022
# Script to compute the apparent temperature at night
###############################################################
import ddsapi
import xarray as xr
import pandas as pd
import matplotlib
from scipy import signal
import os
import datetime
import time
import copy
import shutil
import sys
import netCDF4
from cdo import *
import requests
import numpy as np
import numpy.ma as ma
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
cdo = Cdo()
import math
from glob import glob
from netCDF4 import num2date, date2num

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

import calendar
import locale
from netCDF4 import num2date, date2num, Dataset
from numpy import dtype
from joblib import Parallel, delayed
import joblib
import matplotlib.pyplot as plt 

def fitting_daily(y):
    yfit=np.array(y)
    x=np.arange(0,24 ,1)
    z=np.polyfit(x, yfit,4)
    p = np.poly1d(z)
    out=p(x)
    return(out)

def nightvals(data,utc):
    data=np.array(data)
    ndays=data.shape[0]
    utcval=utc[0,0]
    tnight=np.zeros([ndays-1]) 
    if (math.isnan(utcval)==False):
        t1=int(23-utcval)
        t2=int(6-utcval)
        for iday,day in enumerate(range(1,ndays)):
            x=np.concatenate((data[day-1,t1:24],data[day,0:t2]))
            tnight[iday]=np.mean(x, dtype=np.float64)
    return(tnight)


# Region #
reg_name ='Europe'
min_lon = -15
min_lat = 25
max_lon = 60
max_lat = 70
#lats_bnds = np.array([min_lat,max_lat])
#lons_bnds = np.array([min_lon, max_lon])


mask=xr.open_dataset('/data/csp/vt17420/CLINT_proj/ERA5/ERA5_masks/ERA5_mask_UTC_Europe.nc')
ndays=109 #15 may to 31st of August
nlat=len(mask.coords['lat'])
nlon=len(mask.coords['lon'])
#years=np.linspace(1993,2017,24)
#nyears=len(years)
var='t2m'
if (var=='t2m'):
   lname='2m temperature at night (23-06)'
   pathout = '/data/csp/vt17420/CLINT_proj/ERA5/t2m_night/'
   lons_reg=np.arange(min_lon,max_lon+0.25,0.25)
   lats_reg=np.arange(min_lat,max_lat+0.25,0.25)
   nlon=len(lons_reg)
   nlat=len(lats_reg)
if (var=='atemp2m'):
   lname='Apparent temperature at night (23-06) (Russo et al. 2015)'
#   pathout = '/work/csp/vt17420/ERA5/daily/atemp2m_night/'
   pathout = '/data/csp/vt17420/CLINT_proj/ERA5/atemp2m_night/'
   lons_reg=np.arange(min_lon,max_lon+0.25,0.25)
   lats_reg=np.arange(min_lat,max_lat+0.25,0.25)
   nlon=len(lons_reg)
   nlat=len(lats_reg)

for year in (range(1950,1993)):
    # data_loading
    #------------------------------------
 # selection of the 15 days of may along with the number of years
    days_may=np.linspace(14, 31, num=18)
    obs1=xr.open_dataset('/data/csp/vt17420/CLINT_proj/ERA5/hourly/'+var+'/'+var+'_'+str(year)+'05.nc')
    obs1 = obs1.sel(latitude=slice(max_lat,min_lat),longitude=slice(min_lon,max_lon))
    obs=obs1.sel(time=obs1.time.time.dt.day.isin(days_may))
    # selection of the days in june, july and august
    for mon in range(6,9):
        obs1=xr.open_dataset('/data/csp/vt17420/CLINT_proj/ERA5/hourly/'+var+'/'+var+'_'+str(year)+str(mon).zfill(2)+'.nc')
        obs1 = obs1.sel(latitude=slice(max_lat,min_lat),longitude=slice(min_lon,max_lon))
        obs=xr.merge(([obs,obs1]))

    #splitting time and day dimension
    #------------------------------------
    if np.array_equal(mask.lat,obs.latitude)== False:
        obs=obs.isel(latitude=slice(None, None, -1))
    #tnight=np.zeros([ndays,nlat,nlon])
    obs_array= np.zeros([24,ndays+1,nlat,nlon])
    for hour in range(24):
        obs_array[hour,:,:,:]=obs.sel(time=obs.time.time.dt.hour==hour).to_array()[0,:,:,:]
    obs_array= xr.DataArray(obs_array,dims=['hour','day','lat','lon'])
    ds = xr.Dataset({'Obs': obs_array},
                       coords={'hour': np.linspace(0,23, 24), 
                       'day': np.linspace(0,109, 110), 
                       'lon': mask.coords['lon'], 
                       'lat': mask.coords['lat']})
   
 # adjusting daily cycle 
 #(this was done to be consistent with the predictions, but this might not be needed)
 #---------------------
    obs_fitted = xr.apply_ufunc(
        fitting_daily,
        ds,
        dask="parallelized",
        input_core_dims=[['hour']],
        vectorize=True,
        output_core_dims=[['hour']],
        output_dtypes=[np.float])

    # creating fake mask for the apply
    #--------------------------------
    mask_utc= np.zeros([ndays+1,nlat,nlon,24])
    for hour in range(24):
        for day in range(ndays+1):
            mask_utc[day,:,:,hour]=mask.to_array()[0,:,:]
    mask_utc= xr.DataArray(mask_utc,dims=['day','lat','lon','hour'])

    # computing the temperature at night
    #------------------------------------
    obs_night = xr.apply_ufunc(nightvals,
        obs_fitted,
        mask_utc,
        dask="parallelized",
        input_core_dims=[['day','hour'],['day','hour']],
        vectorize=True, join='outer',
        output_core_dims=[['days']],
        output_dtypes=[np.float])

    # adding nan based on the mask
    #------------------------------------
    tnight=np.array(obs_night.to_array()[0,:,:,:])
    tnight[tnight==0]=np.nan

    # netcdf file
    #------------------------------------
    tnight=tnight.transpose(2,0,1)
    tnight_aux=xr.DataArray(tnight,dims=['time','lat','lon'])

    dates=  obs.isel(time=(obs.time.dt.hour == 0)).time[1:110]

    ncfileout = pathout+var+'_night_'+str(year)+'_15MJJA.nc'
    ncout = Dataset(ncfileout, 'w', format='NETCDF4')
    ncout.createDimension('time', ndays)  # unlimited
    ncout.createDimension('latitude', nlat)
    ncout.createDimension('longitude', nlon)

    #create time axis
    time = ncout.createVariable('time', dtype('double').char, ('time',))
    time.long_name = 'time'
    time.units = 'hours since 1900-01-01 00:00:00.0'
    time.calendar = 'gregorian'
    time.axis = 'T'

    # create latitude axis
    latitude = ncout.createVariable('latitude', dtype('double').char, ('latitude'))
    latitude.standard_name = 'latitude'
    latitude.long_name = 'latitude'
    latitude.units = 'degrees_north'
    latitude.axis = 'Y'

    # create longitude axis
    longitude = ncout.createVariable('longitude', 
    dtype('double').char, ('longitude'))
    longitude.standard_name = 'longitude'
    longitude.long_name = 'longitude'
    longitude.units = 'degrees_east'
    longitude.axis = 'X'

    # create variable array
    vout = ncout.createVariable('atemp2m_night', dtype('double').char, ('time', 'latitude', 'longitude'))
    vout.long_name = lname
    vout.units = 'K'
    vout[:] = tnight_aux
    longitude[:] = mask.coords['lon']
    latitude[:] = mask.coords['lat']
    times2 = np.array(dates)
    times2 = times2.astype('datetime64[s]').tolist()   
    times2 = date2num(times2,units='hours since 1900-01-01 00:00:00.0',calendar='gregorian')
    time[:] = times2[:]
    # close files
    ncout.close()



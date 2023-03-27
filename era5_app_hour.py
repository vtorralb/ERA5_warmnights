################################################################################
# Veronica Torralba
# May 2022
# Script to compute the hourly apparent temperature from the hourly data
# ERA5 data is obtained from the CMCC-DDS
# The output is the hourly apparent temperature computed from: t2m, d2m and slp
# This script also employs the orography to estimate the surface pressure
#################################################################################

import ddsapi
import xarray as xr
import matplotlib
#matplotlib.use('Agg')
from scipy import signal
import os
import datetime
import time
import copy
import shutil
import sys
#sys.path.append('/mnt/nfs/d50/pastel/USERS/lecestres/analyse/')
import netCDF4
from cdo import *
import requests
import numpy as np
import numpy.ma as ma
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
#from function_read import *
cdo = Cdo()
import math
from glob import glob
from netCDF4 import num2date, date2num
#from function_read import *
#from data.set_output_name import *
#from forecast_veri import *
#from data.aux_and_plot import *
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy.stats import gaussian_kde
import calendar
import locale
#locale.setlocale( locale.LC_ALL , 'en_US' )
from netCDF4 import num2date, date2num, Dataset
from numpy import dtype
from joblib import Parallel, delayed
import joblib


#cdo.sellonlatbox('-15,60,25,70',input='/work/csp/vt17420/ERA5/orography.nc',output='/work/csp/vt17420/ERA5/orog.nc')
geop = xr.open_dataset('/data/csp/vt17420/CLINT_proj/ERA5/ERA5_masks/orog.nc')
elevation = geop[['z']].to_array()
elevat= elevation/9.807
del geop
del elevation

def psurface(temp,slp,elevation):
    g=9.807
    R=8.314
    mair=0.029
    lapse=6.5/1000
    tsurf = temp + (lapse * elevation)
    psurf = slp * math.exp((-elevation * g * mair)/(tsurf * R))
    return(psurf)

Psurf=np.vectorize(psurface)


def RH_ecmwf(x,y,p):
    a1 = 611.21
    a3_aux=22.587
    a3 = 17.502
    Rdry = 287.0597
    Rvap = 461.5250
    a4 = 32.19
    a4_aux= -0.7
    T0 = 273.16
    Tice = 250.16
    R = Rdry/Rvap
    E = a1 * math.exp(a3*((y - T0)/(y-a4)))
    Esat1 = a1 * math.exp(a3*((x - T0)/(x-a4)))
    Esat2 = a1 * math.exp(a3_aux*((x - T0)/(x-a4_aux)))
    alpha=0
    if (x<=Tice):
        alpha=0
    elif (Tice < x < T0):
        alpha=((x-Tice)/(T0-Tice))
    elif (x>= T0):
        alpha = 1
    Esat= (alpha* Esat1) + ((1-alpha) * Esat2)
    q = R * (E/(p -((1-R)*E)))
    RH = (p * (q*(1/R)))/(Esat * (1 +(q*((1/R)-1))))
    return(RH *100)

RH_ECMWF=np.vectorize(RH_ecmwf)

def convert_fahr_to_kelvin(temp):
    kelvin = ((temp - 32) * (5 / 9)) + 273.15
    return(kelvin)

def convert_kelvin_to_fahr(temp):
    fahr = ((9 / 5) * (temp - 273.15)) +32
    return(fahr)


def AT_function_russo(temp,RelH):
    # This is the implementation of the method described here:
    # https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    # temp should be included in Kelvins
    temp = convert_kelvin_to_fahr(temp)

    AT = 0.5 * (temp + 61.0 + ((temp-68.0)*1.2) + (RelH*0.094))

    if (AT >= 80):

        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -6.83783*(10 ** -3)
        c6 = -5.481713*(10 ** -2)
        c7 = 1.22874*(10 ** -3)
        c8 = 8.5282* (10 ** -4)
        c9 = -0.199*(10 ** -6)

        AT = c1 + (c2 * temp) + (c3 * RelH) + (c4* temp* RelH) +(c5 * (temp ** 2)) + (c6 * (RelH ** 2)) + (c7 * (temp ** 2) * RelH) + (c8 * (RelH ** 2) * temp)+(c9 * (temp ** 2) * (RelH ** 2))

        if (RelH < 13 and temp >= 80 and temp <= 112):
            ADJ1 = ((13-RelH)/4) * math.sqrt((17-abs(temp-95))/17)
            AT = AT- ADJ1

        if (RelH > 85 and temp >= 80 and temp <= 87):
            ADJ2 = ((RelH-85)/10) * ((87-temp)/5)
            AT = AT + ADJ2

    # The index defined in Russo, only        
    if AT > temp :
        out = AT
    else:
        out = temp

    output = convert_fahr_to_kelvin(out)
    return(output)

AT_fun_russo=np.vectorize(AT_function_russo)


for year in range(1993,2017):
    for month in range(5,9):
        c = ddsapi.Client(directPath=True)
        path1 =c.retrieve("era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "time": {
                        "hour": [
                            "00",
                            "01",
                            "02",
                            "03",
                            "04",
                            "05",
                            "06",
                            "07",
                            "08",
                            "09",
                            "10",
                            "11",
                            "12",
                            "13",
                            "14",
                            "15",
                            "16",
                            "17",
                            "18",
                            "19",
                            "20",
                            "21",
                            "22",
                            "23"
                        ],
                        "year": [
                            str(year)
                        ],
                        "month": [
                            str(month)
                        ],
                        "day": [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                            "7",
                            "8",
                            "9",
                            "10",
                            "11",
                            "12",
                            "13",
                            "14",
                            "15",
                            "16",
                            "17",
                            "18",
                            "19",
                            "20",
                            "21",
                            "22",
                            "23",
                            "24",
                            "25",
                            "26",
                            "27",
                            "28",
                            "29",
                            "30",
                            "31"
                        ]
                    },
                    "variable": [
                        "2_metre_dewpoint_temperature",
                        "2_metre_temperature",
                        "air_pressure_at_mean_sea_level",
                    ],
                    "format": "netcdf"
                },
                "era5-single-levels-reanalysis.nc")
        filetmp='/work/csp/vt17420/tmp'+str(year)+str(month)+'.nc'
        filetmp2='/work/csp/vt17420/tmp2'+str(year)+str(month)+'.nc'
        if  os.path.exists(filetmp):
            os.remove(filetmp)
        if  os.path.exists(filetmp2):
            os.remove(filetmp2)
        #mygrid='/work/csp/vt17420/ERA5/mygrid.txt'
        #cdo.remapbil('/work/csp/vt17420/ERA5/mygrid.txt',input=path1,output=filetmp2)
        filetmp2=path1
        cdo.sellonlatbox('-15,60,25,70',input=filetmp2,output=filetmp)
        ds = xr.open_dataset(filetmp)
        lons=ds.coords['longitude']
        lats=ds.coords['latitude']
        times = ds.coords['time']
        daux = ds[['d2m']].to_array()[0,:,:,:]
        taux = ds[['t2m']].to_array()[0,:,:,:]
        slpaux= ds [['msl']].to_array()[0,:,:,:]
        ntime=len(times)
        nlat=len(lats)
        nlon=len(lons)
        print(nlat)
        print(nlon)
        elevat_aux=np.zeros([nlat,nlon,ntime])
        print(elevat.shape)
        print(elevat_aux.shape)
        for i in range(ntime):
            elevat_aux[:,:,i]=elevat[0,0,:,:]
        del ds
        slp2ps = xr.apply_ufunc(
            Psurf,
            taux,
            slpaux,
            elevat_aux,
            dask="parallelized",
            input_core_dims=[['time'], ['time'],['time']],
            output_core_dims=[['time']],
            output_dtypes=[np.float])
        del slpaux
        # Compute relative humidity from the temperature and dewpoint
        RH = xr.apply_ufunc(
            RH_ECMWF,
            taux,
            daux,
            slp2ps,
            dask="parallelized",
            input_core_dims=[['time'], ['time'],['time']],
            output_core_dims=[['time']],
            output_dtypes=[np.float])
        del slp2ps
        del daux
        AppTempR = xr.apply_ufunc(
            AT_fun_russo,
            taux,
            RH,
            dask="parallelized",
            input_core_dims=[['time'], ['time']],
            output_core_dims=[['time']],
            output_dtypes=[np.float])

        lab1='atemp2m'
        output=AppTempR.transpose('time','latitude','longitude')
        lname='Apparent Temperarture Russo (2017)'
        #--------------------------------------------------------
        mon1="{:02d}".format(month)
        pathout = '/work/csp/vt17420/ERA5/hourly/'+lab1+'/'
        if not os.path.exists(pathout):
            os.makedirs(pathout)

        ncfileout = pathout+lab1+'_'+ str(year) + str(mon1)+'.nc'
        print(ncfileout)
        ncout = Dataset(ncfileout, 'w', format='NETCDF4')

        #--------------------------------------------------------
        ncout.createDimension('time', len(times))  # unlimited
        ncout.createDimension('latitude', len(lats))
        ncout.createDimension('longitude', len(lons))
        # create time axis
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
        longitude = ncout.createVariable('longitude', dtype('double').char, ('longitude'))
        longitude.standard_name = 'longitude'
        longitude.long_name = 'longitude'
        longitude.units = 'degrees_east'
        longitude.axis = 'X'
        # create variable array
        longitude[:] = lons[:]
        latitude[:] = lats[:]
        #times2 = np.array(dates)
        #times2 = times2.astype('datetime64[s]').tolist()
        #times2 = date2num(times2,units='hours since 1900-01-01 00:00:00.0',calendar='gregorian')
        time[:] = times
        vout = ncout.createVariable(lab1, dtype('double').char, ('time', 'latitude', 'longitude'))
        vout.long_name = lname
        vout.units = 'K'
        vout[:] = output

        # close files
        ncout.close()
        os.remove(filetmp)

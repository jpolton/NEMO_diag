# get_ERAint_windspeed_for_AMM60.py
#
# Extract compute and save ERAinterm wind on AMM60 grid:
# 1. load in the ERAinterim wind used for AMM60.
# 2. Compute the mean speed over the simulation period (Jun-Aug 2012)
# 3. interpolate onto AMM60 grid
# 4. save as a netcdf file
# 
# Note: used in internaltideharmonics_NEMO-integrated.ipynb
# 
# jpolton 22/06/18


def time_fn( time_counter, time_origin ):

    origin_datetime = datetime.datetime.strptime(time_origin, '%d-%b-%Y:%H:%M:%S')
    time_datetime = [ origin_datetime + datetime.timedelta(days=float(time_counter[i])) for i in range(len(time_counter)) ]
    time_str =  [ datetime.datetime.strftime(time_datetime[i], '%Y-%m-%d %H:%M:%S') for i in range(len(time_counter)) ]
    flag_err = 0
    return [time_str, time_datetime, flag_err]

def nearest(items, pivot):
    return items.index(min(items, key=lambda x: abs(x - pivot)))




from netCDF4 import Dataset
import numpy as np
import datetime
import matplotlib.pyplot as plt  # plotting
import sys # daving filename to output metadata

# output directory and filename
dir_out = '/work/n01/n01/jelt//NEMO/NEMOGCM_jdha/dev_r4621_NOC4_BDY_VERT_INTERP/NEMOGCM/CONFIG/XIOS_AMM60_nemo_harmPOLCOMS/EXP00/OUTPUT/'
file_out = 'ERAint_jun-aug2012_spd.nc'

# load met data
dir_met = '/work/n01/n01/kariho40/NEMO/FORCINGS/ATM/ERAint/'
file_met = 'ggas_y2012.nc'

f = Dataset(dir_met+file_met)

#U10 = f.variables['U10'][:] # nt,ny,nx
#V10 = f.variables['V10'][:] # nt,ny,nx


#lat = f.variables['latitude'][:] # ny
#lon = f.variables['longitude'][:] # nx

#nt, ny, nx = np.shape(U10)

# load AMM60 grid
#dirname = '/projectsa/pycnmix/jelt/AMM60/'
dir_mod = '/work/n01/n01/jelt//NEMO/NEMOGCM_jdha/dev_r4621_NOC4_BDY_VERT_INTERP/NEMOGCM/CONFIG/XIOS_AMM60_nemo_harmPOLCOMS/EXP00/OUTPUT/'
file_mod = 'AMM60_1d_20120801_20120831_D2_Tides.nc'

fAMM60 = Dataset(dir_mod+file_mod)

nav_lat_grid_T = fAMM60.variables['nav_lat_grid_T'][:] # my, mx
nav_lon_grid_T = fAMM60.variables['nav_lon_grid_T'][:] # my, mx
nav_lat_grid_T[nav_lat_grid_T==0] = np.nan
nav_lon_grid_T[nav_lat_grid_T==0] = np.nan


latbounds = [np.nanmin(nav_lat_grid_T.flatten()), np.nanmax(nav_lat_grid_T.flatten())]
lonbounds = [np.nanmin(nav_lon_grid_T.flatten()), np.nanmax(nav_lon_grid_T.flatten())]

# Load only the relevant spatial data

lat = f.variables['latitude'][:]
lon = f.variables['longitude'][:]
time = f.variables['t'][:] # nt
time_origin =  "01-JAN-2012:00:00:00"
time_units =  "days since 2012-01-01 00:00:00"

# latitude lower and upper index
latli,latui = np.sort([np.argmin( np.abs( lat - latbounds[i] ) ) for i in range(2)])
print 'NB the lat coords in the met data are indexed from north to south'

# longitude lower and upper index
lonli = np.argmin( np.abs( lon - lonbounds[0] ) )
lonui = np.argmin( np.abs( lon - lonbounds[1] ) )


# Identify and extract 3 months of interest
if 'days' in time_units:
    [time_str, time_datetime, flag_err] = time_fn( time, time_origin )

start_date = "01-Jun-2012"
end_date = "31-Aug-2012"

tli = nearest(time_datetime, datetime.datetime.strptime(start_date, '%d-%b-%Y'))
tui = nearest(time_datetime, datetime.datetime.strptime(end_date, '%d-%b-%Y'))

# subset
U10 = f.variables['U10'][ tli:tui , latli:latui , lonli:lonui ]
V10 = f.variables['V10'][ tli:tui , latli:latui , lonli:lonui ]

# Compute speed
#spd = np.sqrt(np.square(U10) + np.square(V10))
spd = np.mean(np.sqrt( U10*U10 + V10*V10 ), axis=0)

lon_sub = lon[lonli:lonui]
lat_sub = lat[latli:latui] 

# Interpolate onto AMM60 grid
from scipy import interpolate
yy,xx = np.meshgrid(lat_sub, lon_sub)



fn = interpolate.interp2d(lon_sub,lat_sub, spd, kind='linear')

spd_flat = [fn(  nav_lon_grid_T.ravel()[i], nav_lat_grid_T.ravel()[i]  ) for i in range(len(nav_lon_grid_T.ravel()))]
spd_AMM60 = np.reshape(spd_flat, np.shape(nav_lon_grid_T) )

spd_AMM60[np.isnan(nav_lat_grid_T)] = np.nan

#f = interpolate.interp2d(yy, xx, spd, kind='linear')
#spd_new = f(nav_lat_grid_T, nav_lon_grid_T)

# Plot image of speed data
plt.subplot(3,1,1)
plt.pcolormesh(np.ma.masked_where(np.isnan(spd_AMM60),spd_AMM60))
plt.clim([0,9])
#plt.pcolormesh(nav_lon_grid_T, nav_lat_grid_T,z)
plt.colorbar()

plt.subplot(3,1,2)
plt.pcolormesh(lon_sub, lat_sub, spd)
plt.colorbar()

plt.subplot(3,1,3)
plt.pcolormesh(nav_lon_grid_T, nav_lat_grid_T, spd_AMM60)
plt.colorbar()
plt.show()

# Save limited area to netcdf file.
print 'Saving to {}'.format( file_out )

########################################################
# Open files and create dimensions and variable handles

dout = Dataset(dir_out+file_out, 'w', format='NETCDF4_CLASSIC')

din = fAMM60
nx = len(din.dimensions['x_grid_T'])
ny = len(din.dimensions['y_grid_T'])

## Create dimensions
lon_h = dout.createDimension('x', nx)
lat_h = dout.createDimension('y', ny)


## Create coordinate variables for 3-dimensions
longitudes = dout.createVariable('nav_lon_grid_T', np.float32, ('y','x',))
latitudes = dout.createVariable('nav_lat_grid_T', np.float32, ('y','x',))

# copy all the lon and lat attributes, and time
var = din.variables['nav_lon_grid_T']
for attr in var.ncattrs():
#    print attr, '=', getattr(var, attr)
    setattr(longitudes,attr,getattr(var,attr))

var = din.variables['nav_lat_grid_T']
for attr in var.ncattrs():
#    print attr, '=', getattr(var, attr)
    setattr(latitudes,attr,getattr(var,attr))


###########################################################################
## Create the 2d wind speed variable

spd_h = dout.createVariable('spd', np.float32, ('y','x'))
spd_h.units = 'm/s'
spd_h.standard_name = 'wind speed at 10m'
spd_h.long_name = 'average wind speed at 10m from 1-Jun-2012 : 31-Aug-2012'


## Populate coordinate variables
latitudes[:] = din.variables['nav_lat_grid_T'][:]
longitudes[:] = din.variables['nav_lon_grid_T'][:]

#####################
## Create variables

spd_h[:] = spd_AMM60



# Create global attributes
import time
dout.description = 'description'
dout.title = 'ocean T grid variables'
dout.Conventions = 'CF-1.5'
dout.production = 'jelt: '+sys.argv[0]
dout.history = 'Created ' + time.ctime(time.time())
dout.source = dir_met + '/' + file_met

#from netCDF4 import chartostring
#print chartostring(constit[:])

################
# Write the file
din.close()
dout.close()

# scratch
#latbounds = [ 40 , 43 ]
#lonbounds = [ -96 , -89 ] # degrees east ? 
#lats = f.variables['latitude'][:] 
#lons = f.variables['longitude'][:]

## latitude lower and upper index
#latli = np.argmin( np.abs( lats - latbounds[0] ) )
#latui = np.argmin( np.abs( lats - latbounds[1] ) ) 

## longitude lower and upper index
#lonli = np.argmin( np.abs( lons - lonbounds[0] ) )
#lonui = np.argmin( np.abs( lons - lonbounds[1] ) ) 

# subset
#airSubset = f.variables['air'][ : , latli:latui , lonli:lonui ] 

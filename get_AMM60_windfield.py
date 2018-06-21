# wind field

def time_fn( time_counter, time_origin ):

    origin_datetime = datetime.datetime.strptime(time_origin, '%d-%b-%Y:%H:%M:%S')
    time_datetime = [ origin_datetime + datetime.timedelta(days=time_counter[i]) for i in range(len(time_counter)) ]
    time_str =  [ datetime.datetime.strftime(time_datetime[i], '%Y-%m-%d %H:%M:%S') for i in range(len(time_counter)) ]
    flag_err = 0
    return [time_str, time_datetime, flag_err]

def nearest(items, pivot):
    return items.index(min(items, key=lambda x: abs(x - pivot)))




from netCDF4 import Dataset
import numpy as np
import datetime

# load met data
dirname = '/work/n01/n01/kariho40/NEMO/FORCINGS/ATM/ERAint/'
filename = 'ggas_y2012.nc'

f = Dataset(dirname+filename)

U10 = f.variables['U10'][:] # nt,ny,nx
V10 = f.variables['V10'][:] # nt,ny,nx

time = f.variables['t'][:] # nt
time_origin =  "01-JAN-2012:00:00:00"
time_units =  "days since 2012-01-01 00:00:00"

lat = f.variables['latitude'][:] # ny
lon = f.variables['longitude'][:] # nx

nt, ny, nx = np.shape(U10)

# load AMM60 grid
dirname = '/projectsa/pycnmix/jelt/AMM60/'
filename = 'AMM60_1d_20120801_20120831_D2_Tides.nc'

fAMM60 = Dataset(dirname+filename)

nav_lat_grid_T = f.variables['nav_lat_grid_T'][:] # my, mx
nav_lon_grid_T = f.variables['nav_lon_grid_T'][:] # my, mx

# Compute speed
sp = np.sqrt(U10**2 * V10**2)

# Identify and extract 3 months of interest
if 'days' in time_units:
    [time_str, time_datetime, flag_err] = time_fn( time_counter, time_origin )

start_date = "01-Jun-2012"
end_date = "31-Aug-2012"

Istart = nearest(time_datetime, datetime.datetime.strptime(start_date, '%d-%b-%Y'))
Iend = nearest(time_datetime, datetime.datetime.strptime(end_date, '%d-%b-%Y'))

# Interpolate onto AMM60 grid
from scipy import interpolate
f = interpolate.interp2d(lat, lon, spd, kind='linear')
spd_new = f(nav_lat_grid_T, nav_lon_grid_T)

# Plot image of speed data
plt.pcolormesh(nav_lat_grid_T, nav_lon_grid_T,spd_new)
plt.colorbar()

# Save limited area to netcdf file.


# scratch

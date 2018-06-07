
#extract_sumbaStrait_indices.py

"""
Extract Sumba Strait indices

Here, the following coordinate for CTD stations:
Station 1 : Longitude 118.775 Latitude -8.9
Station 2 : Longitude 118.775 Latitude -9.04761
Station 3 : Longitude 118.775 Latitude -9.19317
Station 4 : Longitude 118.775 Latitude -9.34818
"""
from netCDF4 import Dataset
import numpy as np


def findJI(lat, lon, lat_grid, lon_grid):
     """
     Simple routine to find the nearest J,I coordinates for given lat lon
     Usage: [J,I] = findJI(49, -12, nav_lat_grid_T, nav_lon_grid_T)
     """
     dist2 = np.square(lat_grid - lat) + np.square(lon_grid - lon)
     [J,I] = np.unravel_index( dist2.argmin(), dist2.shape  )
     return [J,I]


dirname = '/work/n01/n01/jelt/SEAsia/trunk_NEMOGCM_r8395/CONFIG/SEAsia/EXP_openbcs/'
filename = 'SEAsia_1h_19791101_19791130_grid_T.nc'

f = Dataset(dirname+filename)


nav_lat = f.variables['nav_lat'][:] # (y,x)
nav_lon = f.variables['nav_lon'][:] # (y,x)

lats = -8.9, -9.04761, -9.19317, -9.34818
lons = 118.775, 118.775, 118.775, 118.775


for count in range(len(lats)):
    [J,I] = findJI(lats[count], lons[count], nav_lat, nav_lon)
    print 'J:{}, I:{}'.format(J,I)

"""
Need to add some header information
Need to add saveplot commands
jelt 5 Apr 2018
"""


from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plt  # plotting



dirname = ''
config = 'SEAsia'
fr = Dataset(dirname + config + '_rivers.nc')
fd = Dataset(dirname + 'domain_cfg.nc')

rlat = fr.variables['lat'][:] 
rlon = fr.variables['lon'][:] 
runoff = fr.variables['rorunoff'][:]
time_counter = fr.variables['time_counter'][:]
fr.close()

nav_lat = fd.variables['nav_lat'][:]
nav_lon = fd.variables['nav_lon'][:]
bott = fd.variables['bottom_level'][:].squeeze()
fd.close()


# Annual mean river flow
rr = np.sum(runoff[:,:,:],axis=0)
rr_m = np.ma.masked_where( rr==0, rr)
fig = plt.figure(figsize=(7.5,5))
plt.contourf(nav_lon,nav_lat, bott, levels=[1,2,75])
plt.scatter(nav_lon, nav_lat, s=1E3*rr_m, c=rr_m, cmap='Spectral')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.grid('on')
plt.title('Annual discharge')
plt.colorbar()
plt.show()
fig.savefig('SEAsia_rivers.png', dpi=120)

#########################################################################################
#########################################################################################


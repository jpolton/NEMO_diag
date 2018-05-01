"""
Need to add some header information
Need to add saveplot commands
jelt 5 Apr 2018
"""


from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.colors as colors # colormap fiddling



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('nipy_spectral')
#cmap = plt.get_cmap('BrBG_r')
new_cmap = truncate_colormap(cmap, 0.0, 1.0)



dirname = '/work/n01/n01/jelt/SEAsia/INPUTS/'
config = 'SEAsia'
fin = Dataset(dirname + 'cut_down_19791101d05_SEAsia_grid_T.nc')
#fgr = Dataset(dirname + 'domain_cfg.nc')

votemper = fin.variables['votemper'][:] 
vosaline = fin.variables['vosaline'][:] 


nav_lat = fin.variables['nav_lat'][:]
nav_lon = fin.variables['nav_lon'][:]
#bott = fgr.variables['bottom_level'][:].squeeze()
#fgr.close()
fin.close()

sst = votemper[0,0,:,:].squeeze()
sss = vosaline[0,0,:,:].squeeze()

print np.shape(sst)

# SST
fig1 = plt.figure(figsize=(7.5,5))
plt.pcolormesh( nav_lon, nav_lat, sst, cmap='Spectral_r')
plt.xlim([76,135])
plt.ylim([-21,25])
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.grid('on')
plt.title('SST')
plt.colorbar()

fig1.savefig('SST_parent.png', dpi=120)

#########################################################################################
# SST
fig2 = plt.figure(figsize=(7.5,5))
plt.pcolormesh( nav_lon, nav_lat, sss,
             cmap=new_cmap)
#plt.clim([31,37])

#plt.pcolormesh( nav_lon, nav_lat, sss, cmap='Spectral_r')
plt.xlim([76,135])
plt.ylim([-21,25])
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.grid('on')
plt.title('SSS')
plt.colorbar()

fig2.savefig('SSS_parent.png', dpi=120)
plt.show()
#########################################################################################
#########################################################################################


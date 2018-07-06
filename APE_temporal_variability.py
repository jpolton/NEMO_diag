
import sys
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.colors as mc   # fancy symetric colours on log scale
import matplotlib.cm as cm   # colormap functionality
import seaborn as sns # cyclic colormap for harmonic phase
sys.path.insert(0, "/login/jelt/python/ipynb/NEMO/")
#from internaltideharmonics_NEMO import internaltideharmonics_NEMO as ITh
import internaltideharmonics_NEMO as ITh
import numpy.ma as ma # masks

config = 'AMM60'
rho0 = 1E3
constit = 'M2'

# set paths
[rootdir, dirname, dstdir] = ITh.get_dirpaths(config)
filebase =  'AMM60_1d_20120801_20120831'
filebaseTide = 'AMM60_1d_20120801_20120831'

ftide = Dataset(dirname + filebaseTide + '_D2_Tides.nc')
fm = Dataset(dirname + 'mesh_mask.nc')
fw = Dataset(dirname + filebase + '_grid_W.nc')

[constit_list, period_list ] = ITh.harmonictable(dirname+'../harmonics_list.txt')
nh = len(constit_list)

#load lat and lon
nav_lat_grid_T = fm.variables['nav_lat'][:] # (y,x)
nav_lon_grid_T = fm.variables['nav_lon'][:] # (y,x)
[ny, nx] = np.shape(nav_lat_grid_T)

## Load mbathy
mbathy = np.squeeze(fm.variables['mbathy'][:]) # (time_counter, y, x)  --> (y, x)
e3t = np.squeeze(fm.variables['e3t_0'][:]) # (depth, y, x)
e3w = np.squeeze(fm.variables['e3w_0'][:]) # (depth, y, x)
gdept = np.squeeze(fm.variables['gdept_0'][:]) # (depth, y, x)
gdepw = np.squeeze(fm.variables['gdepw_0'][:]) # (depth, y, x)
H = np.zeros((ny,nx))
for J in range(ny):
	for I in range(nx):
		H[J,I] = gdepw[mbathy[J,I], J,I]

iconst = constit_list.index(constit)


## Load in harmonic data
#print 'loading {} data'.format(constit_list[iconst])
#[elev, u_bt, v_bt, u_bc, v_bc, wvel] = ITh.loadharmonic(ftide, constit_list[iconst],mbathy)


dzw = np.copy(e3w) # Define vertical spacing at W-points that is fixed for the top layer too.
dzw[0,:,:] = gdept[0,:,:]

##############################################################################
# Compute APE
print 'Computing APE' 
iday = 1
print 'iday {}'.format(iday)

# load in vertical velocity
w2 = ftide.variables[constit+'x_w3D'][:]**2   + ftide.variables[constit+'y_w3D'][:]**2   # (depth, y, x)  e.g. M2x_u


# load in background stratification
N2 =  fw.variables['N2_25h'][:] # (time_counter, depth, y, x). W-pts. Surface value == 0
[nt,nz,ny,nx] = np.shape(N2)
#N2 = N2[iday,:,:,:]

APE = 0.5*rho0 * np.nansum( N2 * w2 *(period_list[iconst]*3600/4.)**2 * dzw, axis=N2.shape.index(nz)) / H


# mask land
APE[APE==0] = np.nan

timestamps = [int(ii) for ii in np.linspace(0,nt-1,9)] # [0, 5, 10,  20]
for i in timestamps:
	plt.subplot(3,3,1+timestamps.index(i))
	plt.pcolormesh( nav_lon_grid_T, nav_lat_grid_T, np.log10(APE[i,:,:])-np.log10(APE[0,:,:]) )
	plt.clim([-2,2])
	plt.colorbar()
	plt.xlim(-13, -0)
	plt.ylim(46, 53)
	plt.title(str(i))
plt.show()

print 'This suggests that the APE does not change very much over the month (when viewed on a log scale). Relative changes happen with APE is small'


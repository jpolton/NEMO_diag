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
constit = 'K2'

# set paths
[rootdir, dirname, dstdir] = ITh.get_dirpaths(config)
filebase =  'AMM60_1d_20120801_20120831'
filebaseTide = 'AMM60_1d_20120801_20120831'

fileh = Dataset(dirname + filebaseTide + '_D2_Tides.nc')
fm  = Dataset(dirname + 'mesh_mask.nc')

[constit_list, period_list ] = ITh.harmonictable(dirname+'../harmonics_list.txt')
nh = len(constit_list)

#load lat and lon
nav_lat_grid_T = fm.variables['nav_lat'][:] # (y,x)
nav_lon_grid_T = fm.variables['nav_lon'][:] # (y,x)
[ny, nx] = np.shape(nav_lat_grid_T)

## Load mbathy
mbathy = np.squeeze(fm.variables['mbathy'][:]) # (time_counter, y, x)  --> (y, x)
e3t = np.squeeze(fm.variables['e3t_0'][:]) # (depth, y, x)


iconst = constit_list.index(constit)


## Load in harmonic data
print 'loading {} data'.format(constit_list[iconst])
[elev, u_bt, v_bt, u_bc, v_bc, wvel] = ITh.loadharmonic(fileh, constit_list[iconst],mbathy)

##############################################################################
# Compute bottom stress energy loss
print 'Computing harmonic energy loss through bed friction'
# compute bottom layer thickness
e3t_b = [e3t[mbathy[JJ,II], JJ,II] for JJ, II in [(JJ,II) for JJ in range(ny) for II in range(nx)]]
e3t_b = np.reshape(e3t_b,(ny,nx))
D = np.zeros((nh,ny,nx))


D[iconst,:,:] = ITh.get_bedterm( fileh, constit_list[iconst], rho0, e3t_b, mbathy, plotFlag=True)

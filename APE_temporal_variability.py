
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

print len(sys.argv)
print str(sys.argv)
if len(sys.argv) > 1:
	region = str(sys.argv[1])
else:
	region = 'Celtic'

if len(sys.argv) > 2:
	month = str(sys.argv[2])
else:
	month = 'Aug'

config = 'AMM60'
rho0 = 1E3
constit = 'M2'

# set paths
[rootdir, dirname, dstdir] = ITh.get_dirpaths(config)
if 'Jun' in month:
	filebase =  'AMM60_1d_20120601_20120630'
elif 'Jul' in month:
	filebase =  'AMM60_1d_20120701_20120731'
elif 'Aug' in month:
	filebase =  'AMM60_1d_20120801_20120831'
else:
	print 'panic. Not expecting that month'

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

# load in vertical velocity
w2 = ftide.variables[constit+'x_w3D'][:]**2   + ftide.variables[constit+'y_w3D'][:]**2   # (depth, y, x)  e.g. M2x_u


# load in background stratification
N2 =  fw.variables['N2_25h'][:] # (time_counter, depth, y, x). W-pts. Surface value == 0
[nt,nz,ny,nx] = np.shape(N2)

APEonH = 0.5*rho0 * np.nansum( N2 * w2 *(period_list[iconst]*3600/4.)**2 * dzw, axis=N2.shape.index(nz)) / H  # [time_counter, y, x]


# mask land
APEonH[APEonH==0] = np.nan



region = 'Celtic'
APE0 = 1E-2
areaAPE = np.zeros(nt)
area = np.zeros(nt)

# Create mask
if region=='Celtic':
	mask_bathy = np.array(nav_lon_grid_T > -13., dtype=int) * np.array(nav_lon_grid_T < 0. , dtype=int) \
		* np.array(nav_lat_grid_T > 46 , dtype=int) * np.array(nav_lat_grid_T < 53 , dtype=int) \
		* np.array(H < 200, dtype=int) * np.array(H > 10, dtype=int)

elif region=='subCeltic':
	mask_bathy = np.array(nav_lon_grid_T > -10., dtype=int) * np.array(nav_lon_grid_T < -9 , dtype=int)\
		* np.array(nav_lat_grid_T > 49 , dtype=int) * np.array(nav_lat_grid_T < 50 , dtype=int) \
		* np.array(H < 200, dtype=int) * np.array(H > 10, dtype=int)

APE_mask = np.array(APEonH > APE0, dtype=int)

if config == 'AMM60':
	dx2 = (1.8E3)**2
elif config == 'AMM7':
	dx2 = (7E3)**2

for i in range(nt):
	areaAPE[i] = dx2*1E-6*np.nansum( ma.masked_where(mask_bathy*APE_mask[i,:,:] == 0, APEonH[i,:,:]*H ).flatten() )
	area[i] =np.nansum( (mask_bathy*APE_mask[i,:,:]).flatten() )
	print 'SUM: APE = {:06.2e} MJ'.format( areaAPE[i] )
	#plt.plot(i,areaAPE[i],'+')
fig = plt.figure()
plt.rcParams['figure.figsize'] = (15.0, 15.0)
plt.subplot(2,1,1)
plt.plot( np.arange(nt), areaAPE), plt.ylabel('areaAPE')
plt.subplot(2,1,2)
plt.plot( np.arange(nt), area), plt.ylabel('area')
plt.savefig(month+'APE.png')
plt.title(month+ ' APE')
#plt.show()
plt.close(fig)

if(0):

	timestamps = [int(ii) for ii in np.linspace(0,nt-1,9)] # [0, 5, 10,  20]
	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 15.0)
	for i in timestamps:
		plt.subplot(3,3,1+timestamps.index(i))
		plt.pcolormesh( nav_lon_grid_T, nav_lat_grid_T, np.log10(APEonH[i,:,:])-np.log10(APEonH[0,:,:])*0 )
		plt.clim([-2,2])
		plt.colorbar()
		plt.xlim(-13, -0)
		plt.ylim(46, 53)
		plt.title(str(i))
	plt.savefig('AugAPE.png')
	#plt.show()
	plt.close(fig)
	print 'This suggests that the APE (on H) does not change very much over the month (when viewed on a log scale). Relative changes happen with APE is small'



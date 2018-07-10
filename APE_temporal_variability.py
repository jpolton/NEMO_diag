
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
e1u = np.squeeze(fm.variables['e1u'][0,:,:]) # (time_counter, y, x)  --> (y, x)
e2v = np.squeeze(fm.variables['e2v'][0,:,:]) # (time_counter, y, x)  --> (y, x)
#e1u = np.squeeze(fm.variables['e1u'][:]) # (ny, nx)
#e2v = np.squeeze(fm.variables['e2v'][:]) # (ny, nx)
e3t = np.squeeze(fm.variables['e3t_0'][:]) # (depth, y, x)
e3w = np.squeeze(fm.variables['e3w_0'][:]) # (depth, y, x)
gdept = np.squeeze(fm.variables['gdept_0'][:]) # (depth, y, x)
gdepw = np.squeeze(fm.variables['gdepw_0'][:]) # (depth, y, x)
tmask = fm.variables['tmask'][:] # (1, depth, y ,x)

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
wvel = ftide.variables[constit+'x_w3D'][:]   + 1.j*ftide.variables[constit+'y_w3D'][:]   # (depth, y, x)  e.g. M2x_w
w2 = np.square( np.abs(wvel) ) 


# load in background stratification
N2 =  fw.variables['N2_25h'][:] # (time_counter, depth, y, x). W-pts. Surface value == 0
[nt,nz,ny,nx] = np.shape(N2)

APEonH = 0.5*rho0 * np.nansum( N2 * w2 *(period_list[iconst]*3600/4.)**2 * dzw, axis=N2.shape.index(nz)) / H  # [time_counter, y, x]


# mask land
APEonH[APEonH==0] = np.nan

#####
# Compute C_bt -- IN PROGRESS
[elev, u_bt, v_bt, u_bc, v_bc, wvel] = ITh.loadharmonic(ftide, constit_list[iconst],mbathy)    
g = 9.81
print 'compute '+ constit + ' rho_constit [nt,nz,ny,nx]'
rho_constit = -1j * (period_list[iconst]*3600. *0.5/np.pi) * (rho0/g) * N2[:,:,:,:] * wvel # (nt,nz,ny,nx)
rho_constit = np.ma.masked_where( np.tile(tmask,(nt,1,1,1)) == 0, rho_constit)

C_bt = ITh.get_C(u_bt, v_bt, rho_constit, e1u, e2v, e3w, gdepw, H, mbathy)



region = 'Celtic'
APE0 = 1E-2
areaAPE = np.zeros(nt)
area = np.zeros(nt)
sumC = np.zeros(nt)
time = np.zeros(nt)

# Create mask
mask_bathy = ITh.define_mask(nav_lon_grid_T, nav_lat_grid_T, H, region)

APE_mask = np.array(APEonH > APE0, dtype=int)

if config == 'AMM60':
	dx2 = (1.8E3)**2
elif config == 'AMM7':
	dx2 = (7E3)**2

for i in range(nt):
	areaAPE[i] = dx2*1E-6*np.nansum( ma.masked_where(mask_bathy*APE_mask[i,:,:] == 0, APEonH[i,:,:]*H ).flatten() )
	area[i] =np.nansum( (mask_bathy*APE_mask[i,:,:]).flatten() )
	sumC[i] =np.nansum( (mask_bathy*C_bt[i,:,:]).flatten() )
	time[i] = i
	print 'SUM: APE = {:06.2e} MJ'.format( areaAPE[i] )
	#plt.plot(i,areaAPE[i],'+')

# Save variables
import datetime
import pandas as pd
import numpy as np
from AMM60_tools import NEMO_fancy_datestr # NEMO date strings

# load in time
time_counter = fw.variables['time_counter'][:] # vector
time_origin = fw.variables['time_counter'].time_origin
time_calendar = fw.variables['time_counter'].calendar
time_units = fw.variables['time_counter'].units

[time_str, time_datetime, flag_err] = NEMO_fancy_datestr( time_counter, time_origin ) # internal tide data

columns = ['areaAPE','area', 'C']
data = np.array([areaAPE, area, sumC] ).T
df = pd.DataFrame(data, index=time_datetime, columns=columns)

fig = plt.figure()
plt.rcParams['figure.figsize'] = (15.0, 15.0)
plt.subplot(3,1,1)
plt.plot( np.arange(nt), areaAPE), plt.ylabel('areaAPE')
plt.subplot(3,1,2)
plt.plot( np.arange(nt), area), plt.ylabel('area')
plt.subplot(3,1,3)
plt.plot( np.arange(nt), sumC), plt.ylabel('C')
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



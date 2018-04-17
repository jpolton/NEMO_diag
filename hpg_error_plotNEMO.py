# quick plot NEMO output
#########################
#
# jp 1 Feb 2018
#
# Quick script to read in netCDF output from v4 NEMO and see if it is realistic


from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.cm as cm  # colormaps
import sys # Exit command
######%matplotlib inline

def plot_gauges(config):

	if config == SWPacific:
		pass	
		#plt.plot()


## Now do the main routine stuff
if __name__ == '__main__':


  #config = 'SWPacific'
  config = 'SEAsia'

  #flag = 0 # Read output.abort
  flag = 1 # Read SWPacific*nc
  #flag = 2 # bdydta mask
  #flag = 3 # Tide data

  print 'flag =',flag

  # Set path
  dirname = ''#/Users/jeff/Desktop/'
  if flag == 1:
	#filename = config + '_1h_20000101_20000110_SSH.nc'
	#filename = config + '_1h_20000101_20000120_SSH.nc'
	filename = config + '_5d_20000201_20000301_grid_T.nc' #'_5d_20000102_20000131_grid_T.nc' # '_1h_20000102_20000111_SSH.nc'  #20000101_20000101_SSH.nc'
	#filename = config + '_1h_20000101_20000130_SSH.nc'
        filename = dirname + filename
	var = 'zos'
  else:
	print 'Panic!'

  ## Load file and variables
  f = Dataset(filename)

  #load lat and lon
  nav_lat = f.variables['nav_lat'][:] # (y,x)
  nav_lon = f.variables['nav_lon'][:] # (y,x)
  zos = f.variables[var][-2,:,:].squeeze() # (time_counter, y, x)
  lim = np.nanmax(np.abs(zos)) # Find extrema
  print 'max abs(lim)=',lim
  print 'shape zos=',np.shape(zos)
  print 'min nav_lon', np.min(nav_lon.flatten())

  # mask out zero lat and lon points. Doesn't work!
#  zos = np.ma.masked_where( np.square(nav_lon)+np.square(nav_lat)<0.1 , zos)
#  nav_lat = np.ma.masked_where( np.square(nav_lon)+np.square(nav_lat)<0.1 , nav_lat)
#  nav_lon = np.ma.masked_where( np.square(nav_lon)+np.square(nav_lat)<0.1 , nav_lon)
#  zos = np.ma.masked_invalid(zos)
#  nav_lat = np.ma.masked_invalid(nav_lat)
#  nav_lon = np.ma.masked_invalid(nav_lon)

  print 'min nav_lon', np.min(nav_lon.flatten())
  scale = 1E3
  lim = lim *scale

# Plot data
  cmap = cm.Spectral
  fig = plt.figure()
  plt.rcParams['figure.figsize'] = (20.0, 10.0)

  #plt.pcolormesh(nav_lon,nav_lat, nav_lon, cmap=cmap )
  
  #plt.contourf(nav_lon,nav_lat, zos*1E3, cmap=cmap, levels=np.arange(-1.5,1.5+0.1,0.1))
  plt.subplot(1,2,1)
  plt.contourf(nav_lon,nav_lat, zos*scale, cmap=cmap, levels=np.arange(-lim,lim+lim*0.1,lim*0.1))
  plt.clim([-lim,lim])
  plt.xlim([70,140])
  plt.ylim([-21,26])
  plt.colorbar()
  #plt.title('SSH error at/near simulation end [mm]')
  plt.xlabel('long'); plt.ylabel('lat')

  plt.subplot(1,2,2)
  plt.contourf(nav_lon,nav_lat, zos*scale, cmap=cmap, levels=np.arange(-2,2+2*0.1,2*0.1))
  plt.clim([-2,2])
  plt.xlim([70,140])
  plt.ylim([-21,26])
  plt.colorbar()
  plt.xlabel('long'); plt.ylabel('lat')
  #plt.show()
  plt.suptitle('SSH error at/near simulation end [mm]')

  fname = 'hpg_error_map_'+config+'.png'
  plt.savefig(fname) 
  sys.exit("End script")

###############################################################################
###############################################################################

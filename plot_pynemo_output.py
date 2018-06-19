# -*- coding: utf-8 -*-
"""

** Summary: **
Plot boundary conditions from TPXO and FES2014 
** origin: ** 
** Author: ** JP 15 Jun 2018
** Changelog: **
* 15 Jun 2018: convert back to non animated co-tidal charts

** Issues: **
* None. It is amazing.

** Usage: ** plot_pynemo_output.py 'ssh'
"""

##################### LOAD MODULES ###########################

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.gridspec as gridspec
from  mpl_toolkits.basemap import Basemap
import sys
sys.path.append('../jcomp_tools_dev/') # Add the directory with the amm60_data_tools.py file to path. Not used on the NOCL SAN
import os #, sys  # removing files and building animation
import datetime
##%matplotlib inline
from   matplotlib.font_manager import FontProperties
import matplotlib.colors as colors
import matplotlib.ticker as ticker
font0  = FontProperties(); font = font0.copy(); font.set_weight('bold'); font.set_size('large')

## Check host name and locate file path
#import socket
#if str('livmap') in socket.gethostname().lower():
#    dirname = ''
#elif str('livljobs') in socket.gethostname().lower():
#    dirname = ''
#elif str('eslogin') in socket.gethostname().lower():
#    dirname = '/work/n01/n01/jelt/SEAsia/INPUTS/'
#elif str('mola') in socket.gethostname().lower() or str('tardis') in socket.gethostname().lower() :
#    dirname = '/Volumes/work/jelt/NEMO/Solent/INPUTS/'
#else:
#    print 'There is no working ELSE option'

#################### USER PARAMETERS ##########################

print 'Input arguements: {}'.format( str(sys.argv) )
if len(sys.argv) > 1:
	type = str(sys.argv[1])
else:
	#type = 'tide'
	type = 'ssh'
	#type = 'u2d'
	#type = 'u3d'
	#type = 'tra'

region = 'SEAsia'

##############################################################

dirname = '/work/n01/n01/jelt/' + region +'/INPUTS/'
tag='jelt' # extra string on output files
#dirname = '/work/n01/n01/jdha/2018/SEAsia/EXP_openbcs/bcs/'
#tag='jdha' # extra string on output files

if type == 'tide':
	#source = 'FES'
	source = 'TPXO'
	con = 'M2'
	varnam = con
	#dirname = '/work/n01/n01/jelt/Solent_surge/INPUTS/'
	#filename = 'Solent_bdytide_rotT_'+con+'_grid_T.nc'
	#print 'Inspect file: {}'.format(dirname+source+'/'+filename)
	#f = Dataset(dirname+source+'/'+filename)

	filename = region + '_bdytide_rotT_'+con+'_grid_T.nc'
	print 'Inspect file: {}'.format(dirname+filename)
	f = Dataset(dirname+filename)
	z_cos = f.variables['z1'][:] # yb,xb
	z_sin = f.variables['z2'][:] # yb,xb
	print 'Inspect file: {}'.format(dirname+source+'/'+filename)
	var = np.squeeze(z_cos + 1j*z_sin) # 1-dimensionnal

elif type == 'ssh':
	varnam = 'sossheig'
	source = 'ORCA0083'
	filename = region + '_full_bt_bdyT_y1979m11.nc'
	f = Dataset(dirname+filename)
	print 'Inspect file: {}'.format(dirname+filename)
	var = f.variables[varnam] # nt,yb,xb
        count = 0 # time step to inspect
        var = np.squeeze(var[count,:,:]) # 1-dimensional

elif type == 'u2d':
	varnam = 'vobtcrtx'
	source = 'ORCA0083'
	filename = region + '_spl_vel_bdyU_y1979m11.nc'
	f = Dataset(dirname+filename)
	print 'Inspect file: {}'.format(dirname+filename)
	var = f.variables[varnam] # nt,yb,xb
        count = 0 # time step to inspect
        var = np.squeeze(var[count,:,:]) # 1-dimensional

elif type == 'u3d':
	varnam = 'vozocrtx'
	source = 'ORCA0083'
	filename = region + '_spl_vel_bdyU_y1979m11.nc'
	f = Dataset(dirname+filename)
	print 'Inspect file: {}'.format(dirname+filename)
	#var = f.variables['vozocrtx'][:] # nt,nz,yb,xb
	var = f.variables[varnam][:] # nt,nz,yb,xb
        count = 0 # time step to inspect
        lev = 0 # level to inspect
        var = np.squeeze(var[count,lev,:,:]) # 1-dimensional

elif type == 'tra':
	varnam = 'votemper'
	varnam = 'vosaline'
	source = 'ORCA0083'
	filename = region + '_spl_vel_bdyT_y1979m11.nc'
	f = Dataset(dirname+filename)
	print 'Inspect file: {}'.format(dirname+filename)
	var = f.variables[varnam] # nt,nz,yb,xb
        count = 0 # time step to inspect
        lev = 0 # level to inspect
        var = np.squeeze(var[count,lev,:,:]) # 1-dimensional
else:
	print 'panic'

## output filename
fname = "./FIGURES/"+tag+source+'_'+varnam+'_'+filename.replace('.nc','.png')

## Check the loaded variable is the expected size (1D)
if len(np.shape(var)) > 1:
	print 'unexpected dimension size of variable var: {}',format(np.shape(var))

dpts = 10 # subsampling freq
nbidta = f.variables['nbidta'][:] # yb,xb
nbjdta = f.variables['nbjdta'][:] # yb,xb

nav_lat = f.variables['nav_lat'][:] # y,x
nav_lon = f.variables['nav_lon'][:] # y,x


print 'Shape of variable: {}'.format(np.shape(np.real(var)))
yb, xb = np.shape(nbidta)
pts = range(0,xb,dpts)
X = [ nav_lon[ nbjdta[0,i],nbidta[0,i] ] for i in range(0,xb,dpts)]
Y = [ nav_lat[ nbjdta[0,i],nbidta[0,i] ] for i in range(0,xb,dpts)]
Z = [ np.abs( var[i] ) for i in range(0,xb,dpts)]

#fig = plt.figure( figsize = ( 8.4, 2+10*(Latmax-Latmin)/(Lonmax-Lonmin) ) )
fig = plt.figure( figsize = ( 8.4, 2+10) )

## DEFINE AND PLOT BASEMAP PROJECTION
ax = plt.subplot(4,1,1 )
        
plt.plot(pts, np.real(var[pts]), 'b', label='zcos')
plt.plot(pts, np.imag(var[pts]), 'g', label='zsin' )
plt.legend(loc='best')
ax = plt.subplot(4,2,3 )
plt.plot(pts, X, 'r', label='lon')
plt.legend(loc='best')
ax = plt.subplot(4,2,4 )
plt.plot(pts, Y, 'g', label='lat')
plt.legend(loc='best')


amp = np.zeros(np.shape(nav_lat)) * np.nan


#for i in range(xb):
    #amp[ nbjdta[0,i],nbidta[0,i] ] = np.abs(z_cos[0,i] + 1j*z_sin[0,i])

#amp = np.ma.masked_where( np.isnan(amp), amp)

ax = plt.subplot(2,1,2)
#plt.pcolormesh(nav_lon,nav_lat,amp)
plt.scatter(X,Y,c=Z, s=30, alpha=1, edgecolors='face')
#plt.clim([0,2.5])
plt.xlim([nav_lon.min()-0.1, nav_lon.max()+0.1])
plt.ylim([nav_lat.min()-0.1, nav_lat.max()+0.1])
plt.title(source+': abs(z)')
plt.xlabel('lon'); plt.ylabel('lat')
plt.colorbar(extend='max')    


## OUTPUT FIGURES
print 'Output filename: {}'.format(fname)
#plt.show()
plt.savefig(fname, dpi=100)
plt.close()

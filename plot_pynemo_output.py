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
import socket
if str('livmap') in socket.gethostname().lower():
    dirname = ''
elif str('livljobs') in socket.gethostname().lower():
    dirname = ''
elif str('eslogin') in socket.gethostname().lower():
    dirname = ''
elif str('mola') in socket.gethostname().lower():
    dirname = '/Volumes/work/jelt/NEMO/Solent/INPUTS/'
else:
    print 'There is no working ELSE option'

#################### USER PARAMETERS ##########################


source = 'FES'
source = 'TPXO'

con = 'M2'
filename = 'Solent_bdytide_rotT_'+con+'_grid_T.nc'

f = Dataset(dirname+source+'/'+filename)

z_cos = f.variables['z1'][:] # yb,xb
z_sin = f.variables['z2'][:] # yb,xb

nbidta = f.variables['nbidta'][:] # yb,xb
nbjdta = f.variables['nbjdta'][:] # yb,xb

nav_lat = f.variables['nav_lat'][:] # y,x
nav_lon = f.variables['nav_lon'][:] # y,x


yb, xb = np.shape(z_cos)

#fig = plt.figure( figsize = ( 8.4, 2+10*(Latmax-Latmin)/(Lonmax-Lonmin) ) )
fig = plt.figure( figsize = ( 8.4, 2+10) )

## DEFINE AND PLOT BASEMAP PROJECTION
ax = plt.subplot(2,2,1 )
        
plt.plot(np.arange(xb), z_cos[0,:], 'b', label='zcos')
plt.plot(np.arange(xb), z_sin[0,:], 'g', label='zsin' )
plt.legend(loc='best')
ax = plt.subplot(2,2,2 )
plt.plot(np.arange(xb), nbidta[0,:], 'r', label='nbidta')
plt.plot(np.arange(xb), nbjdta[0,:], 'g', label='nbjdta')
plt.legend(loc='best')


amp = np.zeros(np.shape(nav_lat)) * np.nan


for i in range(xb):
    amp[ nbjdta[0,i],nbidta[0,i] ] = np.abs(z_cos[0,i] + 1j*z_sin[0,i])
    amp[ nbjdta[0,i],nbidta[0,i] ] = i
    

amp = np.ma.masked_where( np.isnan(amp), amp)
ax = plt.subplot(2,1,2)
plt.pcolormesh(nav_lon,nav_lat,amp)
plt.clim([0,xb])
plt.xlim([-1.1, -0.9])
plt.ylim([50.5,50.6])
plt.title('abs(z)')
plt.xlabel('lon'); plt.ylabel('lat')
plt.colorbar()    

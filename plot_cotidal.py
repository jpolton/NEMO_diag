#%plot_cotidal.py
#
# plot_cotidal.py
#
# jp 14 June 2018
"""
** Summary: **
Plot pretty cotidal chart
** origin: ** cotidalchart_NEMO.ipynb
** Author: ** JP 31 Jan 2017
** Changelog: **
* 31 Jan 2017: Start with AMM60_MASSMO_fronts.ipynb
* 24 Mar 2017: Copied from AMM60_SSH_anim.py
* 22 Oct 2017: Modified for SE Asia
* 12 Jun 2018: Modified for Solent
* 14 Jun 2018: NB - extend with zoom and time series - things about zoom box are actually hard coded; no basmap projection used for it
* 16 Jun 2018: convert back to non animated co-tidal charts

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
from AMM60_tools import NEMO_fancy_datestr # NEMO date strings
from mpl_toolkits.mplot3d import Axes3D # 3d plotting
from itertools import product, combinations # plotting 3d frame
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
    dirname = '/Users/jeff/Desktop/'
#    dirname = '/Users/jeff/Desktop/OneWeekExpiry/tmp/'
elif str('livljobs') in socket.gethostname().lower():
    dirname = '/scratch/jelt/tmp/'
elif str('eslogin') in socket.gethostname().lower():
    dirname = '/work/n01/n01/jelt/Solent_surge/dev_r8814_surge_modelling_Nemo4/CONFIG/Solent_surge/EXP_TPXO/'
else:
    print 'There is no working ELSE option'

#################### USR PARAMETERS ##########################

filename = 'Solent_surge_TPXO_5mi_20130101_20130101_5min.nc'
levs     = np.arange(-2.5,2.5+0.1,0.1)
ofile    = 'FIGURES/Solent_SSH.gif'


              
source = 'AMM60'
               
dirname = '/Users/jeff/DATA/pycnmix/jelt/AMM60/'
filename = 'cutdown_AMM60_1d_20120731_20120829_D2_Tides.nc'               
               
conlist = ['M2','S2','K2'] # constituent list

#################### INTERNAL FCTNS #########################



def plot_amp_pha( pj, X, Y, amp, pha, levs, label) :

    xx,yy = pj( X, Y )           ## Projecton Basemap

    ## Shade basic plot. Create colorbar and nan color
    cmap0 = plt.cm.get_cmap('Spectral_r', 256)
    cmap0.set_bad('#9b9b9b', 1.0)
    cset  = pj.contourf( xx, yy, amp, cmap = cmap0 )
#    cset  = pj.contourf( xx, yy, amp, levels=levs, cmap = cmap0 )
    cset.set_clim([levs[0],levs[-1]])

    ## contour phase
    hh = pj.contour(xx, yy, pha, levels=np.arange(0,360,30), colors='k', \
                      zorder=10)
    plt.clabel(hh, hh.levels, fmt='%d', inline=True, fontsize=10)

    ## Print title
    #plt.text( -1.19, 50.87, label, color='k', fontsize=13, fontproperties=font  )
    plt.title(label, color='k', fontsize=13, fontproperties=font  )

 
    ## Colorbar
    cax  = plt.axes([0.2, 0.08, 0.6, 0.02])
    cbar = plt.colorbar( cset, cax=cax, orientation='horizontal', extend='both'  )
    cbar.ax.tick_params( labelsize=12 )

###################### LOAD DATA ############################

f = Dataset( dirname+filename )
#print f.variables

## Load lat and lon
nav_lat = f.variables['nav_lat_grid_T'][:] # (y,x)
nav_lon = f.variables['nav_lon_grid_T'][:] # (y,x)
Lonmin = np.min( nav_lon ); Lonmax = np.max(nav_lon) ## min and max lon
Latmin = np.min( nav_lat ); Latmax = np.max(nav_lat) ## min and max lat






###################### CORE CODE ############################

## Now do the main routine stuff
if __name__ == '__main__':

    count   = 1
    X_arr   = nav_lon
    Y_arr   = nav_lat
    files   = []
    
    for con in conlist:
        print con
        label = source + ':'+con+' co-tidal chart'
        
        ssh  = f.variables[con+'x_SSH'][:] + 1j*f.variables[con+'y_SSH'][:] #(y, x)
            
        ## Mask the array
        ssh[ssh==0] = np.nan
        mask = (np.isnan(np.real(ssh)))
        ssh  = np.ma.array( ssh , mask = mask )
        
        ## Convert to amplitude and phase (degrees)
        ssh_amp = np.abs(ssh)
        ssh_pha = np.angle(ssh, deg=True)
            
        # DEFINE FIGURES AND SUBGRID
        fig = plt.figure( figsize = ( 8.4, 2+10*(Latmax-Latmin)/(Lonmax-Lonmin) ) )
        gs  = gridspec.GridSpec( 1, 1 ); gs.update( wspace=0.2, hspace=0.1, top = 0.78, bottom = 0.15, left = 0.01, right = 0.99 )
        
        ## DEFINE AND PLOT BASEMAP PROJECTION
        ax = plt.subplot( gs[0] )
        pj = Basemap( projection='cyl', llcrnrlat=Latmin, urcrnrlat=Latmax, \
                                        llcrnrlon=Lonmin, urcrnrlon=Lonmax, resolution='c' )
        #pj.drawparallels(np.arange(50.,51.5,0.1)); pj.drawmeridians(np.arange(-2.,0.,0.1))
        
        plot_amp_pha( pj, X_arr, Y_arr, ssh_amp, ssh_pha, levs, label )
        
        plt.title('amplitude (m)', fontsize=12, fontproperties=font )
            
        
        ## OUTPUT FIGURES
        fname = "./FIGURES/"+filename.replace('.nc','_'+con+'.png')
        print label,fname
        #plt.show()
        plt.savefig(fname, dpi=100)
        plt.close()




    #for f in files:
    #    os.remove(f)

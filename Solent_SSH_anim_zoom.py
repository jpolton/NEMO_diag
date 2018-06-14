#%load Solent_SSH_anim.py
#
# Solent_SSH_anim.py
#
# jp 12 June 2018
"""
** Summary: **
Build 2D animation of Solent SSH with velocity
Load in some existing output Solent_surge_5mi_20130101_20130101_5min.nc. Plot it. Animate. 
** origin: ** cotidalchart_NEMO.ipynb
** Author: ** JP 31 Jan 2017
** Changelog: **
* 31 Jan 2017: Start with AMM60_MASSMO_fronts.ipynb
* 24 Mar 2017: Copied from AMM60_SSH_anim.py
* 22 Oct 2017: Modified for SE Asia
* 12 Jun 2018: Modified for Solent
* 14 Jun 2019: NB - extend with zoom and time series - things about zoom box are actually hard coded; no basmap projection used for it

** Issues: **
* The longitude label is missing from the final plots
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
if str('livmaf') in socket.gethostname():
    dirname = '/Volumes/archer/jelt/NEMO/NEMOGCM_jdha/dev_r4621_NOC4_BDY_VERT_INTERP/NEMOGCM/CONFIG/XIOS_AMM60_nemo_harmIT2/EXP_harmIT2/OUTPUT/'
#    dirname = '/Users/jeff/Desktop/OneWeekExpiry/tmp/'
elif str('livljobs') in socket.gethostname():
    dirname = '/scratch/jelt/tmp/'
elif str('eslogin') in socket.gethostname():
    dirname = '/Volumes/archer/jelt/NEMO/NEMOGCM_jdha/dev_r4621_NOC4_BDY_VERT_INTERP/NEMOGCM/CONFIG/XIOS_AMM60_nemo_harmIT2/EXP_harmIT2/OUTPUT/'
else:
    print 'There is no working ELSE option'

#################### USR PARAMETERS ##########################

dirname  = '/work/n01/n01/jelt/Solent_surge/dev_r8814_surge_modelling_Nemo4/CONFIG/Solent_surge/EXP_TPXO/' 
filename = 'Solent_surge_TPXO_5mi_20130101_20130101_5min.nc' 
levs     = np.arange(-2.5,2.5+0.1,0.1)
ofile    = 'FIGURES/Solent_SSH.gif'

index_Buoys = [ [-200,700], [200,200], ]  ## Index iLon, iLat of each gauge point to extract time series
out_every   = 1                          ## Generate graphic every X time step
init_step   = 2                           ## which iteration to start outputting
end_step    = 5                           ## stop output at nt-end_step    
color_Buoys = [ '#8d349b', 'b', 'g', 'r' ]

#################### INTERNAL FCTNS #########################

def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'%(delay,loop," ".join(files),output))

def sliceplotshade( pj, X, Y, ssh, u, v, levs, tdate, \
                    streamline = False, maxcurrent = None, \
                    vectorevery = 50, vector_zoom_every = 2 ) :

    xx,yy = pj( X, Y )           ## Projecton Basemap
    speed = np.sqrt(u**2+v**2)   ## Compute   Currents
 
    ## Create colorbar and nan color
    cmap0 = plt.cm.get_cmap('Spectral_r', 256)
    cmap0.set_bad('#9b9b9b', 1.0) 
    cset  = pj.pcolormesh( xx, yy, ssh, cmap = cmap0 )
    cset.set_clim([levs[0],levs[-1]])

    ## Streamline or vector
    if streamline :
       if maxcurrent is None : maxcurrent = np.nanmax(speed)
       lw = 4. * speed / maxcurrent; lw[(np.isnan(lw))] = 0.  ## Compute width for streamline
       pj.streamplot( xx, yy, u[:,:], v[:,:], color='k', density=2.0, linewidth=lw[count,:,:], arrowsize=1., zorder=10 )
    else :
       dumx = xx[::vectorevery, ::vectorevery]
       dumy = yy[::vectorevery, ::vectorevery]
       dumu = u [::vectorevery, ::vectorevery]
       dumv = v [::vectorevery, ::vectorevery]
       dums = speed[ ::vectorevery, ::vectorevery ]
       iLon = int(1400/vectorevery)
       iLat = int( 450/vectorevery)
       plt.text( dumx[iLat,iLon]-0.005, dumy[iLat,iLon]+0.01, '1m/s', fontsize=13, fontproperties=font )
       dumu[iLat,iLon] = 1.
       dumv[iLat,iLon] = 0.
       dums[iLat,iLon] = 1.
       ind = ( dums > 0. )
       pj.quiver(  dumx[ind], dumy[ind], \
                   dumu[ind], dumv[ind], \
                   color='k' , scale=30, alpha=0.7, zorder=10)

    ## Print time
    plt.text( -1.19, 50.87, tdate, color='w', fontsize=13, fontproperties=font  )

    ## Define box for zoom and plot it
    boxlatmin = 50.695; boxlatmax = 50.713
    boxlonmin = -1.552; boxlonmax = -1.530
    ratio  = (boxlatmax-boxlatmin)/(boxlonmax-boxlonmin)
    plt.plot( [boxlonmin,boxlonmax,boxlonmax,boxlonmin,boxlonmin], \
              [boxlatmin,boxlatmin,boxlatmax,boxlatmax,boxlatmin], 'k-',color='#a80000' )

    ## Zoom Box - No projection maps used here
    zax = plt.axes( [0.02, 0.55, 0.26, 0.26*ratio], zorder=99 )
    ind = ( speed[ ::vector_zoom_every, ::vector_zoom_every ] > 0. )
    cmap1 = plt.cm.get_cmap('Spectral_r', 256)
    cmap1.set_bad('#ffffff', 1.0)
    zax.contourf( X, Y, ssh[:,:], levs, cmap=cmap1, zorder=100 )
    zax.quiver(  X[ ::vector_zoom_every, ::vector_zoom_every ][ind], Y[ ::vector_zoom_every, ::vector_zoom_every ][ind], \
                 u[ ::vector_zoom_every, ::vector_zoom_every ][ind], v[ ::vector_zoom_every, ::vector_zoom_every ][ind], \
                 color='k' , scale=30, alpha=0.9, zorder=101 ) 
    zax.set_xlim([boxlonmin,boxlonmax]); zax.set_ylim([boxlatmin,boxlatmax])
    zax.tick_params(axis='x',labelbottom='off', bottom=False, top  =False )
    zax.tick_params(axis='y',labelleft  ='off', left  =False, right=False )

    ## Colorbar 
    cax  = plt.axes([0.2, 0.08, 0.6, 0.02])
    cbar = plt.colorbar( cset, cax=cax, orientation='horizontal', extend='both'  )
    cbar.ax.tick_params( labelsize=12 )

###################### LOAD DATA ############################

f = Dataset( dirname+filename )
ssh  = f.variables['zos' ][:] # ssh   (t, y, x)
ubar = f.variables['ubar'][:] # 2DH u (t, y, x)
vbar = f.variables['vbar'][:] # 2DH v (t, y, x)

## Mask the array
ssh[ssh==0] = np.nan
mask = (np.isnan(ssh))
ssh  = np.ma.array( ssh , mask = mask )
ubar = np.ma.array( ubar, mask = mask )
vbar = np.ma.array( vbar, mask = mask )

## Subsampled every X time step (faster)
ssh  = ssh [ init_step::out_every ]
mask = mask[ init_step::out_every ]
ubar = ubar[ init_step::out_every ] 
vbar = vbar[ init_step::out_every ]
spd  = np.sqrt(ubar**2+vbar**2) ## Compute norm currents
print "MAX SPEED", np.nanmax( spd )

## Load time calendar, unit and values
time_counter  = f.variables['time_counter'][init_step::out_every] # vector
time_origin   = f.variables['time_counter'].time_origin
time_calendar = f.variables['time_counter'].calendar
time_units    = f.variables['time_counter'].units
# Note that the Error flag doesn't work and I haven't actually checked it. What happens with leap years etc...
[time_str, time_datetime, flag_err] = NEMO_fancy_datestr( time_counter, time_origin ) # internal tide data
dats = []
for i in range( len(time_datetime) ) :
    datsec = (time_datetime[i] - time_datetime[0]).total_seconds()
    dats.append( datsec/3600. )
dats = np.array( dats )

## Load lat and lon
nav_lat = f.variables['nav_lat_grid_T'][:] # (y,x)
nav_lon = f.variables['nav_lon_grid_T'][:] # (y,x)
Lonmin = np.min( nav_lon ); Lonmax = np.max(nav_lon) ## min and max lon
Latmin = np.min( nav_lat ); Latmax = np.max(nav_lat) ## min and max lat

###################### CORE CODE ############################

## Now do the main routine stuff
if __name__ == '__main__':

    count   = 1
    [nt,ny,nx] = np.shape( ssh )
    X_arr   = nav_lon
    Y_arr   = nav_lat
    var_arr = ssh[:,:,:] # Note this also has time dimension
    files   = []

    print 'Hardwired timestep in variable dat'
    ## Time timeseries of frames
    #for count in range(nt-25,nt):
    print time_datetime[0], time_datetime[nt-end_step-1]
    print dats[0], dats[nt-end_step-1]
    for count in range(0,nt-end_step):  # By eye this looks like a closed cycle
        #dat = datetime.datetime.strftime(time_datetime[count], '%d %b %Y: %H:%M')
        datsec = (time_datetime[count] - time_datetime[0]).total_seconds() 
        dath = int( datsec/3600 )
        datm = int( (datsec - dath * 3600) / 60 )
        dat  = "{0:02d}:{1:02d}".format( dath, datm ) 

        # DEFINE FIGURES AND SUBGRID
        fig = plt.figure( figsize = ( 8.4, 2+10*(Latmax-Latmin)/(Lonmax-Lonmin) ) )
        gs  = gridspec.GridSpec( 1, 1 ); gs.update( wspace=0.2, hspace=0.1, top = 0.78, bottom = 0.15, left = 0.01, right = 0.99 )

        ## DEFINE AND PLOT BASEMAP PROJECTION
        ax = plt.subplot( gs[0] )
        pj = Basemap( projection='cyl', llcrnrlat=Latmin, urcrnrlat=Latmax, \
                                        llcrnrlon=Lonmin, urcrnrlon=Lonmax, resolution='c' )
        pj.drawparallels(np.arange(50.,51.5,0.1)); pj.drawmeridians(np.arange(-2.,0.,0.1))

        sliceplotshade( pj, X_arr, Y_arr, ssh[count], ubar[count], vbar[count], levs, str(dat), \
                        streamline = False, maxcurrent = np.nanmax(spd), \
                        vectorevery = 50, vector_zoom_every = 2 ) 

        plt.title('Sea Surface Height (m)', fontsize=12, fontproperties=font )

        ## PLOT GAUGE
        for iB, Buoy in enumerate( index_Buoys ) :    
            ax.plot( X_arr[ Buoy[1], Buoy[0] ], Y_arr[ Buoy[1], Buoy[0] ], \
                     'bo', color=color_Buoys[iB], markersize=10, zorder=100, alpha=0.9 )

        ## DEFINE TIME SERIE FIGURES
        ax1 = plt.axes([0.12, 0.82, 0.87, 0.16])
        for iB, Buoy in enumerate( index_Buoys ) :
            ax1.plot( dats, ssh[:, Buoy[1], Buoy[0] ], 'b-', color=color_Buoys[iB], linewidth = 2 )        
        ax1.plot( [dats[count],dats[count]],[-3,3.45], 'k-', linewidth = 3 )
        ax1.set_ylabel( "SSH (m)", fontsize=12, fontproperties=font )
#        ax1.set_xlim( [time_datetime[1],time_datetime[nt-end_step-1]] )
        ax1.set_xlim( [ 0,dats[nt-end_step-1] ] )
        ax1.set_ylim( [-3,3.45] )
        ax1.grid(True)

        ## OUTPUT FIGURES
        fname = "./FIGURES/"+filename.replace('.nc','_'+str(count).zfill(4)+'.png')
        print str(dat),fname
        plt.savefig(fname, dpi=100)
        plt.close()
        
        files.append(fname)

    # Make the animated gif and clean up the files
    make_gif(files,ofile,delay=20)

    #for f in files:
    #    os.remove(f)

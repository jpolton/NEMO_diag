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

** Issues: **
* The longitude label is missing from the final plots
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import sys
sys.path.append('../jcomp_tools_dev/') # Add the directory with the amm60_data_tools.py file to path. Not used on the NOCL SAN
from AMM60_tools import NEMO_fancy_datestr # NEMO date strings
from mpl_toolkits.mplot3d import Axes3D # 3d plotting
from itertools import product, combinations # plotting 3d frame
import os #, sys  # removing files and building animation
import datetime
##%matplotlib inline

## Check host name and locate file path
import socket
if str('livmap') in socket.gethostname().lower():
    dirname = '/Users/jeff/Desktop/'
#    dirname = '/Users/jeff/Desktop/OneWeekExpiry/tmp/'
elif str('livljobs') in socket.gethostname().lower():
    dirname = '/scratch/jelt/tmp/'
elif str('eslogin') in socket.gethostname().lower():
    dirname = '/work/n01/n01/jelt/Solent_surge/dev_r8814_surge_modelling_Nemo4/CONFIG/Solent_surge/EXP00/'
else:
    print 'There is no working ELSE option'


# Define some variables for the configuration
config = 'Solent'
filename = 'Solent_surge_5mi_20130101_20130101_5min.nc'
key_variable = 'speed'
xlim = [-1.67, -0.99]
ylim = [50.53,50.92]

if key_variable == 'zos':
    ofile = 'FIGURES/'+config+'_SSH.gif'
    levs = np.arange(-2,2+0.1,0.1)
elif key_variable == 'speed':
    ofile = 'FIGURES/'+config+'_speed.gif'
    levs = np.arange(-3,3+0.5,0.5)
    spacing = 150 # grid spacing between vectors
    speed_min = 0.1 # Min speed to show vector
else:
    print "Not ready for that eventuality"



f = Dataset(dirname+filename)

## Load in data
zos = f.variables['zos'][:] # (t, y, x)
u = f.variables['ubar'][:]
v = f.variables['vbar'][:]

[nt,ny,nx] = np.shape(zos)

# Compute speed and corresponding line width
speed = np.sqrt(u**2+v**2)
lw = 4. * speed / np.nanmax(speed); lw[(np.isnan(lw))] = 0.

# Scale quiver vectors
sU = u / speed
sV = v / speed

# Mask variables
mask = ((np.sum(zos[0:3,:,:], axis=0)) == 0) # water=True, land=False
zos = np.ma.masked_where(np.tile(mask,(nt,1,1)), zos)
speed = np.ma.masked_where(np.tile(mask,(nt,1,1)), speed)
sU = np.ma.masked_where( speed < speed_min, sU)
sV = np.ma.masked_where( speed < speed_min, sV)

#mask = (np.isnan(zos))
#zos = np.ma.array( zos, mask = mask )



# load in time
time_counter = f.variables['time_counter'][:] # vector
time_origin = f.variables['time_counter'].time_origin
time_calendar = f.variables['time_counter'].calendar
time_units = f.variables['time_counter'].units

#load lat and lon
nav_lat = f.variables['nav_lat_grid_T'][:] # (y,x)
nav_lon = f.variables['nav_lon_grid_T'][:] # (y,x)


# Process the time data
################################
# Note that the Error flag doesn't work and I haven't actually checked it. What happens with leap years etc...
[time_str, time_datetime, flag_err] = NEMO_fancy_datestr( time_counter, time_origin ) # internal tide data



def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'%(delay,loop," ".join(files),output))

def sliceplotshade(X,Y,var,sU,sV,count,xlim,ylim,levs,spacing):

## INSERTED CODE HERE

    #clim = [-10**-6,10**-6]
    #cz = plt.contour( X, Y, var, levels=[-2, -1, 0, 1 ,2],
    #            color='k',
    #            linewidth=2)
    cset = ax.contourf( X, Y, var[count,:,:],
                 levels=levs,
                 extend='both',
                 cmap='Spectral')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('\nlongitude (deg)')
    ax.set_ylabel('\nlatitude (deg)')
    cset.set_clim([levs[0],levs[-1]])
    ax.hold(True)

    fig.colorbar(cset) # Add colorbar


    plt.quiver(  X[::spacing,::spacing],
                 Y[::spacing,::spacing],
                 sU[count,::spacing,::spacing],
                 sV[count,::spacing,::spacing],
           units='xy', scale_units='width',scale=40)

    adjustFigAspect(fig,aspect=(xlim[1]-xlim[0])/(ylim[1]-ylim[0]))



    #lines = [ cz.collections[1], cz.collections[-1] ]
    #labels = ['200m','800m']

    #plt.legend( lines, labels, loc="lower right")

    #plt.title('Upper 200m depth mean current (m/s)')
    #plt.xlabel('longitude'); plt.ylabel('latitude')





def sliceplotshadestream(X,Y,var,u,v,lw,count,xlim,ylim,levs):

    cset = ax.pcolormesh(X, Y, var[count,:,:], cmap='Spectral')


    ax.streamplot( X,Y, u[count,:,:], v[count,:,:], color='k', density=2.0, linewidth=lw[count,:,:], arrowsize=1., zorder=10 )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('\nlongitude (deg)')
    ax.set_ylabel('\nlatitude (deg)')
    cset.set_clim([levs[0],levs[-1]])
    ax.hold(True)
    #cs = ax.contour(X, Y, var[count,:,:], levels=levs, colors='k')

    fig.colorbar(cset) # Add colorbar
    adjustFigAspect(fig,aspect=(xlim[1]-xlim[0])/(ylim[1]-ylim[0]))


def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


## Now do the main routine stuff
if __name__ == '__main__':

    count = 1


    X_arr = nav_lon
    Y_arr = nav_lat

    files = []

    ## Time timeseries of frames
    #for count in range(3):
    for count in range(nt-8):  # By eye this looks like a closed cycle
        print count

        # Plot SST map
        plt.close('all')
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        #sshplot(X_arr,Y_arr,var_arr,count)
        dat = datetime.datetime.strftime(time_datetime[count], '%d %b %Y: %H:%M')
        if key_variable == 'zos':
            sliceplotshadestream(X_arr,Y_arr,zos,u,v,lw,count,xlim,ylim,levs)
            plt.title(config+': Sea Surface Height (m) '+str(dat))
        elif key_variable == 'speed':
            sliceplotshade(X_arr,Y_arr,speed,sU,sV,count,xlim,ylim,levs,spacing)
            plt.title(config+': Tidal Speed (m/s) '+str(dat))
        #dat = 'hrs: '+str(count)
        #dat = 'hour: '+str(count/10)
        ax.text(80, 18, str(dat), fontsize=10)
        plt.axis("on")
        #plt.show() # Can not save to file with this command present
        fname = filename.replace('.nc','_'+str(count).zfill(4)+'.png')

        plt.savefig(fname, dpi=100)
        files.append(fname)


    # Make the animated gif and clean up the files
    make_gif(files,ofile,delay=20)

    for f in files:
        os.remove(f)

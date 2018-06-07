# diff_slice_anim.py
# 
# jp 5 June 2018
"""
** Summary: **
Build 2D animation of two 2D arrays and their difference.
E.g. SSH_1, SSH_2 and SSH_1 - SSH_2

** Useage: **
Aspirational usage (maybe) diff_slice_anim.py ifile1 ifile2 var ofile

** origin: ** AMM60_SSH_anim.py

** Author: ** JP 5 Jun 2018

** Changelog: **
* 5  Jun 2018: Start with AMM60_SSH_anim.py


** Issues: **
* Automate the colour / contour scales 
* Generalise the slice extraction. E.g. over x,y,z not just time.

"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.cm as cm  # colormaps
import sys
#sys.path.append('../jcomp_tools_dev/') # Add the directory with the amm60_data_tools.py file to path. Not used on the NOCL SAN
from AMM60_tools import NEMO_fancy_datestr # NEMO date strings
import os #, sys  # removing files and building animation
import datetime
##%matplotlib inline


## Read in the input parameters

#vtype = 'ssh'
#vtype = 'sst'
#vtype = 'ssu'
vtype = 'ssv'

#ifile1 = sys.argv[1]
#ifile2 = sys.argv[2]
#rootdir = "/Users/jeff/Desktop/OneWeekExpiry/SEAsia/"
#rootdir = "/work/n01/n01/jdha/2018/SEAsia/EXP_openbcs/"
rootdir = "/scratch/jdha/SEAsia/"
#ifile1 = "OUTPUT_SPLIT_NONE/SEAsia_1d_19791101_19791130_grid_T.nc"
#ifile2 = "OUTPUT_FULL2_NONE/SEAsia_1d_19791101_19791130_grid_T.nc"

EXPa = "EXP4"
EXPb = "EXP5"

freq = '5d'
if vtype == 'ssh':
        grid = 'grid_T'
        freq = '1d'
	variable = 'zos'
	levs = np.arange(-2,2+0.2,0.2)
	dlevs = np.arange(-0.1,0.1+0.01,0.01)
elif vtype == 'sst':
        grid = 'grid_T'
        variable = 'toce'
	levs = np.arange(20,30+1,1)
	dlevs = np.arange(-1,1+0.1,0.1)
elif vtype == 'ssu':
	grid = 'grid_U'
        variable = 'uoce'
	levs = np.arange(-2,2+0.2,0.2)
	dlevs = np.arange(-0.1,0.1+0.01,0.01)
elif vtype == 'ssv':
	grid = 'grid_V'
        variable = 'voce'
	levs = np.arange(-2,2+0.2,0.2)
	dlevs = np.arange(-0.1,0.1+0.01,0.01)


 
mfile = "OUTPUT_EXP4/SEAsia_5d_19791101_19791130_grid_T.nc"
ifile1 = "OUTPUT_" + EXPa + "/SEAsia_" + freq + "_19791101_19791130_" + grid + ".nc"
ifile2 = "OUTPUT_" + EXPb + "/SEAsia_" + freq + "_19791101_19791130_" + grid + ".nc"

ofile = 'FIGURES/SEAsia_' + EXPa + '_' + EXPb + '_' + vtype + '_slice.gif'

#levs = [-1.5,-1,-0.5,0,0.5,1,1.5]
    
#variable = 'zos'
#ofile = 'FIGURES/SEAsia_EXP4_EXP5_SSH_slice.gif'
#levs = np.arange(-2,2+0.1,0.1)


f0 = Dataset(rootdir + mfile)
f1 = Dataset(rootdir + ifile1)
f2 = Dataset(rootdir + ifile2)

xlim = [75,135]
ylim = [-20.,20.]

## Load in data
mvar = f0.variables['toce'][0,0,:,:] 
var1 = f1.variables[variable][:] # Size? e.g. (t, y, x)  
var2 = f2.variables[variable][:] # Size? e.g. (t, y, x)  

## Deduce the number of dimensions in var and take surface if spatially 3D
if len(np.shape(var1)) == 4:
	var1 = np.squeeze(var1[:,0,:,:])
	var2 = np.squeeze(var2[:,0,:,:])

[nt,ny,nx] = np.shape(var1)

# mask land
# to do

print np.min(var1)
print np.max(var1)

# load in time
time_counter = f1.variables['time_counter'][:] # vector
time_origin = f1.variables['time_counter'].time_origin
time_calendar = f1.variables['time_counter'].calendar
time_units = f1.variables['time_counter'].units

#load lat and lon
nav_lat = f1.variables['nav_lat'][:] # (y,x)
nav_lon = f1.variables['nav_lon'][:] # (y,x)


# Process the time data
################################
# Note that the Error flag doesn't work and I haven't actually checked it. What happens with leap years etc...
[time_str, time_datetime, flag_err] = NEMO_fancy_datestr( time_counter, time_origin )



def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """
     
    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'%(delay,loop," ".join(files),output))
    

def sliceplotshade(X,Y,var,count,xlim,ylim,levs):
    cmap = cm.Spectral_r
    cmap.set_bad('white',1.)
    cmap.set_over('red',1.)
    cset = ax.pcolormesh(X, Y, var[count,:,:], cmap=cmap)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #ax.set_xlabel('\nlongitude (deg)')
    #ax.set_ylabel('\nlatitude (deg)')
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
    ylim = .4*minsize/ysize * 3 # factor of 3 because of 3 panels in fig 
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

    ## Step through frames
    for count in range(nt-1,nt):
        print count

        plt.close('all')
        fig = plt.figure(figsize=(10,20))
        
        # Plot var1 
        plt.subplot(311)
        ax = fig.gca()
        sliceplotshade(X_arr,Y_arr,var1,count,xlim,ylim,levs)   
        dat = datetime.datetime.strftime(time_datetime[count], '%d %b %Y: %H:%M')
        plt.title('a) '+ifile1)
        ax.text(80, 18, str(dat), fontsize=10)
        plt.axis("on")
        #ax.set_xlabel('\nlongitude (deg)')
        ax.set_ylabel('\nlatitude (deg)')
        
        # Plot var2 
        plt.subplot(312)
        ax = fig.gca()
        sliceplotshade(X_arr,Y_arr,var2,count,xlim,ylim,levs)   
        dat = datetime.datetime.strftime(time_datetime[count], '%d %b %Y: %H:%M')
        plt.title('b) '+ifile2)
        ax.text(80, 18, str(dat), fontsize=10)
        plt.axis("on")
        #ax.set_xlabel('\nlongitude (deg)')
        ax.set_ylabel('\nlatitude (deg)')        

        # Plot var1 - var2 
        plt.subplot(313)
        ax = fig.gca()
        sliceplotshade(X_arr,Y_arr,var1-var2,count,xlim,ylim,dlevs)   
        dat = datetime.datetime.strftime(time_datetime[count], '%d %b %Y: %H:%M')
        plt.title('(a)-(b)')
        ax.text(80, 18, str(dat), fontsize=10)
        plt.axis("on")
        ax.set_xlabel('\nlongitude (deg)')
        ax.set_ylabel('\nlatitude (deg)')
       
 
        #plt.show() # Can not save to file with this command present
        #fname = rootdir+ifile1.replace('.nc','_'+str(count).zfill(4)+'.png')
        fname = ofile.replace('.gif','_'+str(count).zfill(4)+'.png')

        plt.savefig(fname, dpi=100)
        files.append(fname)


    # Make the animated gif and clean up the files
    make_gif(files,ofile,delay=20)

    for f in files:
        os.remove(f)

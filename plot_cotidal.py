#!/usr/bin/python
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
* Migrate to ShelfAssess with a config file

** Usage: ** plot_cotidal.py 'AMM60' 'M2'
"""

##################### LOAD MODULES ###########################

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.gridspec as gridspec
from  mpl_toolkits.basemap import Basemap
import sys
sys.path.append('../jcomp_tools_dev/') # Add the directory with the amm60_data_tools.py file to path. Not used on the NOCL SAN
##%matplotlib inline
from   matplotlib.font_manager import FontProperties
font0  = FontProperties(); font = font0.copy(); font.set_weight('bold'); font.set_size('large')

## Check host name and locate file path
import socket
if str('livmap') in socket.gethostname().lower():
    rootdir = '/Volumes'
#    dirname = '/Users/jeff/Desktop/OneWeekExpiry/tmp/'
elif str('livljobs') in socket.gethostname().lower():
    rootdir = ''
elif str('eslogin') in socket.gethostname().lower():
    rootdir = ''
elif str('mola') in socket.gethostname().lower() or str('tardis') in socket.gethostname().lower() :
    rootdir = '/Volumes'
else:
    print 'There is no working ELSE option'

#################### USR PARAMETERS ##########################

levs     = np.arange(0,2.5+0.1,0.1)

if len(sys.argv) >= 1:
    source = str(sys.argv[1])
else:
    source = 'TPXO'
    #source = 'TPXO'# 'TPXO' #'AMM60'  # 'FES2014'

if len(sys.argv) > 2:
    con = str(sys.argv[2])
else:
    con = 'M2'
    # conlist = ['M2'] # ['M2','S2','K2'] # constituent list


if source == 'AMM60':
	#dirname = '/Users/jeff/DATA/pycnmix/jelt/AMM60/'
	#filename = 'cutdown_AMM60_1d_20120731_20120829_D2_Tides.nc'
	dirname = rootdir+'/projectsa/pycnmix/jelt/AMM60/'
	filename = 'AMM60_1d_20120801_20120831_D2_Tides.nc'
	lon_var = 'nav_lon_grid_T'
	lat_var = 'nav_lat_grid_T'

if source == 'MASSMO5':
	dirname = rootdir+'/scratch/jelt/tmp/'
	#dirname = rootdir+'/work/n01/n01/jelt/MASSMO5_surge/dev_r8814_surge_modelling_Nemo4/CONFIG/MASSMO5_surge/EXP00/'
	filename = 'MASSMO5_surge_576001_1267200_harmonic_grid_T.nc'
	lon_var = 'nav_lon_grid_T'
	lat_var = 'nav_lat_grid_T'


elif source =='TPXO':
	dirname = rootdir+'/work/jelt/tpxo7.2/'
	filename = 'h_tpxo7.2.nc'
	lon_var = 'lon_z'
	lat_var = 'lat_z'

elif source =='FES2014':
    dirname = rootdir+'/projectsa/NEMO/Forcing/FES2014/ocean_tide_extrapolated/'
    filename = con.lower()+'.nc'
    lon_var = 'lon' # 1D
    lat_var = 'lat' # 1D



#################### INTERNAL FCTNS #########################



def plot_amp_pha( pj, X, Y, amp, pha, levs, label) :

    xx,yy = pj( X, Y )           ## Projecton Basemap

    ## Shade basic plot. Create colorbar and nan color
    cmap0 = plt.cm.get_cmap('Spectral_r', 256)
    cmap0.set_bad('#9b9b9b', 1.0)
    cset  = pj.pcolormesh( xx, yy, amp, cmap = cmap0 )
#    cset  = pj.contourf( xx, yy, amp, cmap = cmap0 )
#    cset  = pj.contourf( xx, yy, amp, levels=levs, cmap = cmap0 )
    cset.set_clim([levs[0],levs[-1]])

    ## contour phase
    hh = pj.contour(xx, yy, np.mod(pha,360), levels=np.arange(0,360,30), colors='k', \
                      zorder=10)
    plt.clabel(hh, hh.levels, fmt='%d', inline=True, fontsize=10)
    pj.drawcoastlines(linewidth = 0.2, zorder = 11)

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
nav_lat = f.variables[lat_var][:] # (y,x)
nav_lon = f.variables[lon_var][:] # (y,x)

if source == 'FES2014':
	# convert 1D axes into 2D arrays
 	nav_lon, nav_lat = np.meshgrid(nav_lon, nav_lat)

Lonmin = np.min( nav_lon ); Lonmax = np.max(nav_lon) ## min and max lon
Latmin = np.min( nav_lat ); Latmax = np.max(nav_lat) ## min and max lat

#print 'Lat:',Latmin, Latmax
#print 'Lon:',Lonmin, Lonmax

# set lat lon limits
print 'Solent view'

if source == 'AMM60':
	#Latmin = 50.5; Latmax = 50.9
	#Lonmin = -1.69; Lonmax = -1.00
	Lonmin = -3.69; Lonmax = -0.1
	Latmin = 48.9; Latmax = 51.9

elif source == 'MASSMO5':
	Lonmin = 2.7596083; Lonmax = 40.764503
	Latmin = 73.065506; Latmax = 79.086868

elif source == 'TPXO':
	Lonmin = 356.31; Lonmax = 359.9
	Latmin = 48.9; Latmax = 51.9

elif source == 'FES2014':
	Lonmin = 356.31; Lonmax = 359.9
	Latmin = 48.9; Latmax = 51.9




###################### CORE CODE ############################

## Now do the main routine stuff
if __name__ == '__main__':

    count   = 1
    X_arr   = nav_lon
    Y_arr   = nav_lat
    files   = []

    print 'Constituent {}'.format(con)
    label = source + ':'+con+' co-tidal chart'

    if source == 'AMM60':
        ssh  = f.variables[con+'x_SSH'][:] + 1j*f.variables[con+'y_SSH'][:] #(y, x)
        # Mask the array. Masking the lat and lon fields is not good for mst plotting functions but it seems to otherwise break the basemap projection...
        nav_lat[nav_lat==0] = np.nan
        mask = (np.isnan(nav_lat))
        #mask = (np.isnan(np.real(ssh)))
        #ssh  = np.ma.array( ssh , mask = mask )
        ssh  = np.ma.masked_where( ssh==0, ssh )
        nav_lon[mask] = np.nan

        ## Convert to amplitude and phase (degrees)
        ssh_amp = np.abs(ssh)
        ssh_pha = np.angle(ssh, deg=True)
        ssh_amp = np.ma.masked_where( ssh==0, ssh_amp )
        ssh_pha = np.ma.masked_where( ssh==0, ssh_pha )

    elif source == 'TPXO':
        tpxo_conlist = f.variables['con'][:]
        if con == 'M2':
            ssh  = f.variables['hRe'][0,:,:] + 1j*f.variables['hIm'][0,:,:] #(nc, y, x)
        elif con == 'S2':
            ssh  = f.variables['hRe'][1,:,:] + 1j*f.variables['hIm'][1,:,:] #(nc, y, x)
        else:
            print 'not ready for that constituent'


        ## Convert to amplitude and phase (degrees)
        ssh_amp = np.abs(ssh)
        ssh_pha = np.angle(ssh, deg=True)
        ssh_amp = np.ma.masked_where( ssh==0, ssh_amp )
        ssh_pha = np.ma.masked_where( ssh==0, ssh_pha )



    if source == 'MASSMO5':
        ssh_amp = f.variables[con+'amp_ssh'][:]
        ssh_pha = f.variables[con+'pha_ssh'][:]

    if source == 'FES2014':
        ssh_amp = f.variables['amplitude'][:]/100. # convert units to metres
        ssh_pha = f.variables['phase'][:]


    # DEFINE FIGURES AND SUBGRID
    fig = plt.figure( figsize = ( 8.4, 2+10*(Latmax-Latmin)/(Lonmax-Lonmin) ) )
    gs  = gridspec.GridSpec( 1, 1 ); gs.update( wspace=0.2, hspace=0.1, top = 0.78, bottom = 0.15, left = 0.01, right = 0.99 )

    ## DEFINE AND PLOT BASEMAP PROJECTION
    ax = plt.subplot( gs[0] )
    #pj = Basemap( projection='cyl', llcrnrlat=Latmin, urcrnrlat=Latmax, \
    pj = Basemap( projection='tmerc',lon_0=0.,lat_0=52., llcrnrlat=Latmin, urcrnrlat=Latmax, \
                                    llcrnrlon=Lonmin, urcrnrlon=Lonmax, resolution='f' )
    #pj.drawparallels(np.arange(50.,51.5,0.1)); pj.drawmeridians(np.arange(-2.,0.,0.1))

    plot_amp_pha( pj, X_arr, Y_arr, ssh_amp, ssh_pha, levs, label )


    plt.title('amplitude (m)', fontsize=12, fontproperties=font )


    ## OUTPUT FIGURE
    fname = "./FIGURES/"+filename.replace('.nc','_'+con+'.png')
    if source == 'FES2014':
        fname = "./FIGURES/"+source+'_'+filename.replace('.nc','_'+con+'.png')
    print label,fname
    #plt.show()
    plt.savefig(fname, dpi=100)
    plt.close()

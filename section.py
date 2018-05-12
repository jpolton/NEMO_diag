#
# jp 12 May 2018
#
# Section plot

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.cm as cm  # colormaps
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import sys # Exit command

import sys # sys.exit('Error message)
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.colors as mc   # fancy symetric colours on log scale 
import matplotlib.cm as cm  # colormaps
import seaborn as sns # cyclic colormap for harmonic phase


######%matplotlib inline


##############################################################################
## Define functions
##############################################################################

def extract_depthsection(lats2d, lons2d, dep3d, var3d, lats_endpts, lons_endpts, units='deg', npts='auto'):
    """
    For plotting depth-distance transects from 3D data.
    Constructs 2D lat/lon vs depth arrays for input lat,lon (2d), depth (3d) and variable (3d) fields.
    Uses nearest neighbour interpolation.
    
    author: jpolton
    
    changelog: 
        29 Nov 2017: start
        
    inputs: 
        lats2d - array of parent latitudes [y,x]
        lons2d - array of parent longitudes [y,x]
        dep3d  - array of parent depth data to subset [z,y,x]
        var3d  - array of parent variable data to subset [z,y,x]
        lats_endpts - lat endpoints of transect [lat0, lat1]
        lons_endpts - lon endpoints of transect [lon0, lon1]
        units - [deg/ind]. Choice of coordinate input format.
        npts - number of horizontal points to extract along the section. Default 'auto' uses parent grid resolution
        
    outputs:
        lats_sec - latitudes along transect [z,npts]
        lons_sec - longitudes along transect [z,npts]
        dep_sec - depths along transect [z,npts]
        var_sec - variable along transect [z,npts]
        
    Usage: 
        lats_endpts = [48,49]
        lons_endpts = [-10,-6]
        [lats_sec, lons_sec, dep_sec, var_sec] = extract_depthsection( nav_lat_grid_T, nav_lon_grid_T, gdepw, N2_3d[0,:,:,:], lats_endpts, lons_endpts )

        cmap = cm.Spectral_r
        cmap.set_over('red',1.)
        cmap.set_under('white',1.)

        fig = plt.figure()
        plt.rcParams['figure.figsize'] = (12.0, 15.0)
        ax = fig.add_subplot(111)
        plt.pcolormesh( lons_sec, np.log10(dep_sec), var_sec, cmap=cm.Spectral_r )
        plt.xlabel('longitude'); plt.xlim(lons_endpts)
        plt.ylabel('log10(depth [m])');plt.ylim([3,0])
        plt.clim([0,1E-3])
        plt.colorbar()

 
     Notes: 
        * This could be more efficient if it only output the transect indices for the the 2d horizontal grids but then
        it is quite fiddly to implement on the 3d arrays. It is easier to use if the output is processed arrays rather
        than sets of indices.
        
        * This could be upated to do linear interpolation instead of nearest neighbour...

        * units are not very elegantly handled: need lat and lon grid even if units = 'ind'
    """    
    
    [nz,ny,nx] = np.shape(dep3d)
    #print 'Input array size [nz,ny,nx]:{},{},{}'.format(nz,ny,nx)
    
    if np.shape(lats2d) != (ny,nx) or np.shape(lons2d) != (ny,nx):
        print 'Inputs arrays are not of compatible shapes. lons: {}, lons: {}, 3d array {}'.format(np.shape(lons2d), np.shape(lats2d), np.shape(z3d))
        
    if units=='ind':
        lons2d = np.tile(np.arange(0,nx),  (ny,1))
        lats2d = np.tile(np.arange(0,ny).T,(nx,1)).T

    # Find nearest neighbour indices (in degree space). Store in J,I arrays
    J=np.NaN*np.arange(2) # store lat indices
    I=np.NaN*np.arange(2) # store lon indices
    for count in range(2):
        dist2 = []
        dist2 = np.square(lats2d - lats_endpts[count]) + np.square(lons2d - lons_endpts[count])
        [J[count],I[count]] = np.unravel_index( dist2.argmin(), dist2.shape )
 
    # linearly interpolate between the end point indices
    if npts == 'auto': # Set Npts as parent grid resolution
        Npts = int(max( np.abs(J[1]-J[0]), np.abs(I[1]-I[0]) ))
    else:
        Npts = int(npts)
        
    #print 'Npts:{}'.format(Npts)
    
    JJ = [int(count) for count in np.round(np.linspace(J[0],J[1],num=Npts))]
    II = [int(count) for count in np.round(np.linspace(I[0],I[1],num=Npts))]

    indices2d = zip(JJ,II) # size [Npts,2]
    #print 'indices2d shape:{}'.format( np.shape(indices2d) ) 

    # Compute the variables along the section
    lats_sec = np.tile([lats2d[row] for row in indices2d],(nz,1))  # [nz,Npts]
    lons_sec = np.tile([lons2d[row] for row in indices2d],(nz,1))  # [nz,Npts]
    dep_sec  = np.transpose([dep3d[:,row[0],row[1]] for row in indices2d]) # [nz,Npts]
    var_sec  = np.transpose([var3d[:,row[0],row[1]] for row in indices2d]) # [nz,Npts]
    
    return [lats_sec, lons_sec, dep_sec, var_sec]


def e3_to_depth(pe3t, pe3w, jpk):
    '''
    funtion e3_to_depth
    Purpose :   compute t- & w-depths of model levels from e3t & e3w scale factors
    Method  :   The t- & w-depth are given by the summation of e3w & e3t, resp. 
    Action  :   pe3t, pe3w : scale factor of t- and w-point (m)
    '''

    pdepw      = np.zeros_like(pe3w)
    pdepw[0,:] = 0.
    pdept      = np.zeros_like(pe3t)
    pdept[0,:] = 0.5 * pe3w[0,:]

    for jk in np.arange(1,jpk,1):
        pdepw[jk,:] = pdepw[jk-1,:] + pe3t[jk-1,:]
        pdept[jk,:] = pdept[jk-1,:] + pe3w[jk  ,:]

    return pdept, pdepw


def plot_section_panel(X,Z,var,ylims,xlims,clims,cmap,tstr,ystr,ipanel):
    """
    Plot function to save space
    Useage:
    plot_section_panel(grid_sec1,bath_sec1,np.log10(var1_sec1),[200,10],[ min(lons1), max(lons1)],[-6,-3],cmap, lab1+':log10(N2)','depth (m)',411)
    """
    ax = fig.add_subplot(ipanel)
    plt.pcolormesh(X, Z, var, cmap=cmap)
    plt.ylim(ylims)
    plt.xlim(xlims)
    plt.title(tstr)
    plt.ylabel(ystr)
    plt.clim(clims)
    plt.colorbar()


##################################
## Now do the main routine stuff
if __name__ == '__main__':

 # directory
 dir = '/work/n01/n01/jelt/SEAsia/trunk_NEMOGCM_r8395/CONFIG/SEAsia/EXP_tide_initcd/'
 # Configuration File
 f_cfg = 'domain_cfg.nc'

 # Bathymetry File
 f_bth = 'bathy_meter.nc'

 # Variable File and name
 #f_var = 'output.abort.nc'
 f_var = 'output.init.nc'
 varstr1 = 'votemper'
 varstr2 = 'vosaline'



 # Define sections of interest: Amorican, Celtic onshore, Celtic offshore

 rootgrp = Dataset(dir+f_cfg, "r", format="NETCDF4")
 e3t_0   = np.squeeze(rootgrp.variables['e3t_0'][:])
 e3w_0   = np.squeeze(rootgrp.variables['e3w_0'][:])
 e3t_1d  = np.squeeze(rootgrp.variables['e3t_1d'][:])
 e3w_1d  = np.squeeze(rootgrp.variables['e3w_1d'][:])
 stiff   = np.squeeze(rootgrp.variables['stiffness'][:])
 m       = np.squeeze(rootgrp.variables['bottom_level'][:])

 mv      = np.copy(m)
 mv[0:-1]= np.minimum(m[0:-1],m[1:])

 jpk = len(e3t_0[:,0])
 #jpi = np.max(m.shape)

 #if len(i_coords)==1:
 # e3v_0   = np.squeeze(rootgrp.variables['e3v_0' ][:])
 # e3vw_0  = np.squeeze(rootgrp.variables['e3vw_0'][:])
 # coords  = j_coords
 #elif len(j_coords)==1:
 # e3v_0   = np.squeeze(rootgrp.variables['e3u_0' ][:])
 # e3vw_0  = np.squeeze(rootgrp.variables['e3uw_0'][:])
 # coords  = i_coords
 #else:
 # raise Exception('Problem with input coordinates.')

 gdept   , gdepw     = e3_to_depth( e3t_0,  e3w_0, jpk)
 #gdepv   , gdepvw    = e3_to_depth( e3v_0, e3vw_0, jpk)
 #gdept_1d, gdepw_1d  = e3_to_depth(e3t_1d[:,np.newaxis], e3w_1d[:,np.newaxis], jpk)

 rootgrp.close()

 # read in bathymetry

 rootgrp = Dataset(dir+f_bth, "r", format="NETCDF4")
 bathy   = np.squeeze(rootgrp.variables['Bathymetry'][:])
 rootgrp.close()

 # read in variable

 rootgrp = Dataset(dir+f_var, "r", format="NETCDF4")
 var1   = np.squeeze(rootgrp.variables[varstr1][:])
 var2   = np.squeeze(rootgrp.variables[varstr2][:])
 nav_lat = np.squeeze(rootgrp.variables['nav_lat'][:,:])
 nav_lon = np.squeeze(rootgrp.variables['nav_lon'][:,:])
 rootgrp.close()


 
 #lats_endpts = [-14,-6]
 #lons_endpts = [118,118]
 #[lats_sec, lons_sec, dep_sec, var_sec] = extract_depthsection( nav_lat, nav_lon, gdepw, var, lats_endpts, lons_endpts, units='deg' )

 lats_endpts = [140,160]
 lons_endpts = [588,588]
 [ny,nx] = np.shape(nav_lon)
 nav_lon = np.tile(np.arange(0,nx),  (ny,1))
 nav_lat = np.tile(np.arange(0,ny).T,(nx,1)).T





 # Plot sections
 ################
 fig = plt.figure()
 plt.rcParams['figure.figsize'] = (15.0, 15.0)
 ax = fig.add_subplot(3,3,1)
 plt.contourf( nav_lon, nav_lat, bathy, levels=[10,100, 200, 1000, 5000])
 plt.plot( lons_endpts, lats_endpts, 'k')
 #plt.plot( lons2, lats2, 'k')
 #plt.plot( lons3, lats3, 'k')
 #plt.plot( lons4, lats4, 'k')


 #cmap = cm.jet
 cmap = cm.Spectral_r
 cmap.set_bad('white',1.)
 cmap.set_over('red',1.)

 # Section 1
 ##############
 #fig = plt.figure()
 #plt.rcParams['figure.figsize'] = (15.0, 15.0)

 [lats_sec, lons_sec, dep_sec, var_sec] = extract_depthsection( nav_lat, nav_lon, gdepw, var1, lats_endpts, lons_endpts, units='ind' )
 
 plot_section_panel(lats_sec,np.log10(dep_sec),var_sec,np.log10([5000,10]),[ min(lats_endpts), max(lats_endpts)],[np.min(var_sec),np.max(var_sec)],cmap, varstr1,'depth (m)',312)

 [lats_sec, lons_sec, dep_sec, var_sec] = extract_depthsection( nav_lat, nav_lon, gdepw, var2, lats_endpts, lons_endpts, units='ind' )
 
 plot_section_panel(lats_sec,np.log10(dep_sec),var_sec,np.log10([5000,10]),lats_endpts,[30,np.max(var_sec)],cmap, varstr2,'depth (m)',313)

 ## Save output
 fname = 'section_'+f_var.replace('.nc','').replace('.','-')+'.png'
 plt.savefig(fname)



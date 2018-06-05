# -*- coding: utf-8 -*-
"""
AMM60_tools.py

 Various scripts to make AMM60 diagnostics easier
 
 Created on 21 March 2016

 @author: jelt
"""
import getpass # get username
import datetime
import numpy as np  
import socket # get hostname

#==============================================================================
# Function: returns username
#    Input:
#    Output:
#        username
#    Usage:
#        username = getusername()
#==============================================================================        
def getusername():
    username = getpass.getuser()
    return username

#==============================================================================
# Function: returns hostname
#    Input:
#    Output:
#        hostname
#    Usage:
#        hostname = gethost()
#============================================================================== 
def gethost():
    hostname = socket.gethostname()
    return hostname

#==============================================================================
# Function: Take number input. Convert to a formatted date string referenced to a specific data.
#    Input:
#        time_counter - seconds since 1950
#        time_origin - attributes for input: refence time
#		 	time_origin = dataset4.variables['time_counter'].time_origin
#    Output:
#        time_str - fancy string for plotting 
#        time_datetime - date as a datenumber
#        flag_err - [1/0]: [1] if the reference date is not readable
# 			[0] if the reference date is readable, everything is OK
#    Useage:
#        [time_str, time_datetime, flag_err] = NEMO_fancy_datestr( time_counter, time_origin )
#==============================================================================        
def NEMO_fancy_datestr( time_counter, time_origin):

    origin_datetime = datetime.datetime.strptime(time_origin, '%Y-%m-%d %H:%M:%S')
    time_datetime = [ origin_datetime + datetime.timedelta(seconds=time_counter[i]) for i in range(len(time_counter)) ]
    time_str =  [ datetime.datetime.strftime(time_datetime[i], '%Y-%m-%d %H:%M:%S') for i in range(len(time_counter)) ] 
    flag_err = 0
    return [time_str, time_datetime, flag_err]




#==============================================================================        
# doodsonX0.py
#
# The Doodson X0 filter is a simple filter designed to damp out the main tidal frequencies. 
# It takes hourly values, 19 values either side of the central one. A weighted average is taken with the following weights
#  (1010010110201102112 0 2112011020110100101)/30.
#  http://www.ntslf.org/files/acclaimdata/gloup/doodson_X0.html
#
# Learning how to apply a Doodson filter - starting from https://github.com/pwcazenave/PyFVCOM/blob/master/PyFVCOM/tappy.py
#
# In "Data Analaysis and Methods in Oceanography":
#
# "The cosine-Lanczos filter, the transform filter, and the
# Butterworth filter are often preferred to the Godin filter,
# to earlier Doodson filter, because of their superior ability
# to remove tidal period variability from oceanic signals."
#==============================================================================        
def doodsonX0(dates, elevation):
	kern = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1, 1, 0, 2, 1, 1, 2,
    		0,
		2, 1, 1, 2, 0, 1, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]

	half_kern = len(kern)//2
	nslice = slice(half_kern, -half_kern)

	kern = [i/30.0 for i in kern]
	relevation = np.apply_along_axis(lambda m: np.convolve(m, kern, mode=1), axis=0, arr=elevation)

        # Pad out filter signal with NaNs
	relevation = np.lib.pad(relevation[nslice,:,:], ((19,19),(0,0),(0,0)), 'constant', constant_values=(np.nan))

	#relevation = np.convolve(elevation, kern, mode = 1)
	#return dates[nslice], relevation[nslice,:,:]
	return relevation


#==============================================================================        
# getNEMObathydepth.py
#
# Extract a bathymetry value with a supplied lat and lon

from scipy.io import netcdf
import numpy as np

def getNEMObathydepth(lat0,lon0, filename='/Users/jeff/DATA/anyTide/NEMO/bathy_fix_AMM60.rmax.04.nc'):

	f = netcdf.netcdf_file(filename, 'r')
	# load in temperature, salinity 
	bathy = f.variables['Bathymetry'].data # (y, x) 

	#load lat and lon
	lat = f.variables['lat'].data # (y,x)
	lon = f.variables['lon'].data # (y,x)

	f.close()
	diff = abs(lat-lat0) + abs(lon-lon0)
	return bathy[np.where( diff == diff.min())]

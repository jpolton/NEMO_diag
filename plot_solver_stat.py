import pandas as pd
import  matplotlib.pyplot as plt
import re
import numpy as np
import itertools
################################################################################
## Functions
################################################################################

#dir = '/Users/jeff/Desktop/'
dir = ''
config = 'SEAsia'

data = pd.read_table(dir + 'solver.stat', sep=r"\s+", skipinitialspace=True,  header=None)
nrow,ncol = data.shape

"""
 head solver.stat
 it :       1 ssh2:  0.1400401715067860D-03 Umax:  0.2582854442058599D-03 Smin:  0.3404606704631556D+02
 it :       2 ssh2:  0.3541085792571205D-03 Umax:  0.5169316099832227D-03 Smin:  0.3404604024385377D+02
"""

def extract_namelist_variable(varname, filename='output.namelist.dyn'):
    """
    Extract variable value from output.namelist.dyn
    inputs:
        varname  - name of variable in file, case insensitive [str]
        filename - name of file [str]
    outputs:
        val - value for varname [float]
        
    Usage:
        rn_rdt = extract_namelist_variable('rn_rdt','/Users/jeff/Desktop/output.namelist.dyn')
    """

    err_occur = []                         # The list where we will store results.
    substr = "rn_rdt"                        # Substring to use for search.
    try:                              # Try to:
        with open (filename, 'rt') as in_file:        # open file for reading text.
            for linenum, line in enumerate(in_file):    # Keep track of line numbers
                if line.lower().find(substr) != -1: #If case-insensitive substring search matches, then:
                    err_occur.append((linenum, line.rstrip('\n'))) # strip linebreaks, store line and line number in list as tuple.
            for linenum, line in err_occur:              # Iterate over the list of tuples, and
                print("Line ", linenum )  # , ": ", line)  # print results as "Line [linenum]: [line]".
                value = float(re.sub("[^0-9.]", "",line))
                print(substr +" : ", value )
    except ValueError:                   # If log file not found,
        print("Log file not found.")                # print an error message.
        
    return value

################################################################################
################################################################################

data = pd.read_table(dir + 'solver.stat', sep=r"\s+", skipinitialspace=True,  header=None)
nrow,ncol = data.shape

"""
 head solver.stat
 it :       1 ssh2:  0.1400401715067860D-03 Umax:  0.2582854442058599D-03 Smin:  0.3404606704631556D+02
 it :       2 ssh2:  0.3541085792571205D-03 Umax:  0.5169316099832227D-03 Smin:  0.3404604024385377D+02
"""
# Extract the data from the columns


def floatval(datacol,nrow):
	"""
	Function to process the data columns and convert the values to a numpy array.
	The data read seems to read the nan values in a floats but the floats as string, because the
	 'Exponent' symbol is a 'D' not and 'E'. Hence the odd type checking
	"""
	#print np.shape(np.array(datacol))
	col = np.nan*np.zeros((nrow,1))
	for ind in range(nrow):
		#print ('ind {}, data {}, type {}').format(ind, datacol[ind], type(datacol[ind]))
		if type(datacol[ind])==str:

		        if "nan" in datacol[ind] or "NaN" in datacol[ind]:
				print col[ind] 
				break
			else:
				col[ind] = float(datacol[ind].replace('D','E'))
		elif type(datacol[ind])==float:
			if np.isnan(datacol[ind]):
				break
		#print ('ind {}, data {}, out {}').format(ind, datacol[ind], col[ind])
	return col

count = np.array([float(data[2][x]) for x in range(nrow)])
for index in range(3,ncol):
    if 'Umax' in data[index][0]:
        print 'Processing Umax'
        umax = floatval( data[index+1],nrow)
    elif 'Smin' in data[index][0]:
        print 'Processing Smin'
        smin = floatval( data[index+1],nrow)
    elif 'ssh2' in data[index][0]:
        print 'Processing ssh2'
        #ssh2 = [i for i in floatval( data[index+1],nrow)]	
        ssh2 =  floatval( data[index+1],nrow)
################################################################################
################################################################################

rn_rdt = extract_namelist_variable('rn_rdt', dir + 'output.namelist.dyn')
NN_IT000 = extract_namelist_variable('NN_IT000', dir + 'output.namelist.dyn')
NN_ITEND = extract_namelist_variable('NN_ITEND', dir + 'output.namelist.dyn')
CN_EXP = extract_namelist_variable('CN_EXP', dir + 'output.namelist.dyn')

################################################################################
## Plot time series
################################################################################
plt.subplot(3,1,1)
plt.plot(count*rn_rdt/3600./24,  smin, color='r')
plt.ylabel('Smin')

plt.subplot(3,1,2)
plt.plot(count*rn_rdt/3600./24,  umax, color='b')
plt.ylabel('umax')

plt.subplot(3,1,3)
plt.plot(count*rn_rdt/3600./24,  ssh2, color='g')
plt.ylabel('ssh2')
plt.xlabel('simulation time (days)')


#plt.show()

fname = CN_EXP+'_solverStat_'+str(NN_IT000)+'_'+str(NN_ITEND)+'.png'
plt.savefig(fname)

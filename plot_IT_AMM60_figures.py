# plot_IT_AMM60_figures.py

import sys
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.colors as mc   # fancy symetric colours on log scale
import matplotlib.cm as cm   # colormap functionality
import seaborn as sns # cyclic colormap for harmonic phase
sys.path.insert(0, "/login/jelt/python/ipynb/NEMO/")
#from internaltideharmonics_NEMO import internaltideharmonics_NEMO as ITh
import internaltideharmonics_NEMO as ITh
import numpy.ma as ma # masks

##%matplotlib inline

##################################################################################
## Plot the div F as u.gradp terms
def budget_term_logplot(panel, clim, var, iconst, constit_list, label, logth):
	ax = fig.add_subplot(panel)
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, var[iconst,:,:], clim, label, logthresh=logth )
	plt.title(constit_list[iconst]+':'+label+' W/m^2')
	plt.contour( nav_lon_grid_T, nav_lat_grid_T, H, levels=[200], colors='k')
	#plt.xlabel('long'); plt.ylabel('lat')
	return

def celt_budget_term_logplot(panel, maskval, clim, var, iconst, constit_list, label, logth):
	ax = fig.add_subplot(panel)
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, ma.masked_where(APE[iconst,:,:]<maskval,var[iconst,:,:]), clim, label, logthresh=logth )
	plt.title(constit_list[iconst]+':'+label+' W/m^2')
	#plt.xlabel('long'); plt.ylabel('lat')
	plt.contour( nav_lon_grid_T, nav_lat_grid_T, H, levels=[0,200], colors='k')
	plt.xlim(-13, -0)
	plt.ylim(46, 53)
	return
##################################################################################
## Setting and paths
##################################################################################

config = 'AMM60'
#config = 'AMM7'
#cutdown = 'cutdown_'
cutdown = ''
iday = 1

[rootdir, dirname, dstdir] = ITh.get_dirpaths(config)
dstdir = dstdir + 'v2/'

if config == 'AMM60':
    #filename = rootdir + dirname + 'AMM60_1d_20120601_20120630_processed.nc'
    filename = rootdir + dirname + cutdown + 'AMM60_1d_20120801_20120831_processed.nc'
elif config == 'AMM7':
    filename = rootdir + dirname + cutdown + 'GA_1d_20120601_20120829_processed.nc'

print 'dstdir: {}'.format(dstdir)
print 'filename: {}'.format(filename)
din = Dataset(filename)
din2 = Dataset(filename.replace('processed','processed_part2'))


[constit_list, period_list ] = ITh.harmonictable(dirname+'../harmonics_list.txt')

##################################################################################
## Load variables
##################################################################################

C_bt = din.variables['C_bt'][:]
#C_tot = din.variables['C_tot'][:]
D = din.variables['D'][:] #*(-1)
# issue with nan'ed data where it was of unexpected sign. Will set it to zero.

APE = din.variables['APE'][:]
KE_bt = din.variables['KE_bt'][:]
KE_bc = din.variables['KE_bc'][:]

Fu_bt = din.variables['Fu_bt'][:]
Fv_bt = din.variables['Fv_bt'][:]
Fu_bc = din.variables['Fu_bc'][:]
Fv_bc = din.variables['Fv_bc'][:]

divF_bt = din.variables['divF_bt'][:]
divF_bc = din.variables['divF_bc'][:]

minteps = din.variables['inteps'][iday,:,:]
minteps_deep = din.variables['inteps_deep'][iday,:,:]
meps_pyc = din.variables['eps_pyc'][iday,:,:]
#minteps = np.mean( din.variables['inteps'][:], axis=0 )
#minteps_deep = np.mean( din.variables['inteps_deep'][:], axis=0 )
#meps_pyc = np.mean( din.variables['eps_pyc'][:], axis=0 )

#load lat and lon and time
nav_lat_grid_T = din.variables['nav_lat_grid_T'][:] # (y,x)
nav_lon_grid_T = din.variables['nav_lon_grid_T'][:] # (y,x)
#nav_lat_grid_T = ma.masked_where(nav_lat_grid_T == 0, nav_lat_grid_T)
#nav_lon_grid_T = ma.masked_where(nav_lat_grid_T == 0, nav_lon_grid_T)

H = din.variables['H'][:] # (y,x)


constit_period =  din.variables['T_p'][:]

constit_chararr =  din.variables['constit'][:]
nh = np.shape(constit_chararr)[0]
constit_list = [ ''.join(constit_chararr[i,:]) for i in range(nh) ]


#C_bt = np.sum(C_bt[:,:,:], axis=0)
#C_tot = np.sum(C_tot[:,:,:], axis=0)
#divF_bt = np.sum(divF_bt[:,:,:], axis=0)
#divF_bc = np.sum(divF_bc[:,:,:], axis=0)

advKE = din2.variables['advKE'][:]
prodKE = din2.variables['prodKE'][:]
ugradp_bt = din2.variables['ugradpbt'][:]
ugradp_bc = din2.variables['ugradpbc'][:]

##################################################################################
## Fix mask issues
##################################################################################

# mask land
I_land = [divF_bt==0]

C_bt[I_land] = np.nan
#C_tot[I_land] = np.nan

# issue with nan'ed data where it was of unexpected sign. Treat that here. Set it to zero.
D[np.isnan(D)] = 0
D[I_land] = np.nan
D = np.ma.masked_where(divF_bt==0, D)

Fu_bt[I_land] = np.nan
Fv_bt[I_land] = np.nan
Fu_bc[I_land] = np.nan
Fv_bc[I_land] = np.nan

divF_bt[I_land] = np.nan
divF_bc[I_land] = np.nan

advKE[I_land] = np.nan
prodKE[I_land] = np.nan
ugradp_bt[I_land] = np.nan
ugradp_bc[I_land] = np.nan


ugradp_bc = np.ma.masked_where(np.isnan(ugradp_bc), ugradp_bc)
ugradp_bt = np.ma.masked_where(np.isnan(ugradp_bt), ugradp_bt)


#Fu_bt[np.isnan(Fu_bt)] = 0
#Fv_bt[np.isnan(Fv_bt)] = 0

##################################################################################
## Start plots
##################################################################################
## make a dummy figure to sort spacing out
fig = plt.figure()
plt.rcParams['figure.figsize'] = (15.0, 15.0)

## Plot div F as u.grad p terms, and differences
################################################

def plot_divterms(region='Whole'):
	## M2
	constit = 'M2'
	iconst = constit_list.index(constit)

	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 15.0)
	budget_term_logplot(111, [-10**1,10**1], ugradp_bt, iconst,constit_list, 'u.gradp_bt', 3)
	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_ugradpbt_' + constit + '_' + config + '.png'
	plt.savefig(fname)


	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 15.0)
	budget_term_logplot(111, [-10**1,10**1], ugradp_bc, iconst,constit_list, 'u.gradp_bc', 3)
	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_ugradpbc_' + constit + '_' + config + '.png'
	plt.savefig(fname)


	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 15.0)
	budget_term_logplot(111, [-10**1,10**1], divF_bt, iconst,constit_list, 'divF_bt', 3)
	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_divFbt_' + constit + '_' + config + '.png'
	plt.savefig(fname)


	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 15.0)
	budget_term_logplot(111, [-10**1,10**1], divF_bc, iconst,constit_list, 'divF_bc', 3)
	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_divFbc_' + constit + '_' + config + '.png'
	plt.savefig(fname)


	## Plot the difference between u.grad p_bt and divF_bt
	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0,7.0)
	budget_term_logplot(121, [-10**1,10**1], divF_bt-ugradp_bt, iconst,constit_list, 'divF-u.gradp bt', 3)
	budget_term_logplot(122, [-10**1,10**1], divF_bc-ugradp_bc, iconst,constit_list, 'divF-u.gradp bc', 3)
	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_diff_divF_' + constit + '_' \
	+ '_' + region + config + '.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)
	return


## Plot Barotropic fluxes for different species.
################################################

## Plot Barotropic fluxes for different species. Whole domain
#############################################################

def plot_barotropic_fluxes_for_harmonic_bands(region='Whole'):

	print '{} loaded constituents: {}'.format(nh, constit_list)
	# Bands APE
	# Define bands by constituent indices
	index_total = [constit_list.index(lab) for lab in constit_list]
	index_diurnal = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='1']
	index_semidiu = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='2']
	index_quatdiu = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='4']
	index_dic = {'label':['total', 'diurnal', 'semi-diurnal','quarter-diurnal'],
			             'field':[index_total, index_diurnal, index_semidiu, index_quatdiu],
				                   'Fbtclim':[[0,1000], [0,100],  [0,1000],     [0,100]] }
	#              'Fbtclim':[[0,4000], [0,1000],  [0,4000],     [0,100]] }

	for count in range(len(index_dic['label'])):
		indices = index_dic['field'][count]
		print 'band: ',index_dic['label'][count], [constit_list[ind] for ind in indices]

		Fu = np.nansum(Fu_bt[indices,:,:],axis=0)
		Fv = np.nansum(Fv_bt[indices,:,:],axis=0)

		plt.rcParams['figure.figsize'] = (7.0, 7.0)

		fig = plt.figure()
		plt.rcParams['figure.figsize'] = (15.0, 15.0)
		ITh.plot_F(fig,Fu,Fv,nav_lon_grid_T,nav_lat_grid_T, mode='bt', type='contourf', clim=index_dic['Fbtclim'][count], side='lhs',zoom=False)
		plt.title(index_dic['label'][count]+': Barotropic flux')
		## Save output
		fname = dstdir + 'internaltideharmonics_NEMO_Fbt_' + \
				index_dic['label'][count] + '_' + region + '_' + config + '.png'
		plt.savefig(fname)
		print fname
		plt.close(fig)

		return

## Plot APE budgets for all species, bands and total
#############################################################

# Note that APE is computed for a snapshot N2, index iday


## Plot APE budgets for all species and total
def plot_APE_by_harmonic_bands():
	print 'Note that APE is computed for a snapshot N2, index', iday
	print 'Generate APE maps. Available constituents: {}'.format(constit_list)
	APE[APE==0] = np.nan

	# Bands APE
	# Define bands by constituent indices
	index_total = [constit_list.index(lab) for lab in constit_list]
	index_diurnal = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='1']
	index_semidiu = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='2']
	index_quatdiu = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='4']
	index_dic = {'label':['total', 'diurnal', 'semi-diurnal','quarter-diurnal'], 'field':[index_total, index_diurnal, index_semidiu, index_quatdiu]}

	for count in range(len(index_dic['label'])):
		indices = index_dic['field'][count]
		print 'band: ',index_dic['label'][count], [constit_list[ind] for ind in indices]

		var = np.sum(APE[indices,:,:],axis=0)/H
		logrange = list(np.array([0,1,2,3,4]) - np.array([2,2,2,2,2]))


		fig = plt.figure()
		plt.rcParams['figure.figsize'] = (15.0, 15.0)
		cmap = cm.Spectral_r
		cmap.set_over('#840000',1.)
		cmap.set_under('white',1.)

		plt.contourf( nav_lon_grid_T, nav_lat_grid_T, np.log10(var), logrange, cmap=cmap, extend="both" )
		plt.clim([min(logrange),max(logrange)])
		plt.colorbar()
		plt.title(config+' '+index_dic['label'][count]+': log10(APE [J/m^3])')
		plt.contour( nav_lon_grid_T, nav_lat_grid_T, H, [0,200], colors='k' )

		## Save output
		fname = dstdir +'internaltideharmonics_NEMO_APE_' + index_dic['label'][count] + '_' + config + '.png'
		plt.savefig(fname)
		print fname
		plt.close(fig)



if(0):

	## Plot C_bt
	######################

	constit = 'M2'
	iconst = constit_list.index(constit)


	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 15.0)


	#ax = fig.add_subplot(221)
	#clim = [-10**-4,10**-4]
	#[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, C_tot[iconst,:,:], clim, 'C_tot', logthresh=10 )
	#plt.title(constit+': Barotropic to baroclinic tidal energy conversion rate, C_tot, W/m^2')
	#plt.xlabel('long'); plt.ylabel('lat')

	ax = fig.add_subplot(222)
	clim = [-10**1,10**1]
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, C_bt[iconst,:,:], clim, 'C_bt', logthresh=3 )
	plt.title(constit+': C_bt, W/m^2')
	plt.xlabel('long'); plt.ylabel('lat')

	#ax = fig.add_subplot(223)
	#clim = [-10**-3,10**-3]
	#[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, C_tot[iconst,:,:], clim, 'C_tot', logthresh=7 )
	#plt.title(constit+': Barotropic to baroclinic tidal energy conversion rate, C_tot, W/m^2')
	#plt.xlabel('long'); plt.ylabel('lat')
	#plt.xlim(-13, -3)
	#plt.ylim(46, 53)

	ax = fig.add_subplot(224)
	clim = [-10**1,10**1]
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, C_bt[iconst,:,:], clim, 'C_bt', logthresh=3 )
	plt.title(constit+': C_bt, W/m^2')
	plt.xlabel('long'); plt.ylabel('lat')
	plt.xlim(-13, -3)
	plt.ylim(46, 53)

	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_C_bt' + constit + '_' + config + '.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)

	## Plot Barotropic fluxes for different species. Whole domain
	#############################################################

	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (7.0, 7.0)

	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 15.0)

	clim = [-10**1,10**1]
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, minteps_deep, clim, 'minteps_deep', logthresh=3 )
	plt.title('Beneath pycnocline time mean depth int epsilon25h [W/m^2]')
	plt.xlabel('long'); plt.ylabel('lat')

	## Save output
	fname = dstdir + 'internaltideharmonics_NEMO_' + 'deepinteps_' + config + '.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)


	## Plot Harmonic components Budget. Whole domain
	###################
	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 7.0)

	## M2
	iconst = constit_list.index('M2')
	budget_term_logplot(421, [-10**1,10**1], divF_bt, iconst,constit_list, 'divF_bt', 3)
	budget_term_logplot(423, [-10**1,10**1], D, iconst,constit_list, 'D', 3)
	budget_term_logplot(425, [-10**1,10**1], C_bt, iconst,constit_list, 'C_bt', 3)
	budget_term_logplot(427, [-10**1,10**1], divF_bc, iconst,constit_list, 'divF_bc', 3)

	## S2
	iconst = constit_list.index('S2')
	budget_term_logplot(422, [-10**1,10**1], divF_bt, iconst,constit_list, 'divF_bt', 3)
	budget_term_logplot(424, [-10**1,10**1], D, iconst,constit_list, 'D', 3)
	budget_term_logplot(426, [-10**1,10**1], C_bt, iconst,constit_list, 'C_bt', 3)
	budget_term_logplot(428, [-10**1,10**1], divF_bc, iconst,constit_list, 'divF_bc', 3)

	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_budget_terms_' + config + '.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)


	## K2: Plot budget by components K2
	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 15.0)
	constit = 'K2'
	iconst = constit_list.index(constit)
	budget_term_logplot(221, [-10**1,10**1], divF_bt, iconst,constit_list, 'divF_bt', 3)
	budget_term_logplot(222, [-10**1,10**1], D, iconst,constit_list, 'D', 3)
	budget_term_logplot(223, [-10**1,10**1], C_bt, iconst,constit_list, 'C_bt', 3)
	budget_term_logplot(224, [-10**1,10**1], divF_bc, iconst,constit_list, 'divF_bc', 3)
	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_budget_terms_' + constit + '.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)


	## Plot Harmonic components Budget. Celtic Sea
	###################


	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (7.0, 15.0)

	maskval = 0 #1 # APE value [J/m^2] to mask fields
	## M2
	iconst = constit_list.index('M2')
	celt_budget_term_logplot(421,maskval, [-10**1,10**1], divF_bt, iconst,constit_list, 'divF_bt', 3)
	celt_budget_term_logplot(423,maskval, [-10**1,10**1], D, iconst,constit_list, 'D', 3)
	celt_budget_term_logplot(425,maskval, [-10**1,10**1], C_bt, iconst,constit_list, 'C_bt', 3)
	lons4 = [-10, 0]
	lats4 = [47.5, 51]
	plt.plot(lons4, lats4 ,'k')
	celt_budget_term_logplot(427,maskval, [-10**1,10**1], divF_bc, iconst,constit_list, 'divF_bc', 3)

	## S2
	iconst = constit_list.index('S2')
	celt_budget_term_logplot(422,maskval, [-10**1,10**1], divF_bt, iconst,constit_list, 'divF_bt', 3)
	celt_budget_term_logplot(424,maskval, [-10**1,10**1], D, iconst,constit_list, 'D', 3)
	celt_budget_term_logplot(426,maskval, [-10**1,10**1], C_bt, iconst,constit_list, 'C_bt', 3)
	celt_budget_term_logplot(428,maskval, [-10**1,10**1], divF_bc, iconst,constit_list, 'divF_bc', 3)
	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_budget_terms_celtic' + config + '.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)


	## Plot Harmonic totals Budget. Whole domain
	##################################

	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (15.0, 7.0 )

	## M2
	iconst = constit_list.index('M2')
	budget_term_logplot(121, [-10**1,10**1], divF_bc+divF_bt+D+C_bt, iconst,constit_list, 'sum: divF_bc+divF_bt+D+C_bt', 3)

	## S2
	iconst = constit_list.index('S2')
	budget_term_logplot(122, [-10**1,10**1], divF_bc+divF_bt+D+C_bt, iconst,constit_list, 'sum: divF_bc+divF_bt+D+C_bt', 3)

	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_budget_sum_' + config + '.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)


	## Plot Total Budget: Sum
	###################
	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (20.0, 20.0)


	ax = fig.add_subplot(221)
	clim = [-10**1,10**1]
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, np.sum(D+C_bt+ugradp_bt+ugradp_bc,axis=0), clim, 'D+C_tot+ugradp_bt+ugradp_bc', logthresh=3 )
	#[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, np.sum(D+C_bt+divF_bt+divF_bc,axis=0), clim, 'D+C_tot+divF_bt+divF_bc', logthresh=3 )
	plt.title('all: D+C_bt+divF_bt+divF_bc, W/m^2')
	plt.xlabel('long'); plt.ylabel('lat')

	ax = fig.add_subplot(222)
	clim = [-10**1,10**1]
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, minteps_deep, clim, 'minteps_deep', logthresh=3 )
	plt.title('Beneath pycnocline time mean depth int epsilon25h [W/m^2]')
	plt.xlabel('long'); plt.ylabel('lat')

	ax = fig.add_subplot(223)
	clim = [-10**1,10**1]
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, np.sum(D+C_bt+ugradp_bt+ugradp_bc,axis=0), clim, 'D+C_tot+ugradp_bt+ugradp_bc', logthresh=3 )
	#[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, np.sum(D+C_bt+divF_bt+divF_bc,axis=0), clim, 'D+C_tot+divF_bt+divF_bc', logthresh=3 )
	plt.title('all: D+C_bt+divF_bt+divF_bc, W/m^2')
	plt.xlabel('long'); plt.ylabel('lat')
	plt.xlim(-13, -3)
	plt.ylim(46, 53)

	ax = fig.add_subplot(224)
	clim = [-10**1,10**1]
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, minteps_deep, clim, 'minteps_deep', logthresh=3 )
	plt.title('Beneath pycnocline time mean depth int epsilon25h [W/m^2]')
	plt.xlabel('long'); plt.ylabel('lat')
	plt.xlim(-13, -3)
	plt.ylim(46, 53)


	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_budget' + config + '.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)
	###############


	## Plot Total Budget: Sum
	###################
	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (20.0, 20.0)
	for iconst in range(9):
		budget_term_logplot(331+iconst, [-10**1,10**1], divF_bc+divF_bt+D+C_bt, iconst,constit_list, 'divF_bc+divF_bt+D+C_bt', 3)
	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_sums_by_component.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)


	## Plot harmonic totals for each components. Whole domain
	#########################################################

	eps_deep = np.tile(minteps_deep[np.newaxis,:],(nh,1,1))/nh # distribute non-harmonic variables across harmonic bins.
	varsum_a = eps_deep + ugradp_bt +  D + ugradp_bc + C_bt + advKE + prodKE
	varsum_b = eps_deep + divF_bt   +  D + divF_bc + C_bt +  advKE + prodKE

	var_arr_a = [eps_deep,   ugradp_bt,  D,        C_bt,      ugradp_bc,   advKE   , prodKE,    varsum_a]
	var_lst_a = ['eps bot',                   'ugradpt',  'D      ','C_bt   ', 'ugradpc' ,  'advKE  ','prodKE ', 'varsuma' ]

	var_arr_b = [eps_deep, divF_bt,   D,         divF_bc,   C_bt,       advKE   , prodKE,    varsum_b]
	var_lst_b = ['eps bot',                 'divF_bt', 'D      ', 'divF_bc', 'C_bt   ',  'advKE  ','prodKE ', 'varsumb' ]

	var_arr = var_arr_a
	var_lst = var_lst_a


if(0): # DELETE ME
	## Plot Whole domain
	####################

	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (20.0, 20.0)

	for ind in range(len(var_arr)):
		print var_lst[ind], np.nanmean(var_arr[ind].flatten())
		ax = fig.add_subplot(331+ind)
		clim = [-10**1,10**1]
		[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, np.sum(var_arr[ind],axis=0), clim, var_lst[ind], logthresh=3 )
		plt.xlabel('long'); plt.ylabel('lat')
		plt.contour( nav_lon_grid_T, nav_lat_grid_T, H, [0,200], colors='k' )

	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_harmtotals_all_components.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)


	## Plot Celtic Zoom
	###################

	fig = plt.figure()
	plt.rcParams['figure.figsize'] = (20.0, 20.0)

	for ind in range(len(var_arr)):
		print var_lst[ind], np.nanmean(var_arr[ind].flatten())
		ax = fig.add_subplot(331+ind)
		clim = [-10**1,10**1]
		[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, np.sum(var_arr[ind],axis=0), clim, var_lst[ind], logthresh=3 )
		plt.xlabel('long'); plt.ylabel('lat')
	    	plt.xlim(-13, -0)
	    	plt.ylim(46, 53)
		plt.contour( nav_lon_grid_T, nav_lat_grid_T, H, [0,200], colors='k' )

	## Save output
	fname = dstdir +'internaltideharmonics_NEMO_harmtotals_all_components_Celtic.png'
	plt.savefig(fname)
	print fname
	plt.close(fig)




## Plot components and total by harmonic band: Whole domain
###########################################################

def plot_components_and_total_by_harmonic_band(region='Whole'):
#def plot_components_and_total_by_harmonic_band(constit_list,var_arr,var_lst,nav_lon_grid_T,nav_lat_grid_T,H,config='AMM60'):

	# Define bands by constituent indices
	index_total = [constit_list.index(lab) for lab in constit_list]
	index_diurnal = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='1']
	index_semidiu = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='2']
	index_quatdiu = [constit_list.index(lab) for lab in constit_list if constit_list[constit_list.index(lab)][1]=='4']
	index_dic = {'label':['total', 'diurnal', 'semi-diurnal','quarter-diurnal'], 'field':[index_total, index_diurnal, index_semidiu, index_quatdiu]}

	for count in range(len(index_dic['label'])):
		indices = index_dic['field'][count]
		print 'band: ',index_dic['label'][count], [constit_list[ind] for ind in indices]

		fig = plt.figure()
		plt.rcParams['figure.figsize'] = (15.0, 15.0)

		cmap = cm.Spectral_r
		cmap.set_over('#840000',1.)
		cmap.set_under('white',1.)

		for ind_term in range(len(var_arr)):
			print var_lst[ind_term], np.nanmean(var_arr[ind_term].flatten())
			ax = fig.add_subplot(331+ind_term)
			clim = [-10**1,10**1]
			[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, np.sum(var_arr[ind_term][indices,:,:],axis=0), \
			 				clim, var_lst[ind_term]+index_dic['label'][count], logthresh=3 )
			plt.xlabel('long'); plt.ylabel('lat')
			plt.contour( nav_lon_grid_T, nav_lat_grid_T, H, [0,200], colors='k' )
			if region=='Celtic':
				    plt.xlim(-13, -0)
				    plt.ylim(46, 53)

		## Save output
		fname = dstdir +'internaltideharmonics_NEMO_harmtotals_' + index_dic['label'][count] \
		       + '_' + region + '_' + config + '.png'
		plt.savefig(fname)
		print fname
		plt.close(fig)



## Plot totals by harmonic band: Celtic domain
##############################################


################################################
################################################
################################################

## Plot APE budgets for all species and total
plot_APE_by_harmonic_bands()

## Plot div F as u.grad p terms, and differences
plot_divterms(region='Whole')

## Plot components and total by harmonic band: Whole domain
#plot_components_and_total_by_harmonic_band(constit_list,var_arr,var_lst,nav_lon_grid_T,nav_lat_grid_T,H,config='AMM60')
plot_components_and_total_by_harmonic_band(region='Whole')
plot_components_and_total_by_harmonic_band(region='Celtic')









if(0):
	# Pickle variables
	import pickle

	with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
	    pickle.dump([var_arr_a, nav_lon_grid_T, nav_lat_grid_T], f)
	    #pickle.dump(var_arr_a, f)

	# Recover variables: (Note minteps wasn't size nh,ny,nx befrore)
	f = open('objs.pkl', 'rb')
	#[minteps_deep,   ugradp_bt,  D,        C_bt,      ugradp_bc,   advKE   , prodKE,    varsum_a] = pickle.load(f)
	[[minteps_deep,   ugradp_bt,  D,        C_bt,      ugradp_bc,   advKE   , prodKE,    varsum_a], nav_lon_grid_T, nav_lat_grid_T] = pickle.load(f)
	f.close()

# plot_IT_AMM60_figures.py


from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt  # plotting
import matplotlib.colors as mc   # fancy symetric colours on log scale
import matplotlib.cm as cm   # colormap functionality
import seaborn as sns # cyclic colormap for harmonic phase
import internaltideharmonics_NEMO as ITh
import numpy.ma as ma # masks

%matplotlib inline

##################################################################################
## Plot the div F as u.gradp terms
def budget_term_logplot(panel, clim, var, iconst, constit_list, label, logth):
	ax = fig.add_subplot(panel)
	[img,cb] = ITh.contourf_symlog( nav_lon_grid_T, nav_lat_grid_T, var[iconst,:,:], clim, label, logthresh=logth )
	plt.title(constit_list[iconst]+':'+label+' W/m^2')
	plt.contour( nav_lon_grid_T, nav_lat_grid_T, H, levels=[200], colors='k')
	#plt.xlabel('long'); plt.ylabel('lat')
	return

##################################################################################


config = 'AMM60'
#config = 'AMM7'
#cutdown = 'cutdown_'
cutdown = ''
iday = 1

[rootdir, dirname, dstdir] = ITh.get_dirpaths(config)

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


## Load variables

C_bt = din.variables['C_bt'][:]
#C_tot = din.variables['C_tot'][:]
D = din.variables['D'][:] #*(-1)
if(1): #if np.nanmin(D) < 0:
	print 'NB Applied sign fix to D. D should be positive.'
	print 'Mean D = {}'.format(np.nanmean(D.flatten()))

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

# mask land
I_land = [divF_bt==0]

C_bt[I_land] = np.nan
#C_tot[I_land] = np.nan
D[I_land] = np.nan

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


## Plot div F as u.grad p terms
###############################

#load lat and lon and time
nav_lat_grid_T = din2.variables['nav_lat_grid_T'][:] # (y,x)
nav_lon_grid_T = din2.variables['nav_lon_grid_T'][:] # (y,x)

H = din.variables['H'][:] # (y,x)


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


## Plot Barotropic fluxes for different species.
################################################

## Plot Barotropic fluxes for different species. Whole domain
#############################################################

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
	fname = dstdir + 'internaltideharmonics_NEMO_Fbt_' + index_dic['label'][count] + '_' + config + '.png'
	plt.savefig(fname) 


## Plot APE budgets for all species, bands and total
#############################################################

# Note that APE is computed for a snapshot N2, index iday


## Plot APE budgets for all species and total

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




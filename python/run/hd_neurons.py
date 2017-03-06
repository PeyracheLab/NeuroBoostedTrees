import warnings
import pandas as pd
import scipy.io
import numpy as np
from fonctions import *
import sys, os


colors=['#F5A21E', '#02A68E', '#EF3E34', '#134B64', '#FF07CD','b']


#####################################################################
# DATA LOADING
#####################################################################
adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree.mat'))

# m1_imported = scipy.io.loadmat('/home/guillaume/spykesML/data/m1_stevenson_2011.mat')

#####################################################################
# DATA ENGINEERING
#####################################################################
data 			= 	pd.DataFrame()
data['time'] 	= 	np.arange(len(adrien_data['Ang']))
data['ang'] 	= 	adrien_data['Ang'].flatten()
data['x'] 		= 	adrien_data['X'].flatten()
data['y'] 		= 	adrien_data['Y'].flatten()
data['vel'] 	= 	adrien_data['speed'].flatten()

print data.head()

#######################################################################
# FIRST STAGE LEARNING
#######################################################################
print('Features are:\n %s' %list(data.drop('time', axis=1).keys()))

methods = ['glm_pyglmnet','xgb_run']

Models = {}
for method in methods:
    Models[method] = dict()
    Models[method]['PR2']  = list()
    Models[method]['Yt_hat']  = list()

Models['ens'] = dict()
Models['ens']['Yt_hat'] = list()
Models['ens']['PR2'] = list()

X = data.drop('time', axis=1).values

Y = np.hstack((adrien_data['Pos'], adrien_data['ADn'])).transpose()

nneurons = len(Y)

for i in xrange(nneurons):

	print '\n running for neuron %d' % i

	y = Y[i]
		
	for method in methods:
	    print('Running '+method+'...')
	    Yt_hat, PR2 = fit_cv(X, y, algorithm = method, n_cv=8, verbose=1)	    

	    Models[method]['Yt_hat'].append(Yt_hat)
	    Models[method]['PR2'].append(PR2)
	
	######################################################################
	# SECOND STAGE LEARNING : ENSEMBLE
	######################################################################
	X_ens = list()
	for method in methods:
	    X_ens.append(Models[method]['Yt_hat'][i])
	    
	# The columns of X_ens are the predictions of each of the above methods
	X_ens = np.transpose(np.array(X_ens))
	
	#We can use XGBoost as the 2nd-stage model
	Yt_hat, PR2 = fit_cv(X_ens, y, algorithm = 'xgb_run', n_cv=8, verbose=1)
	Models['ens']['Yt_hat'].append(Yt_hat)
	Models['ens']['PR2'].append(PR2)
	

	
	
########################################################################
# PLOTTINGs
########################################################################
plot_model_comparison(	['glm_pyglmnet', 'xgb_run','ens'],
						models=Models,
                      	color=colors,
                      	labels = ['GLM','XGB','Ens.\n (XGBoost)'])


plt.show()

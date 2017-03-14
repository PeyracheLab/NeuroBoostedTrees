#!/usr/bin/env python

'''
    File name: main_hd_data.py
    Author: Guillaume Viejo
    Date created: 06/03/2017    
    Python Version: 2.7

To fit the tuning curve (model-based) to the hd data of the thalamus and to compare to XGBoost
Fit only with the angle 

'''

import warnings
import pandas as pd
import scipy.io
import numpy as np
# Should not import fonctions if already using tensorflow for something else
from fonctions import *
import sys, os
import itertools
import cPickle as pickle


#####################################################################
# DATA LOADING
#####################################################################
adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree.mat'))
# m1_imported = scipy.io.loadmat('/home/guillaume/spykesML/data/m1_stevenson_2011.mat')

final_data = {}


#####################################################################
# DATA ENGINEERING
#####################################################################
data 			= 	pd.DataFrame()
data['time'] 	= 	np.arange(len(adrien_data['Ang']))		# TODO : import real time from matlab script
data['ang'] 	= 	adrien_data['Ang'].flatten() 			# angular direction of the animal head
data['x'] 		= 	adrien_data['X'].flatten() 				# x position of the animal 
data['y'] 		= 	adrien_data['Y'].flatten() 				# y position of the animal
data['vel'] 	= 	adrien_data['speed'].flatten() 			# velocity of the animal 
# Engineering features
data['cos']		= 	np.cos(adrien_data['Ang'].flatten())	# cosinus of angular direction
data['sin']		= 	np.sin(adrien_data['Ang'].flatten())	# sinus of angular direction
# Firing data
for i in xrange(adrien_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = adrien_data['Pos'][:,i]
for i in xrange(adrien_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = adrien_data['ADn'][:,i]


methods = ['mb_10', 'mb_100', 'mb_100', 'xgb_run', 'nn']
colors=['#F5A21E', '#02A68E', '#EF3E34', '#134B64', '#FF07CD','b']
labels = ['GLM', 'NN', 'XGB','Ens.\n (XGBoost)', 'NN']

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def test_features(features, targets, learners = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens']):
	'''
		Main function of the script
		Return : dictionnary with for each models the score PR2 and Yt_hat
	'''
	X = data[features].values
	Y = data[targets].values	
	Models = {method:{'PR2':[],'Yt_hat':[]} for method in learners}
	learners_ = list(learners)
	print learners_

	# Special case for glm_pyglmnet to go parallel
	if 'glm_pyglmnet' in learners:
		print('Running glm_pyglmnet...')				
		# # Map the targets for 6 cores by splitting Y in 6 parts
		pool = multiprocessing.Pool(processes = 6)
		value = pool.map(glm_parallel, itertools.izip(range(6), itertools.repeat(X), itertools.repeat(Y)))
		Models['glm_pyglmnet']['Yt_hat'] = np.vstack([value[i]['Yt_hat'] for i in xrange(6)]).transpose()
		Models['glm_pyglmnet']['PR2'] = np.vstack([value[i]['PR2'] for i in xrange(6)])		
		learners_.remove('glm_pyglmnet')		

	for i in xrange(Y.shape[1]):
		y = Y[:,i]
		# TODO : make sure that 'ens' is the last learner
		for method in learners_:
			if method != 'ens': # FIRST STAGE LEARNING			
				print('Running '+method+'...')								
				Yt_hat, PR2 = fit_cv(X, y, algorithm = method, n_cv=8, verbose=1)	    
				Models[method]['Yt_hat'].append(Yt_hat)
				Models[method]['PR2'].append(PR2)			

			elif method == 'ens': # SECOND STAGE LEARNING
				X_ens = np.transpose(np.array([Models[m]['Yt_hat'][i] for m in learners if m != method]))
				#We can use XGBoost as the 2nd-stage model
				Yt_hat, PR2 = fit_cv(X_ens, y, algorithm = 'xgb_run', n_cv=8, verbose=1)		
				Models['ens']['Yt_hat'].append(Yt_hat)	    
				Models['ens']['PR2'].append(PR2)				    	

	for m in Models.iterkeys():
		Models[m]['Yt_hat'] = np.array(Models[m]['Yt_hat'])
		Models[m]['PR2'] = np.array(Models[m]['PR2'])
		
	return Models

########################################################################
# COMBINATIONS DEFINITIONS
########################################################################
combination = {
	10: 	{
		 	'features' 	:	['ang'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']], # all neurons
			'figure'	:	'../../figures/10_PR2_feat_ang_targ_pos_adn.pdf'
		},			
}

########################################################################
# MAIN LOOP
########################################################################
for k in np.sort(combination.keys()):
	features = combination[k]['features']
	targets = combination[k]['targets']	

	results = test_features(features, targets, methods)
	
	final_data[k] = results
	colors = ['#F5A21E']*3 + ['#134B64', '#02A68E']
	plot_model_comparison(results, ['MB \n 10 bins', 'MB \n 100 bins', 'MB \n 1000 bins', 'XGB', 'NN'], colors, methods)
	savefig(combination[k]['figure'], dpi=900, bbox_inches = 'tight')
	




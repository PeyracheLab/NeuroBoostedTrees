#!/usr/bin/env python

'''
    File name: main_hd_data.py
    Author: Guillaume Viejo
    Date created: 04/03/2017    
    Python Version: 2.7

This is the main file for characterizing thalamic head-direction data using glm and xgboost

Here are listed all the combination that are tested

1. 	Features : cos and sinus of the angle of the head-direction
	Model : GLM, NN, XGBOOST, ENSEMBLE
2. 	Features : ang, x, y, vel
	Model : GLM, NN, XGBOOST, ENSEMBLE
3. 	Features : cos, sinus, ang, x, y, vel
	Model : GLM, NN, XGBOOST, ENSEMBLE
4. 	Features : Pos (Test is on ADn)
	Model : GLM, NN, XGBOOST, ENSEMBLE
5.	Features : ADn (Test is on Pos)
	Model : GLM, NN, XGBOOST, ENSEMBLE
6 	Features : cos, sin, Pos
	Model : GLM, NN, XGBOOST, ENSEMBLE
7 	Features : cos, sin, ADn
	Model : GLM, NN, XGBOOST, ENSEMBLE
8	Features : everything, Pos
	Model : GLM, NN, XGBOOST, ENSEMBLE
9 	Features : everything, ADn
	Model : GLM, NN, XGBOOST, ENSEMBLE

Number order are the keys for the dictionnary
'''

import warnings
import pandas as pd
import scipy.io
import numpy as np
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

print('Features are:\n %s' %list(data.drop('time', axis=1).keys()))

methods = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens']
colors=['#F5A21E', '#02A68E', '#EF3E34', '#134B64', '#FF07CD','b']
labels = ['GLM', 'NN', 'XGB','Ens.\n (XGBoost)']

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def fit_cv_star(n, X, Y):
	# Break Y data for 4 cores
	index = np.arange(Y.shape[1])
	index = np.array_split(index, 4)[n]	
	Y_smaller = Y[:,index]
	tmp = {'Yt_hat':[], 'PR2':[]}
	for i in xrange(Y_smaller.shape[1]):
		y = Y_smaller[:,i]
		Yt_hat, PR2 = fit_cv(X, y, algorithm = 'glm_pyglmnet', n_cv=8, verbose=1)	    
		tmp['Yt_hat'].append(Yt_hat)
		tmp['PR2'].append(PR2)
	tmp['Yt_hat'] = np.array(tmp['Yt_hat'])
	tmp['PR2'] = np.array(tmp['PR2'])
	return tmp

def glm_parallel(a_b):
	return fit_cv_star(*a_b)

def test_features(features, targets, learners = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens']):
	'''
		Main function of the script
		Return : dictionnary with for each models the score PR2 and Yt_hat
	'''
	X = data[features].values
	Y = data[targets].values	
	Models = {method:{'PR2':[],'Yt_hat':[]} for method in learners}
	learners_ = list(learners)

	# Special case for glm_pyglmnet to go parallel
	if 'glm_pyglmnet' in learners:
		print('Running glm_pyglmnet...')				
		# # Map the targets for 4 cores by splitting Y in 4 parts
		pool = multiprocessing.Pool(processes = 4)
		value = pool.map(glm_parallel, itertools.izip(range(4), itertools.repeat(X), itertools.repeat(Y)))
		Models['glm_pyglmnet']['Yt_hat'] = np.vstack([value[i]['Yt_hat'] for i in xrange(4)])
		Models['glm_pyglmnet']['PR2'] = np.vstack([value[i]['PR2'] for i in xrange(4)])		
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
	1: 	{
		 	'features' 	:	['cos', 'sin'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']], # all neurons
			'figure'	:	'../../figures/1_PR2_feat_cos_sin_targ_pos_adn.pdf'
		},
	2:	{
			'features' 	:	['ang', 'x', 'y', 'vel', 'cos', 'sin'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']], # all neurons
			'figure'	:	'../../figures/2_PR2_feat_all_targ_pos_adn.pdf'
		},
	3:	{
			'features' 	:	['ang', 'x', 'y', 'vel'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']], # all neurons
			'figure'	:	'../../figures/3_PR2_feat_raw_targ_pos_adn.pdf'
		},
	4:	{
			'features' 	:	[i for i in list(data) if i.split(".")[0] in ['Pos']],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['ADn']],
			'figure'	:	'../../figures/4_PR2_feat_pos_targ_adn.pdf'
		},
	5:	{
			'features' 	:	[i for i in list(data) if i.split(".")[0] in ['ADn']],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['Pos']], 
			'figure'	:	'../../figures/5_PR2_feat_adn_targ_pos.pdf'
		},
	6:	{
			'features' 	:	[i for i in list(data) if i.split(".")[0] in ['cos', 'sin', 'Pos']],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['ADn']],
			'figure'	:	'../../figures/6_PR2_feat_cos_sin_pos_targ_adn.pdf'
		},								
	7:	{
			'features' 	:	[i for i in list(data) if i.split(".")[0] in ['cos', 'sin', 'ADn']],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['Pos']],
			'figure'	:	'../../figures/7_PR2_feat_cos_sin_adn_targ_pos.pdf'
		},		
	8:	{
			'features' 	:	[i for i in list(data) if i.split(".")[0] in ['ang', 'x', 'y', 'vel', 'cos', 'sin', 'Pos']],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['ADn']],
			'figure'	:	'../../figures/8_PR2_feat_all_pos_targ_adn.pdf'
		},
	9:	{
			'features' 	:	[i for i in list(data) if i.split(".")[0] in ['ang', 'x', 'y', 'vel', 'cos', 'sin', 'ADn']],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['Pos']],
			'figure'	:	'../../figures/9_PR2_feat_all_adn_targ_pos.pdf'
		}				
}

########################################################################
# MAIN LOOP
########################################################################
for k in np.sort(combination.keys()):
	features = combination[k]['features']
	targets = combination[k]['targets']	

	results = test_features(features, targets)
	
	final_data[k] = results

	plot_model_comparison(results, labels, colors)
	savefig(combination[k]['figure'], dpi=900, bbox_inches = 'tight')
	
	name = combination[k]['figure'].split("/")[-1].split(".")[0]
	with open("../data/"+name+".pickle", 'wb') as f:
		pickle.dump(final_data[k], f)


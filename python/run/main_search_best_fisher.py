#!/usr/bin/env python

'''
    File name: main_fig2.py
    Author: Guillaume Viejo
    Date created: 27/03/2017    
    Python Version: 2.7

Fig 2

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
# adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree_20ms.mat'))
# m1_imported = scipy.io.loadmat('/home/guillaume/spykesML/data/m1_stevenson_2011.mat')
for ses in os.listdir("../data/sessions/wake/"):
	adrien_data = scipy.io.loadmat(os.path.expanduser('../data/sessions/wake/'+ses))

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



	#######################################################################
	# FONCTIONS DEFINITIONS
	#######################################################################
	def extract_tree_threshold(trees):
		n = len(trees.get_dump())
		thr = {}
		for t in xrange(n):
			gv = xgb.to_graphviz(trees, num_trees=t)
			body = gv.body		
			for i in xrange(len(body)):
				for l in body[i].split('"'):
					if 'f' in l and '<' in l:
						tmp = l.split("<")
						if thr.has_key(tmp[0]):
							thr[tmp[0]].append(float(tmp[1]))
						else:
							thr[tmp[0]] = [float(tmp[1])]					
		for k in thr.iterkeys():
			thr[k] = np.sort(np.array(thr[k]))
		return thr

	def tuning_curve(x, f, nb_bins, tau = 40.0):	
		bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
		index = np.digitize(x, bins).flatten()    
		tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)])  	
		occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
		tcurve = (tcurve/occupancy)*tau
		x = bins[0:-1] + (bins[1]-bins[0])/2.
		# tcurve = tcurve
		return (x, tcurve)

	def fisher_information(x, f):
		fish = np.zeros(len(f)-1)
		slopes_ = []
		tmpf = np.hstack((f[-1],f,f[0:3]))
		binsize = x[1]-x[0]
		# very bad # to correct of the offset of x points
		tmpx = np.hstack((np.array([x[0]-binsize-(x.min()+(2*np.pi-x.max()))]),x,np.array([x[-1]+i*binsize+(x.min()+(2*np.pi-x.max())) for i in xrange(1,4)])))
		
		# plot(tmpx, tmpf, 'o')	
		for i in xrange(len(f)):
			slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(tmpx[i:i+3], tmpf[i:i+3])
			slopes_.append(slope)	
			# plot(tmpx[i:i+3], tmpx[i:i+3]*slope+intercept, '-')

		fish = np.power(slopes_, 2)
		# for i in xrange(len(fish)):
		# 	fish[i] = np.power((f[i+1]-f[i])/binsize, 2)
		# fish = fish/fish.sum()
		# fish = fish/f[0:-1]
		return (x, fish)

	for n in data.keys():
		if 'Pos' in n:
			#####################################################################
			# COMBINATIONS DEFINITION
			#####################################################################
			combination = {
				'ang':{
					n.split(".")[0]:	{
						'features' 	:	['ang'],
						'targets'	:	[n]
							}
					}
				}

			#####################################################################
			# LEARNING XGB Exemples
			#####################################################################
			params = {'objective': "count:poisson", #for poisson output
			    'eval_metric': "poisson-nloglik", #loglikelihood loss
			    'seed': 2925, #for reproducibility
			    'silent': 1,
			    'learning_rate': 0.1,
			    'min_child_weight': 2, 'n_estimators': 120,		    
			    'max_depth': 5, 'gamma': 0.5}        

			num_round = 10
			bsts = {}
			for i in combination.iterkeys():
				bsts[i] = {}
				for j in combination[i].iterkeys():
					features = combination[i][j]['features']
					targets = combination[i][j]['targets']	
					X = data[features].values
					Yall = data[targets].values		
					for k in xrange(Yall.shape[1]):
						dtrain = xgb.DMatrix(X, label=Yall[:,k])
						bst = xgb.train(params, dtrain, num_round)
						bsts[i][j] = bst

			# bsts = pickle.load(open("fig2_bsts.pickle", 'rb'))

			#####################################################################
			# TUNING CURVE
			#####################################################################
			X = data['ang'].values
			example = [n]
			tuningc = {}
			for k in example:
				Y = data[k].values
				tuningc[k.split(".")[0]] = tuning_curve(X, Y, nb_bins = 60)


			#####################################################################
			# EXTRACT TREE STRUCTURE
			#####################################################################
			thresholds = {}
			for i in bsts.iterkeys():
				thresholds[i] = {}
				for j in bsts[i].iterkeys():
					thresholds[i][j] = extract_tree_threshold(bsts[i][j])		



			from pylab import *
			

			e = n.split(".")[0]
			figure(figsize = (20, 20))
			ax = subplot(1,1,1)
			[ax.axvline(l, alpha = 0.1, color = 'grey', linewidth = 0.1) for l in thresholds['ang'][e]['f0']]	
			fisher = fisher_information(tuningc[e][0], tuningc[e][1])
			ax2 = ax.twinx()
			ax2.plot(fisher[0], fisher[1], 'o-', color = 'black', label = 'Fisher', linewidth = 1.5)	
			ax.plot(tuningc[e][0], tuningc[e][1], 'o-', color = 'blue', linewidth = 1.5)
			ax.set_xlim(0, 2*np.pi)
			ax.set_xticks([0, 2*np.pi])
			ax.set_xticklabels(('0', '$2\pi$'))
			ax.set_xlabel('Head-direction (rad)', labelpad = -3.9)
			ax.set_ylabel('Firing rate', labelpad = 2.1)		
			ax.locator_params(axis='y', nbins = 3)
			ax.locator_params(axis='x', nbins = 4)
			title(ses+"."+n)

			show()

			
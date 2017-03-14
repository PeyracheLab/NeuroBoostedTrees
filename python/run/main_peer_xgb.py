#!/usr/bin/env python

'''
    File name: main_hd_data.py
    Author: Guillaume Viejo
    Date created: 06/03/2017    
    Python Version: 2.7

To test the xbg algorithm with peer-prediction

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
import networkx as nx


#####################################################################
# DATA LOADING
#####################################################################
adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree.mat'))
# m1_imported = scipy.io.loadmat('/home/guillaume/spykesML/data/m1_stevenson_2011.mat')

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

def tuning_curve(x, f, nb_bins):	
	bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
	index = np.digitize(x, bins).flatten()    
	tcurve = np.array([np.mean(f[index == i]) for i in xrange(1, nb_bins+1)])  	
	x = bins[0:-1] + (bins[1]-bins[0])/2.
	return (x, tcurve)


#####################################################################
# COMBINATIONS DEFINITION
#####################################################################
# targets = ['Pos.8', 'Pos.9', 'Pos.10', 'ADn.9', 'ADn.10', 'ADn.11']
targets = [i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']]
combination = {}
for k in targets:
	combination[k] = {	'features'	: [i for i in list(data) if i.split(".")[0] == k.split(".")[0] and i != k],
						'targets'	: k,
						'figure'	: ''
						}

#####################################################################
# LEARNING XGB
#####################################################################

bsts = {i:{} for i in combination.iterkeys()} # to keep the boosted tree
params = {'objective': "count:poisson", #for poisson output
    'eval_metric': "logloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.05,
    'min_child_weight': 2, 'n_estimators': 580,
    'subsample': 0.6, 'max_depth': 5, 'gamma': 0.4}        
num_round = 100

for k in combination.keys():
	features = combination[k]['features']
	targets = combination[k]['targets']	
	X = data[features].values
	Yall = data[targets].values		
	dtrain = xgb.DMatrix(X, label=Yall)
	bst = xgb.train(params, dtrain, num_round)
	bsts[k] = bst

#####################################################################
# TUNING CURVE
#####################################################################
all_neurons = [i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']]
X = data['ang'].values
Yall = data[all_neurons].values
tuningc = {all_neurons[i]:tuning_curve(X, Yall[:,i], nb_bins = 100) for i in xrange(Yall.shape[1])}


#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
thresholds = {}
for i in bsts.iterkeys():
	thresholds[i] = extract_tree_threshold(bsts[i])		

# need to sort the features by the number of splits
sorted_features = dict.fromkeys(thresholds)
for k in sorted_features.iterkeys():
	count = np.array([len(thresholds[k][f]) for f in thresholds[k].iterkeys()])
	name = np.array([combination[k]['features'][int(i[1:])] for i in thresholds[k].iterkeys()])
	sorted_features[k] = [name[np.argsort(count)], np.sort(count)]

# matrix relation of splits between neurons
relation = {}
for k in ['Pos', 'ADn']:
	keys = [i for i in thresholds.iterkeys() if i.split(".")[0] == k]
	tmp = np.zeros((len(keys), len(keys)))
	for l in keys:
		for m,n in zip(sorted_features[l][0], sorted_features[l][1]):
			i = int(l.split(".")[1])
			j = int(m.split(".")[1])
			tmp[i, j] = n
	relation[k] = tmp
	#let's normalize the matrix
	relation[k] = relation[k]/np.max(relation[k])

#####################################################################
# plot 15
#####################################################################
order = ['Pos.8', 'Pos.9', 'Pos.10', 'ADn.9', 'ADn.10', 'ADn.11']

rcParams.update({   'backend':'pdf',
                    'savefig.pad_inches':0.1,
                    'font.size':8 })

figure(figsize = (12,17))

for k, i in zip(order, xrange(1,18,3)):
	subplot(6,3,i)
	count = sorted_features[k][1][::-1]
	name = sorted_features[k][0][::-1]
	bar(left = np.arange(len(count)), height = count, align = 'center', facecolor = 'grey')
	ylabel('Number of split')
	xticks(np.arange(len(count)), name, rotation = 45)
	simpleaxis(gca())

	subplot(6,3,i+1)
	norm_tuningc = tuningc[k][1]/np.max(tuningc[k][1])
	plot(tuningc[k][0], norm_tuningc, '-', color = 'black', linewidth = 2)
	a = 0.0	
	for l in sorted_features[k][0]:
		norm_tuningc = tuningc[l][1]/np.max(tuningc[l][1])
		plot(tuningc[l][0], norm_tuningc, '-', color = 'blue', alpha = a)	
		a+= 0.07
	
	title(k)
	xlim(0, 2*np.pi)
	xlabel('Angle (rad)')
	ylabel('Normalized f')
	simpleaxis(gca())
	
	subplot(6,3,i+2)
	
	center = tuningc[k][0][np.argmax(tuningc[k][1])]
	max_sorted_features = np.array([tuningc[l][0][np.argmax(tuningc[l][1])] for l in sorted_features[k][0][::-1]])			
	
	plot(np.arange(len(sorted_features[k][0])), np.abs(max_sorted_features - center), 'o-')
	ylabel('Absolute distance between \n ang direction')
	xticks(np.arange(len(name)), name, rotation = 45)	
	simpleaxis(gca())

subplots_adjust(hspace = 0.3, wspace = 0.3)

savefig('../../figures/15_XGB_peer_prediction.pdf', dpi = 900, bbox_inches = 'tight')

#####################################################################
# plot 16
#####################################################################

rcParams.update({   'backend':'pdf',
                    'savefig.pad_inches':0.1,
                    'font.size':8 })

figure(figsize = (8,10))

subplot(221)
imshow(relation["ADn"], aspect= 'auto', interpolation = 'none', origin = 'lower left')
xlabel("Features ADn")
ylabel("Target ADn")
simpleaxis(gca())

subplot(222)
imshow(relation["Pos"], aspect= 'auto', interpolation = 'none', origin = 'lower left')
xlabel("Features Pos")
ylabel("Target Pos")
simpleaxis(gca())

subplots_adjust(hspace = 0.3, wspace = 0.3)

# for the fun
subplot(223)
dist = np.dstack((np.triu(relation['ADn']).transpose(), np.tril(relation['ADn']))).mean(2)
dist += dist.transpose()
dist = 1.0-dist
G = nx.from_numpy_matrix(dist)
G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), tuple([str(i) for i in xrange(len(G.nodes()))]))))

nx.draw(G, with_labels = True)

subplot(224)
dist = np.dstack((np.triu(relation['Pos']).transpose(), np.tril(relation['Pos']))).mean(2)
dist += dist.transpose()
dist = 1.0-dist
G = nx.from_numpy_matrix(dist)
G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), tuple([str(i) for i in xrange(len(G.nodes()))]))))

nx.draw(G, with_labels = True)

savefig('../../figures/16_XGB_peer_prediction.pdf', dpi = 900, bbox_inches = 'tight')



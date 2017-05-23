#!/usr/bin/env python

'''
    File name: main_hd_data.py
    Author: Guillaume Viejo
    Date created: 06/03/2017    
    Python Version: 2.7

To test the xbg algorithm and to see the splits

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
# adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree.mat'))
adrien_data = scipy.io.loadmat(os.path.expanduser('../data/sessions/wake/boosted_tree.Mouse25-140124.mat'))

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
for i in xrange(adrien_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = adrien_data['Pos'][:,i].astype('float')
for i in xrange(adrien_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = adrien_data['ADn'][:,i].astype('float')


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

def test_features(features, targets, learners = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens']):	
	X = data[features].values
	Y = np.vstack(data[targets].values)
	Models = {method:{'PR2':[],'Yt_hat':[]} for method in learners}
	learners_ = list(learners)
	# print learners_

	for i in xrange(Y.shape[1]):
		y = Y[:,i]
		# TODO : make sure that 'ens' is the last learner
		for method in learners_:
			print('Running '+method+'...')                              
			print 'targets ', targets[i]
			Yt_hat, PR2 = fit_cv(X, y, algorithm = method, n_cv=8, verbose=1)       
			Models[method]['Yt_hat'].append(Yt_hat)
			Models[method]['PR2'].append(PR2)           

	for m in Models.iterkeys():
		Models[m]['Yt_hat'] = np.array(Models[m]['Yt_hat'])
		Models[m]['PR2'] = np.array(Models[m]['PR2'])
		
	return Models

#####################################################################
# COMBINATIONS DEFINITION
#####################################################################
combination = {	
	14: {
			'features'	:	['ang'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']],			
		},	
	}


#####################################################################
# LEARNING XGB
#####################################################################

# bsts = {i:{} for i in combination.iterkeys()} # to keep the boosted tree
# params = {'objective': "count:poisson", #for poisson output
#     'eval_metric': "logloss", #loglikelihood loss
#     'seed': 2925, #for reproducibility
#     'silent': 0,
#     'learning_rate': 0.1,
#     'min_child_weight': 2, 'n_estimators': 1,
#     'subsample': 0.6, 'max_depth': 1000, 'gamma': 0.01,
#     'reg_alpha': 0.0,
#     'reg_lambda':0.0}

# num_round = 1

# X = data['ang'].values
# Y = data['Pos.5']
# dtrain = xgb.DMatrix(np.vstack(X), label = np.vstack(Y))
# bst = xgb.train(params, dtrain, num_round)
# a = bst.get_dump()
# print a[0]
# sys.exit()

methods = ['xgb_run']
for k in np.sort(combination.keys()):
    features = combination[k]['features']
    targets = combination[k]['targets'] 

    results = test_features(features, targets, methods)
    
    

sys.exit()


#####################################################################
# TUNING CURVE
#####################################################################
X = data['ang'].values
Yall = data[[i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']]].values
tuningc = {targets[i]:tuning_curve(X, Yall[:,i], nb_bins = 100) for i in xrange(Yall.shape[1])}

sys.exit()

#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
thresholds = {}
for i in bsts.iterkeys():
	thresholds[i] = {}
	for j in bsts[i].iterkeys():
		thresholds[i][j] = extract_tree_threshold(bsts[i][j])		



#####################################################################
# plot 11 (2.1)
#####################################################################
order = ['Pos.8', 'Pos.9', 'Pos.10', 'ADn.9', 'ADn.10', 'ADn.11']
rcParams.update({   'backend':'pdf',
                    'savefig.pad_inches':0.1,
                    'font.size':8 })
figure(figsize = (12,15))
for k, i in zip(order, xrange(1,7)):
	subplot(3,2,i)
	[axvline(j, alpha = 0.1, color = 'grey') for j in thresholds[11][k]['f0']]
	plot(tuningc[k][0], tuningc[k][1])
	
	title(k)
	xlim(0, 2*np.pi)
	xlabel('Angle (rad)')
	ylabel('f')

	simpleaxis(gca())

subplots_adjust(hspace = 0.3, wspace = 0.3)

savefig(combination[11]['figure'], dpi = 900, bbox_inches = 'tight')


#####################################################################
# plot 12 (2.2)
#####################################################################

figure(figsize = (12,14))
for k, i in zip(order, xrange(1,7)):
	subplot(3,2,i)	
	[axvline(j, alpha = 0.1, color = 'grey') for j in thresholds[12][k]['f0']]
	[axhline(j, alpha = 0.1, color = 'grey') for j in thresholds[12][k]['f1']]	
	plot(data['x'].values, data['y'].values, '-', alpha = 0.3)

	xlabel('x')
	ylabel('y')
	title(k)
	simpleaxis(gca())

subplots_adjust(hspace = 0.3, wspace = 0.3)

savefig(combination[12]['figure'], dpi = 900, bbox_inches = 'tight')



#####################################################################
# plot 13 (2.3)
#####################################################################
trans = {'f0':'Ang','f1':'x','f2':'y'}

figure(figsize = (12,17))

for k, i in zip(order, xrange(1,18,3)):
	subplot(6,3,i)
	count = np.array([len(thresholds[13][k][f]) for f in thresholds[13][k].keys()])
	name = np.array([trans[f] for f in thresholds[13][k].keys()])
	bar(left = np.arange(len(count)), height = count, tick_label = name, align = 'center', facecolor = 'grey')
	ylabel('Number of split')
	simpleaxis(gca())

	subplot(6,3,i+1)
	[axvline(j, alpha = 0.1, color = 'grey') for j in thresholds[13][k]['f0']]
	plot(tuningc[k][0], tuningc[k][1])
	
	title(k)
	xlim(0, 2*np.pi)
	xlabel('Angle (rad)')
	ylabel('f')
	simpleaxis(gca())
	
	subplot(6,3,i+2)
	[axvline(j, alpha = 0.1, color = 'grey') for j in thresholds[13][k]['f1']]
	[axhline(j, alpha = 0.1, color = 'grey') for j in thresholds[13][k]['f2']]	
	plot(data['x'].values, data['y'].values, '-', alpha = 0.5)

	xlabel('x')
	ylabel('y')
	title(k)
	simpleaxis(gca())

subplots_adjust(hspace = 0.3, wspace = 0.3)

savefig(combination[13]['figure'], dpi = 900, bbox_inches = 'tight')


#####################################################################
# plot 14 (2.4)
#####################################################################
angdens = {}
for k in thresholds[14].iterkeys():
	thr = np.copy(thresholds[14][k]['f0'])
	tun = np.copy(tuningc[k])
	bins = np.linspace(0, 2.*np.pi+1e-8, 20+1)
	hist, bin_edges = np.histogram(thr, bins, normed = True)
	x = bins[0:-1] + (bins[1]-bins[0])/2.
	x[x>np.pi] -= 2*np.pi
	# correct x with offset of tuning curve
	offset = tun[0][np.argmax(tun[1])]
	if offset <= np.pi : 
		x -= offset 
	else : 
		x += (2.*np.pi - offset)
	x[x>np.pi] -= 2*np.pi
	x[x<= -np.pi] += 2*np.pi
	hist = hist[np.argsort(x)]
	angdens[k] = (np.sort(x), hist)

xydens = {}
for k in thresholds[15].iterkeys():
	xt = np.copy(thresholds[15][k]['f0'])
	yt = np.copy(thresholds[15][k]['f1'])
	xbins = np.linspace(20, 100, 20+1)
	ybins = np.linspace(0, 80, 20+1)
	xh, bin_edges = np.histogram(xt, xbins, normed = True)
	yh, bin_edges = np.histogram(yt, ybins, normed = True)
	x = ybins[0:-1] + (ybins[1]-ybins[0])/2.
	xydens[k] = (x, xh, yh)
	



rcParams.update({   'backend':'pdf',
                    'savefig.pad_inches':0.1,
                    'font.size':6 })
figure(figsize = (4, 4))
	
subplot(4,2,1)
keys = [k for k in angdens.iterkeys() if 'ADn' in k]
for k in keys:
	plot(angdens[k][0], angdens[k][1], '-', color = 'blue', linewidth = 0.4)
ylabel("Density of splits \n to center")
xlabel("Angle (rad)")
title("ADn")
simpleaxis(gca())

subplot(4,2,2)
keys = [k for k in angdens.iterkeys() if 'Pos' in k]
for k in keys:
	plot(angdens[k][0], angdens[k][1], '-', color = 'green', linewidth = 0.4)
ylabel("Density of splits \n to center")
xlabel("Angle (rad)")
title("Pos")
simpleaxis(gca())

start = 3
for i,l in zip([1, 2], ['x', 'y']):
	subplot(4,2,start)
	keys = [k for k in xydens.iterkeys() if 'ADn' in k]
	for k in keys:
		plot(xydens[k][0], xydens[k][i], '-', color = 'blue', linewidth = 0.4)
	ylabel("Density of splits")
	xlabel(l+" pos")	
	simpleaxis(gca())
	
	subplot(4,2,start+1)
	keys = [k for k in xydens.iterkeys() if 'Pos' in k]
	for k in keys:
		plot(xydens[k][0], xydens[k][i], '-', color = 'green', linewidth = 0.4)
	ylabel("Density of splits")
	xlabel(l+" pos")
	simpleaxis(gca())

	start+=2

for s in ['ADn', 'Pos']:
	subplot(4,2,start)
	keys = [k for k in xydens.iterkeys() if s in k]
	tmp = np.array([np.vstack(xydens[k][1])*xydens[k][2] for k in keys]).mean(0)

	imshow(tmp, aspect = 'auto')
	xlabel("x")
	ylabel("y")

	start+=1




subplots_adjust(hspace = 0.7, wspace = 0.7)

savefig(combination[15]['figure'], dpi = 900, bbox_inches = 'tight')


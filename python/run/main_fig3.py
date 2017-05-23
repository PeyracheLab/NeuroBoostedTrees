#!/usr/bin/env python

'''
	File name: main_fig4.py
	Author: Guillaume Viejo
	Date created: 30/03/2017    
	Python Version: 2.7


'''

import warnings
import pandas as pd
import scipy.io
import numpy as np
# Should not import fonctions if already using tensorflow for something else
import sys, os
import itertools
import cPickle as pickle
from sklearn.model_selection import KFold
import xgboost as xgb

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def xgb_run(Xr, Yr, Xt):
	# params = {'objective': "count:poisson", #for poisson output
	# 'eval_metric': "logloss", #loglikelihood loss
	# 'seed': 2925, #for reproducibility
	# 'silent': 1,
	# 'learning_rate': 0.05,
	# 'min_child_weight': 2, 'n_estimators': 580,
	# 'subsample': 0.6, 'max_depth': 5, 'gamma': 0.4}        
	params = {'objective': "count:poisson", #for poisson output
	'eval_metric': "poisson-nloglik", #loglikelihood loss
	'seed': 2925, #for reproducibility
	'silent': 1,
	'learning_rate': 0.05,
	'min_child_weight': 2, 'n_estimators': 250,
	'subsample': 0.6, 'max_depth': 40, 'gamma': 0.0}
	dtrain = xgb.DMatrix(Xr, label=Yr)
	dtest = xgb.DMatrix(Xt)
	num_round = 250
	bst = xgb.train(params, dtrain, num_round)
	Yt = bst.predict(dtest)
	return Yt

def poisson_pseudoR2(y, yhat, ynull):    
	yhat = yhat.reshape(y.shape)
	eps = np.spacing(1)
	L1 = np.sum(y*np.log(eps+yhat) - yhat)
	L1_v = y*np.log(eps+yhat) - yhat
	L0 = np.sum(y*np.log(eps+ynull) - ynull)
	LS = np.sum(y*np.log(eps+y) - y)
	R2 = 1-(LS-L1)/(LS-L0)
	return R2

def fit_cv(X, Y, algorithm, n_cv=10, verbose=1):
	"""Performs cross-validated fitting. Returns (Y_hat, pR2_cv); a vector of predictions Y_hat with the
	same dimensions as Y, and a list of pR2 scores on each fold pR2_cv.
	
	X  = input data
	Y = spiking data
	algorithm = a function of (Xr, Yr, Xt) {training data Xr and response Yr and testing features Xt}
				and returns the predicted response Yt
	n_cv = number of cross-validations folds
	
	"""
	if np.ndim(X)==1:
		X = np.transpose(np.atleast_2d(X))

	cv_kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
	skf  = cv_kf.split(X)

	i=1
	Y_hat=np.zeros(len(Y))
	pR2_cv = list()
	

	for idx_r, idx_t in skf:
		if verbose > 1:
			print( '...runnning cv-fold', i, 'of', n_cv)
		i+=1
		Xr = X[idx_r, :]
		Yr = Y[idx_r]
		Xt = X[idx_t, :]
		Yt = Y[idx_t]           
		Yt_hat = eval(algorithm)(Xr, Yr, Xt)        
		Y_hat[idx_t] = Yt_hat
		pR2 = poisson_pseudoR2(Yt, Yt_hat, np.mean(Yr))
		pR2_cv.append(pR2)
		if verbose > 1:
			print( 'pR2: ', pR2)

	if verbose > 0:
		print("pR2_cv: %0.6f (+/- %0.6f)" % (np.mean(pR2_cv),
											 np.std(pR2_cv)/np.sqrt(n_cv)))

	return Y_hat, pR2_cv

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
	tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)])    
	occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
	tcurve = (tcurve/occupancy)*200.0
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
			# print('Running '+method+'...')                              
			Yt_hat, PR2 = fit_cv(X, y, algorithm = method, n_cv=8, verbose=0)       
			Models[method]['Yt_hat'].append(Yt_hat)
			Models[method]['PR2'].append(PR2)           

	for m in Models.iterkeys():
		Models[m]['Yt_hat'] = np.array(Models[m]['Yt_hat'])
		Models[m]['PR2'] = np.array(Models[m]['PR2'])
		
	return Models


#####################################################################
# DATA LOADING | ALL SESSIONS WAKE
#####################################################################
os.system("scp -r guillaume@z620.mni.mcgill.ca:~/results_peer_fig3/ ../data/")
data = {'pr2':{}, 'bsts':{}}
for f in os.listdir("../data/results_peer_fig3/wake/"):	
	if 'pr2' in f:
		data['pr2'][f.split(".")[1]] = pickle.load(open("../data/results_peer_fig3/wake/"+f, 'rb'))
	elif 'bsts' in f:
		data['bsts'][f.split(".")[1]] = pickle.load(open("../data/results_peer_fig3/wake/"+f, 'rb'))

# only loading pr2 values for the moment
pr2_sleep = {}
for ep in ['rem', 'sws']:
	pr2_sleep[ep] = {}
	for g in ['ADn', 'Pos']:
		pr2_sleep[ep][g] = {}
		for w in ['peer', 'cros']:
			pr2_sleep[ep][g][w] = []
			for f in os.listdir("../data/results_peer_fig3/"+ep+"/"):	
				if 'pr2' in f:
					tmp = pickle.load(open("../data/results_peer_fig3/"+ep+"/"+f, 'rb'))
					pr2_sleep[ep][g][w].append(tmp[g][w]['PR2'])
			pr2_sleep[ep][g][w] = np.vstack(pr2_sleep[ep][g][w])
# pr2_sleep = pickle.load(open("../data/fig3_pr2_sleep.pickle", 'rb'))


final_data = {}
bsts = {}
corr = {}
for g in ['ADn', 'Pos']:
	final_data[g] = {}
	bsts[g] = {}
	corr[g] = {}
	for t in ['peer', 'cros']:
		final_data[g][t] = []
		bsts[g][t] = {}
		for s in data['pr2'].iterkeys():
			final_data[g][t].append(data['pr2'][s][g][t]['PR2'])
			if t=='peer':
				tmp = data['pr2'][s][g][t]['corr'].item()  # TO CHANGE
				for k in tmp.iterkeys():
					corr[g][k] = tmp[k]
			for k in data['bsts'][s][g][t].iterkeys():
				bsts[g][t][k] = data['bsts'][s][g][t][k]
		final_data[g][t] = np.vstack(final_data[g][t])


# #####################################################################
# # TUNING CURVE
# #####################################################################
tuningc = {}
for f in os.listdir("../data/results_density/wake/"):
	tmp = pickle.load(open("../data/results_density/wake/"+f))
	tmp = tmp['tuni']
	for k in tmp.iterkeys():
		tuningc[k] = tmp[k]	

#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
names = pickle.load(open("../data/fig3_names.pickle", 'rb'))

thresholds = {}
for g in ['ADn', 'Pos']:
	thresholds[g] = {}
	for w in ['peer', 'cros']:
		thresholds[g][w] = {}
		for k in bsts[g][w].iterkeys():
			thresholds[g][w][k] = extract_tree_threshold(bsts[g][w][k])		

# need to sort the features by the number of splits
sorted_features = {}
for g in thresholds.iterkeys():
	sorted_features[g] = {}
	for w in thresholds[g].iterkeys():
		sorted_features[g][w] = {}
		for k in thresholds[g][w].iterkeys(): # PREDICTED NEURONS
			count = np.array([len(thresholds[g][w][k][f]) for f in thresholds[g][w][k].iterkeys()])
			name = np.array([names[g][w][k][int(f[1:])] for f in thresholds[g][w][k].iterkeys()])
			sorted_features[g][w][k] = np.array([name[np.argsort(count)], np.sort(count)])

# number of splits versus mean firing rate
splitvar = {}
plotsplitvar = {}
for g in thresholds.iterkeys():
	splitvar[g] = {}
	plotsplitvar[g] = {}
	for w in thresholds[g].iterkeys():
		splitvar[g][w] = {}
		plotsplitvar[g][w] = {'nsplit':[], 'meanf':[]}
		for k in thresholds[g][w].iterkeys():
			mean_firing_rate = []
			for n in sorted_features[g][w][k][0]:
				mean_firing_rate.append(np.mean(tuningc[n][1]))
			mean_firing_rate = np.array(mean_firing_rate)			
			splitvar[g][w][k] = np.array([mean_firing_rate, sorted_features[g][w][k][1]])
			plotsplitvar[g][w]['meanf'].append(mean_firing_rate)
			plotsplitvar[g][w]['nsplit'].append(sorted_features[g][w][k][1].astype('float'))
		plotsplitvar[g][w]['meanf'] = np.hstack(np.array(plotsplitvar[g][w]['meanf']))
		plotsplitvar[g][w]['nsplit'] = np.hstack(np.array(plotsplitvar[g][w]['nsplit']))
		

#####################################################################
# DISTANCE TO CENTER OF FIELD
#####################################################################
distance = {}
plotdistance = {}
for g in thresholds.iterkeys():
	distance[g] = {}
	plotdistance[g] = {}
	for w in thresholds[g].iterkeys():
		distance[g][w] = {}
		plotdistance[g][w] = {'nsplit':[], 'distance':[]}
		for k in thresholds[g][w].iterkeys():
			com_neuron = tuningc[k][0][np.argmax(tuningc[k][1])]				
			com = np.array([tuningc[n][0][np.argmax(tuningc[n][1])] for n in sorted_features[g][w][k][0]])			
			dist = np.abs(com - com_neuron)
			tmp = 2*np.pi - dist[dist>np.pi]
			dist[dist>np.pi] = tmp
			plotdistance[g][w]['distance'].append(dist)
			plotdistance[g][w]['nsplit'].append(sorted_features[g][w][k][1].astype('int'))
		plotdistance[g][w]['distance'] = np.hstack(np.array(plotdistance[g][w]['distance']))
		plotdistance[g][w]['nsplit'] = np.hstack(np.array(plotdistance[g][w]['nsplit']))

#####################################################################
# PEER CORRELATION 
#####################################################################
peercorr = {}
for g in corr.iterkeys():
	peercorr[g] = []
	for k in corr[g].iterkeys():
		for n,i in zip(corr[g][k][0], xrange(len(corr[g][k][0]))):
			comn = tuningc[n][0][np.argmax(tuningc[n][1])]
			comk = tuningc[k][0][np.argmax(tuningc[k][1])]
			dist = np.abs(comn - comk) 
			if dist > np.pi: dist = 2*np.pi - dist
			if comn < comk: dist *= -1.0
			peercorr[g].append(np.array([dist, float(corr[g][k][1][i])]))
	peercorr[g] = np.array(peercorr[g])




#####################################################################
# PLOTTING
#####################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.5              # height in inches
	# fig_height = 4.696
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def myticks(x,pos):
	if x == 0: return "$0$"
	exponent = int(np.log10(x))
	coeff = x/10**exponent
	return r"${:2.0f} \times 10^{{ {:2d} }}$".format(coeff,exponent)

import matplotlib as mpl

mpl.use("pdf")



pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 7,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 6,
	"ytick.labelsize": 6,
	"figure.figsize": figsize(1),     # default fig size of 0.9 textwidth
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.5,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}    
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *

methods = ['xgb_run']

labels = {'mb_10':'MB \n 10 bins', 
			'mb_60':'MB \n 60 bins', 
			'mb_360':'MB \n 360 bins', 
			'lin_comb':'Lin', 
			'nn':'NN', 
			'xgb_run':'XGB'}

colors_ = {'ADn':'#EE6C4D', 'Pos':'#3D5A80'}


labels_plot = [labels[m] for m in methods[0:-1]]



figure(figsize = figsize(1))
outerspace = gridspec.GridSpec(1,2, width_ratios =[1.1,0.9])

#################################################################
# LEFT
#################################################################
# outer = gridspec.GridSpec(outerspace[0], height_ratios=[0.5,1.2])

# SUBPLOT 1 ################################################################
outer = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec = outerspace[0], height_ratios=[0.5, 1.3])

subplot(outer[0])
simpleaxis(gca())
y = []
err = []
x = [0.0]
color = []
for g in ['ADn', 'Pos']:            
	for w in ['peer', 'cros']:		
		# wake
		PR2_art = final_data[g][w]	
		color.append(colors_[g])
		y.append(np.mean(PR2_art))
		err.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))
		x.append(x[-1]+0.62)
		# REM / SWS
		for e in ['rem', 'sws']:
			PR2_art = pr2_sleep[e][g][w]	
			color.append(colors_[g])
			y.append(np.mean(PR2_art))
			err.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))
			x.append(x[-1]+0.62)			
		x[-1] += 0.3	
	x[-1] += 0.3
		
x = np.array(x)[0:-1]
y = np.array(y)
err = np.array(err)		
x_adn = x[0:6]
y_adn = y[0:6]
e_adn = err[0:6]
x_pos = x[6:12]
y_pos = y[6:12]
e_pos = err[6:12]

ind = [0,3]
bar(x_adn[ind], y_adn[ind], 0.4, align='center',
			ecolor='k', color = colors_['ADn'], alpha=1, ec='w', yerr=e_adn[ind], label = 'Antero Dorsal nucleus')
bar(x_pos[ind], y_pos[ind], 0.4, align='center',
			ecolor='k', color = colors_['Pos'], alpha=1, ec='w', yerr=e_pos[ind], label = 'Post Subiculum')
ind = [1,4]
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = 'white', edgecolor='black', alpha=1, hatch="////////", linewidth = 0, label = 'REM sleep')
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = colors_['ADn'], edgecolor='black', alpha=1, hatch="////////", linewidth = 0)
bar(x_pos[ind], y_pos[ind], 0.4, align='center', facecolor = colors_['Pos'], edgecolor='black', alpha=1, hatch="////////", linewidth = 0)
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = 'none', alpha=1, edgecolor='w', yerr=e_adn[ind], ecolor = 'black')
bar(x_pos[ind], y_pos[ind], 0.4, align='center', facecolor = 'none', alpha=1, edgecolor='w', yerr=e_pos[ind], ecolor = 'black')

ind = [2,5]
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = 'white', edgecolor='black', alpha=1, hatch="xxxxxx", linewidth = 0, label = 'Slow wave sleep')
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = colors_['ADn'], edgecolor='black', alpha=1, hatch="xxxxxx", linewidth = 0)
bar(x_pos[ind], y_pos[ind], 0.4, align='center', facecolor = colors_['Pos'], edgecolor='black', alpha=1, hatch="xxxxxx", linewidth = 0)
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = 'none', alpha=1, edgecolor='w', yerr=e_adn[ind], ecolor = 'black')
bar(x_pos[ind], y_pos[ind], 0.4, align='center', facecolor = 'none', alpha=1, edgecolor='w', yerr=e_pos[ind], ecolor = 'black')


plot(x, y, 'k.', markersize=3)         
locator_params(nbins=4)				
xlim(np.min(x)-0.3, np.max(x)+0.3)
ylabel('Pseudo-R2 (XGBoost)')
xticks(x[[1,4,7,10]], 
	["AD$\Rightarrow$AD", "Post-S$\Rightarrow$AD", "Post-S$\Rightarrow$Post-S", "AD$\Rightarrow$Post-S"], 
	# rotation = 30, 
	# ha = 'right'
	fontsize = 5
	)

legend(bbox_to_anchor=(0.6, 1.1), loc='upper center', ncol=2, frameon = False)

# figtext(0.2, -0.2, "ADn $\Rightarrow$ ADn \n Post-S $\Rightarrow$ Post-S \n \scriptsize{(Features $\Rightarrow$ Target)}")
# figtext(0.6, -0.14, "ADn $\Rightarrow$ Post-S \n Post-S $\Rightarrow$ ADn")

# SUBPLOT 2 ################################################################
gs = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec = outer[1], hspace = 0.5, wspace = 0.5)
matplotlib.rcParams.update({"axes.labelsize": 	4,
							"font.size": 		4,
							"legend.fontsize": 	4,
							"xtick.labelsize": 	4,
							"ytick.labelsize": 	4,   
							})               # Make the legend/label fonts a little smaller
title_ = ["ADn $\Rightarrow$ ADn \n(wake)", "Post-S $\Rightarrow$ Post-S \n(wake)"]							

count = 0
for g in plotsplitvar.keys():
	for w in ['peer']:
		subplot(gs[count])
		simpleaxis(gca())
		plot(plotdistance[g][w]['distance'], plotdistance[g][w]['nsplit'], 'o', color = colors_[g], markersize = 1)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plotdistance[g][w]['distance'], plotdistance[g][w]['nsplit'])
		print p_value
		plot(plotdistance[g][w]['distance'], plotdistance[g][w]['distance']*slope + intercept, '--', color = 'black', linewidth = 0.9)

		locator_params(nbins=2)				
		# ticklabel_format(style='sci', axis='x', scilimits=(0,0), fontsize = 4)
		# ticklabel_format(style='sci', axis='y', scilimits=(0,0), fontsize = 4)
		xticks([0, np.pi], ['0', '$\pi$'], fontsize = 4)
		yticks(fontsize = 4)		
		xlabel("Angular distance", fontsize = 4, labelpad = 0.4)				
		ylabel("Number of splits", fontsize = 4)
		title(title_[count], fontsize = 4)#, loc = 'left', y = 1.3)
		xlim(0, np.pi)


		subplot(gs[count+2])
		simpleaxis(gca())		
		plot(plotsplitvar[g][w]['meanf'], plotsplitvar[g][w]['nsplit'], 'o', color = colors_[g], markersize = 1)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plotsplitvar[g][w]['meanf'], plotsplitvar[g][w]['nsplit'])
		plot(plotsplitvar[g][w]['meanf'], plotsplitvar[g][w]['meanf']*slope + intercept, '--', color = 'black', linewidth = 0.9)
		print p_value
		xticks(fontsize = 4)
		yticks(fontsize = 4)		
		xlabel("Mean firing rate", fontsize = 4, labelpad = 0.5)
		ylabel("Number of splits", fontsize = 4)		


		subplot(gs[count+4])
		simpleaxis(gca())		
		# plot(plotsplitvar[g][w]['meanf'], plotsplitvar[g][w]['nsplit'], 'o', color = colors_[g], markersize = 1)
		# for k in splitvar[g][w].keys():
		# 	plot(splitvar[g][w][k][1], splitvar[g][w][k][0], '-', color = colors_[g], markersize = 1, alpha = 0.4)
		plot(peercorr[g][:,0], peercorr[g][:,1], 'o', color = colors_[g], markersize = 1)
		# locator_params(nbins=2)					
		# ticklabel_format(style='sci', axis='x', scilimits=(0,0), fontsize = 4)
		# ticklabel_format(style='sci', axis='y', scilimits=(0,0), fontsize = 4)
		xticks(fontsize = 4)
		yticks(fontsize = 4)		
		xlabel("Angular distance", fontsize = 4, labelpad = 0.5)
		ylabel("R", fontsize = 4)
		

		count += 1

#################################################################
# RIGHT
#################################################################
outer = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec = outerspace[1])

subplot(outer[0])

plot(np.random.rand(100))

subplot(outer[1])

plot(np.random.rand(100))

subplot(outer[2])

plot(np.random.rand(100))


savefig("../../figures/fig3.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig3.pdf &")

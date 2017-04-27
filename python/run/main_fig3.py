#!/usr/bin/env python

'''
    File name: main_fig3.py
    Author: Guillaume Viejo
    Date created: 28/03/2017    
    Python Version: 2.7

fig3.py

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
combination = {
	'1.ADn':	{
			'features' 	:	['ang'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] == 'ADn']
		},
	'1.Pos':	{
			'features' 	:	['ang'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] == 'Pos']
		},		
	'2.ADn':	{
			'features' 	:	['ang', 'x', 'y'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] == 'ADn']
		},
	'2.Pos':	{
			'features' 	:	['ang', 'x', 'y'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] == 'Pos']
		},	
	'3.ADn':	{
			'features' 	:	['x', 'y'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] == 'ADn']
		},
	'3.Pos':	{
			'features' 	:	['x', 'y'],
			'targets'	:	[i for i in list(data) if i.split(".")[0] == 'Pos']
		}
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
    'min_child_weight': 2, 'n_estimators': 10,
    'subsample': 0.6, 'max_depth': 400, 'gamma': 0.4}        
num_round = 400

for k in combination.keys():
	features = combination[k]['features']
	targets = combination[k]['targets']	
	X = data[features].values
	Yall = data[targets].values	
	for i in xrange(Yall.shape[1]):
		dtrain = xgb.DMatrix(X, label=Yall[:,i])
		bst = xgb.train(params, dtrain, num_round)
		bsts[k][targets[i]] = bst

#####################################################################
# TUNING CURVE
#####################################################################
X = data['ang'].values
tuningc = {}
alln = [i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']]
for t in alln:
	Y = data[t].values
	tuningc[t] = tuning_curve(X, Y, nb_bins = 60)


#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
thresholds = {}
for i in bsts.iterkeys():
	thresholds[i] = {}
	for j in bsts[i].iterkeys():
		thresholds[i][j] = extract_tree_threshold(bsts[i][j])		


#####################################################################
# DENSITY OF SPLIT TO CENTER
#####################################################################
angdens = {}
mean_angdens = {}
for g in thresholds.iterkeys():	
	if int(g.split(".")[0]) in [1,2]:  
		angdens[g] = {}
		mean_angdens[g] = []
		for k in thresholds[g].iterkeys():
			thr = np.copy(thresholds[g][k]['f0'])
			tun = np.copy(tuningc[k])
			# correct thr with offset of tuning curve
			offset = tun[0][np.argmax(tun[1])]
			thr -= offset 
			thr[thr<= -np.pi] += 2*np.pi
			thr[thr> np.pi] -= 2*np.pi
			if thr.max() > np.pi or thr.min() < -np.pi:
				print "ERror"
				sys.exit()
					
			bins = np.linspace(-np.pi, np.pi+1e-8, 20+1)
			hist, bin_edges = np.histogram(thr, bins, density = False)
			hist = hist/float(hist.sum())
			x = bin_edges[0:-1] + (bin_edges[1]-bin_edges[0])/2.
			x[x>np.pi] -= 2*np.pi
			hist = hist[np.argsort(x)]
			angdens[g][k] = (np.sort(x), hist)
			mean_angdens[g].append(hist)
		mean_angdens[g] = (x, np.mean(mean_angdens[g], 0))


xydens = {}
mean_xydens = {}
for g in thresholds.iterkeys():
	if int(g.split(".")[0]) in [2,3]:  
		xydens[g] = {}
		mean_xydens[g] = {'x':[], 'y':[]}
		for k in thresholds[g].iterkeys():
			xt = np.copy(thresholds[g][k]['f0'])
			yt = np.copy(thresholds[g][k]['f1'])
			# let's normalize xt and yt
			xt -= xt.min()
			xt /= xt.max()
			yt -= yt.min()
			yt /= yt.max()			
			bins = np.linspace(0, 1, 20+1)
			xh, bin_edges = np.histogram(xt, bins, density = False)
			yh, bin_edges = np.histogram(yt, bins, density = False)
			xh = xh/float(xh.sum())
			yh = yh/float(yh.sum())			
			x = bin_edges[0:-1] + (bin_edges[1]-bin_edges[0])/2.
			xydens[g][k] = (x, xh, yh)
			mean_xydens[g]['x'].append(xh)
			mean_xydens[g]['y'].append(yh)
		mean_xydens[g]['x'] = np.mean(mean_xydens[g]['x'], 0)
		mean_xydens[g]['y'] = np.mean(mean_xydens[g]['y'], 0)
		
ratio = {}
for g in ['2.Pos', '2.ADn']:	
	ratio[g.split('.')[1]] = {}
	for k in thresholds[g].iterkeys():		
		ratio[g.split('.')[1]][k] = np.array([len(thresholds[g][k][f]) for f in thresholds[g][k].iterkeys()])
		ratio[g.split('.')[1]][k] = ratio[g.split('.')[1]][k]/float(np.sum(ratio[g.split('.')[1]][k]))




#####################################################################
# PLOTTING
#####################################################################
def figsize(scale):
    fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean *0.5          # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)


import matplotlib as mpl

mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 6,               # LaTeX default is 10pt font.
    "font.size": 7,
    "legend.fontsize": 6,               # Make the legend/label fonts a little smaller
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

colors_ = ['#134B64', '#F5A21E']
title_ = {	'ADn':'Antero-dorsal nucleus',
			'Pos':'Post-subiculum'	}




figure(figsize = figsize(1))

for g,i in zip(['1.ADn', '1.Pos'], xrange(2)):
	subplot(1,3,i+1)
	subplots_adjust(wspace = 0.5)
	simpleaxis(gca())
	for k in angdens[g].iterkeys():
		plot(angdens[g][k][0], angdens[g][k][1], '-', color = colors_[i], linewidth = 0.4, alpha = 0.5)

	plot(mean_angdens[g][0], mean_angdens[g][1], '-', color = colors_[i], linewidth = 1.2, alpha = 1)	
	# plot(mean_angdens['2.'+g.split('.')[1]][0], mean_angdens['2.'+g.split('.')[1]][1], ':', color = colors_[i], linewidth = 1, alpha = 1)

	ylabel("Density of angular splits")
	# xlabel("Angle (rad)")
	title(title_[g.split(".")[1]])
	xlim(-np.pi, np.pi)
	xticks([-np.pi, 0, np.pi], ('$-\pi$', '0', '$\pi$'))
	xlabel('Centered', labelpad = 0.4)

subplot(1,3,3)
simpleaxis(gca())
x = np.arange(3, dtype = float)
for g, i in zip(ratio.iterkeys(), xrange(2)):
	mean = []
	for k in ratio[g].iterkeys():
		plot(x, ratio[g][k], 'o', alpha = 0.5, color = colors_[i], markersize = 2)
		mean.append(ratio[g][k])
	mean = np.mean(mean, 0)
	bar(x, mean, 0.4, align='center',
        ecolor='k', alpha=.9, color=colors_[i], ec='w')

	x += 0.41
ylabel('Density of splits')

xticks(np.arange(3)+0.205, ('Angle','x pos','y pos'))





# subplots_adjust(hspace = 0.7, wspace = 0.7)

savefig('../../figures/fig3.pdf', dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig3.pdf &")

#!/usr/bin/env python
"""
    Fig 2 of the article : splits position within the tuning curve of the head-direction neuron
    File name: main_fig2_splits.py
    Date created: 27/03/2017    
    Python Version: 2.7    
    Copyright (C) 2017 Guillaume Viejo

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import warnings
import pandas as pd
import scipy.io
import numpy as np
import sys, os
import itertools
import cPickle as pickle
import xgboost as xgb

#####################################################################
# DATA LOADING
#####################################################################
hd_data = scipy.io.loadmat('data_test_boosted_tree_20ms.mat')

#####################################################################
# DATA ENGINEERING
#####################################################################
data 			= 	pd.DataFrame()
data['time'] 	= 	np.arange(len(hd_data['Ang']))		
data['ang'] 	= 	hd_data['Ang'].flatten() 												# angular direction of the animal head
data['x'] 		= 	hd_data['X'].flatten() 													# x position of the animal 
data['y'] 		= 	hd_data['Y'].flatten() 													# y position of the animal
data['vel'] 	= 	hd_data['speed'].flatten() 												# velocity of the animal 
# Engineering features
data['cos']		= 	np.cos(hd_data['Ang'].flatten())										# cosinus of angular direction
data['sin']		= 	np.sin(hd_data['Ang'].flatten())										# sinus of angular direction
# Firing data
for i in xrange(hd_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = hd_data['Pos'][:,i]		# Neurons from Post-subiculum
for i in xrange(hd_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = hd_data['ADn'][:,i]		# Neurons from Antero-dorsal nucleus

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def extract_tree_threshold(trees):
	""" Take BST TREE and return a dict = {features index : [splits position 1, splits position 2, ...]}
	"""
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
	""" Compute Fisher Information over the tuning curves
		x : array of angular position
		f : firing rate
		return (angular position, fisher information)
	"""
	fish = np.zeros(len(f)-1)
	slopes_ = []
	tmpf = np.hstack((f[-1],f,f[0:3]))
	binsize = x[1]-x[0]	
	tmpx = np.hstack((np.array([x[0]-binsize-(x.min()+(2*np.pi-x.max()))]),x,np.array([x[-1]+i*binsize+(x.min()+(2*np.pi-x.max())) for i in xrange(1,4)])))		
	for i in xrange(len(f)):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(tmpx[i:i+3], tmpf[i:i+3])
		slopes_.append(slope)			
	fish = np.power(slopes_, 2)
	fish = fish/(f+1e-4)
	return (x, fish)

#####################################################################
# COMBINATIONS DEFINITION
#####################################################################
combination = {
	'ang':{
		'ADn':	{
			'features' 	:	['ang'],
			'targets'	:	['ADn.8']
				},
		'Pos':	{
			'features' 	:	['ang'],
			'targets'	:	['Pos.9']
				},		
		},
	}

#####################################################################
# LEARNING XGB Exemples
#####################################################################
params = {'objective': "count:poisson", #for poisson output
	'eval_metric': "logloss", #loglikelihood loss
	'seed': 2925, #for reproducibility
	'silent': 1,
	'learning_rate': 0.01,
	'min_child_weight': 2,
	'max_depth': 5}        

num_round = 30
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

#####################################################################
# TUNING CURVE
#####################################################################
X = data['ang'].values
example = [combination['ang'][k]['targets'][0] for k in ['ADn', 'Pos']]
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


########################################################################
# PLOTTING
########################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean            # height in inches
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 5,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 5,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 5,
	"ytick.labelsize": 5,
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
from mpl_toolkits.axes_grid.inset_locator import inset_axes


trans = {'f0':'Angle','f1':'x pos','f2':'y pos'}
colors_ = ['#EE6C4D', '#3D5A80']
title_ = ['Antero-dorsal nucleus', 'Post-subiculum']


figure(figsize = figsize(1))
# subplots_adjust(hspace = 0.4, right = 0.4)
# outer = gridspec.GridSpec(2,2, height_ratios = [1, 0.6])
outer = gridspec.GridSpec(2,2, width_ratios = [1.6,0.7], wspace = 0.2, hspace = 0.3)

##PLOT 1#################################################################################################################
# Examples subplot 1 et 2
gs = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outer[0], wspace = 0.4)
limts = [(1.5,5),(0.7, 3.6)]
for e, i in zip(['ADn','Pos'],range(2)):			
	ax = subplot(gs[i])
	simpleaxis(ax)	
	[ax.axvline(l, alpha = 0.9, color = 'grey', linewidth = 0.2) for l in np.unique(thresholds['ang'][e]['f0'])[0:100]]	
	fisher = fisher_information(tuningc[e][0], tuningc[e][1])
	ax2 = ax.twinx()
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)	
	ax2.plot(fisher[0], fisher[1], '-', color = '#25393C', label = 'Fisher \n Information', linewidth = 1.0)
	ax2.set_yticks([])
	ax.plot(tuningc[e][0], tuningc[e][1], color = colors_[i], linewidth = 1.5)
	# ax.set_xlim(limts[i])
	# ax.set_xticks([0, 2*np.pi])
	# ax.set_xticklabels(('0', '$2\pi$'))
	ax.set_xlabel('Head-direction (rad)', labelpad = 0.0)
	ax.set_ylabel('Firing rate (Hz)', labelpad = 2.1)		
	ax.locator_params(axis='y', nbins = 3)
	ax.locator_params(axis='x', nbins = 4)
	if j == 0:
		ax.set_title(title_[i], loc = 'right')
	leg = ax2.legend(fontsize = 4, loc = 'best')
	leg.get_frame().set_linewidth(0.0)
	# leg.get_frame().set_facecolor('white')
	ax.set_title(title_[i], fontsize = 6)


savefig("fig2_splits.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince fig2_splits.pdf &")



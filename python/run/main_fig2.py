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
adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree_20ms.mat'))
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

def tuning_curve(x, f, nb_bins, tau = 50.0):	
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
	binsize = x[1]-x[0]
	for i in xrange(len(fish)):
		fish[i] = np.power((f[i+1]-f[i])/binsize, 2)
	fish = fish/fish.sum()
	return (x[0:-1], fish)

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
			'targets'	:	['Pos.8']
				},		
		},
	# 'angxy': 	{
	# 	'ADn':		{
	# 		'features' 	:	['ang', 'x', 'y'],
	# 		'targets'	:	['ADn.8']
	# 				},
	# 	'Pos':		{
	# 		'features' 	:	['ang', 'x', 'y'],
	# 		'targets'	:	['Pos.8']
	# 				},				
	# 			},
	'xy': 	{
		'ADn':		{
			'features' 	:	['x', 'y'],
			'targets'	:	['ADn.8']
					},
		'Pos':		{
			'features' 	:	['x', 'y'],
			'targets'	:	['Pos.8']
					},				
				}				
	}

#####################################################################
# LEARNING XGB Exemples
#####################################################################
params = {'objective': "count:poisson", #for poisson output
    'eval_metric': "logloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.05,
    'min_child_weight': 2, 'n_estimators': 150,
    'subsample': 0.6, 'max_depth': 4, 'gamma': 0.0}        
num_round = 150
bsts = {}
# for i in combination.iterkeys():
# 	bsts[i] = {}
# 	for j in combination[i].iterkeys():
# 		features = combination[i][j]['features']
# 		targets = combination[i][j]['targets']	
# 		X = data[features].values
# 		Yall = data[targets].values		
# 		for k in xrange(Yall.shape[1]):
# 			dtrain = xgb.DMatrix(X, label=Yall[:,k])
# 			bst = xgb.train(params, dtrain, num_round)
# 			bsts[i][j] = bst

bsts = pickle.load(open("fig2_bsts.pickle", 'rb'))

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
# DENSITY PICKLE LOAD
########################################################################
all_data = pickle.load(open("../data/fig2_density.pickle", 'rb'))
angdens = all_data['angdens']
mean_angdens = all_data['mean_angdens']
ratio = all_data['ratio']
twod = all_data['twod_xydens']

########################################################################
# CORRELATION
########################################################################
corr = {'ADn':[], 'Pos':[]}
for g in corr.iterkeys():
	for n in angdens['1.'+g].iterkeys():
		tun = tuning_curve(data['ang'].values, data[n].values, nb_bins = 20)		
		fis = fisher_information(tun[0], tun[1])[1] # TODO
		fis = np.hstack((fis, fis[0]))
		dens = angdens['1.'+g][n][1]
		corr[g].append(scipy.stats.pearsonr(fis, dens)[0])



########################################################################
# PLOTTING
########################################################################
def figsize(scale):
    fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean*0.85            # height in inches
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

order = [['ADn.9', 'ADn.10', 'ADn.11'], ['Pos.8', 'Pos.9', 'Pos.10']]
trans = {'f0':'Angle','f1':'x pos','f2':'y pos'}
colors_ = ['#330174', '#249f87']
title_ = ['Antero-dorsal nucleus', 'Post-subiculum']








figure(figsize = figsize(1))
# subplots_adjust(hspace = 0.4, right = 0.4)
# outer = gridspec.GridSpec(2,2, height_ratios = [1, 0.6])
outer = gridspec.GridSpec(2,2, width_ratios = [1.6,0.7], wspace = 0.2, hspace = 0.3)

##PLOT 1#################################################################################################################
# Examples subplot 1 et 2
gs = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outer[0], wspace = 0.4)

for e, i in zip(['ADn','Pos'],range(2)):			
	ax = subplot(gs[i])
	simpleaxis(ax)
	[ax.axvline(l, alpha = 0.1, color = 'grey', linewidth = 0.1) for l in thresholds['ang'][e]['f0']]	
	fisher = fisher_information(tuningc[e][0], tuningc[e][1])
	ax2 = ax.twinx()
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)	
	ax2.plot(fisher[0], fisher[1], '-', color = 'black', label = 'Fisher', linewidth = 0.5)
	ax2.set_yticks([])
	ax.plot(tuningc[e][0], tuningc[e][1], color = colors_[i], linewidth = 1.5)
	ax.set_xlim(0, 2*np.pi)
	ax.set_xticks([0, 2*np.pi])
	ax.set_xticklabels(('0', '$2\pi$'))
	ax.set_xlabel('Head-direction (rad)', labelpad = -3.9)
	ax.set_ylabel('Firing rate', labelpad = 2.1)		
	ax.locator_params(axis='y', nbins = 3)
	ax.locator_params(axis='x', nbins = 4)
	if j == 0:
		ax.set_title(title_[i], loc = 'right')
	leg = ax2.legend(fontsize = 4)
	leg.get_frame().set_linewidth(0.0)
	# leg.get_frame().set_facecolor('white')
	ax.set_title(title_[i])

##PLOT 2#################################################################################################################
# Centered density
gs = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outer[2], wspace = 0.4)
tmp = []
for g,i in zip(['1.ADn', '1.Pos'], xrange(2)):
	ax = subplot(gs[i])	
	simpleaxis(ax)
	for k in angdens[g].iterkeys():
		plot(angdens[g][k][0], angdens[g][k][1]*100.0, '-', color = colors_[i], linewidth = 0.4, alpha = 0.5)

	plot(mean_angdens[g][0], mean_angdens[g][1]*100.0, '-', color = colors_[i], linewidth = 1.2, alpha = 1)		
	axhline(5, linestyle = '--', color = 'black', linewidth=0.6)
	ylabel("Density of angular splits$(\%)$")
	xlim(-np.pi, np.pi)
	xticks([-np.pi, 0, np.pi], ('$-\pi$', '0', '$\pi$'))
	xlabel('Centered', labelpad = 0.4)
	locator_params(axis='y', nbins = 5)
	locator_params(axis='x', nbins = 5)
	#inset	
	tmp.append(ax.get_position().bounds)


ai = axes([tmp[0][0]+tmp[0][2]*0.7,tmp[0][1]+tmp[0][3]*0.8, 0.06, 0.07])
ai.get_xaxis().tick_bottom()
ai.get_yaxis().tick_left()
n, bins, patches = ai.hist(corr['ADn'], 8, normed = 1, facecolor = 'white', edgecolor = colors_[0])
ai.set_xlim(-1, 1)
ai.set_xticks([-1, 0, 1])
ai.set_xticklabels([-1,'',1])
ai.set_yticks([])
ai.set_title("Corr(Fisher)", fontsize = 4, position = (0.5, 0.9))
ai.set_xlabel("$R^2$", fontsize = 4, labelpad = -2.4)

aii = axes([tmp[1][0]+tmp[1][2]*0.7,tmp[1][1]+tmp[1][3]*0.8, 0.06, 0.07])
aii.get_xaxis().tick_bottom()
aii.get_yaxis().tick_left()
n, bins, patches = aii.hist(corr['Pos'], 8, normed = 1, facecolor = 'white', edgecolor = colors_[1])
aii.set_xlim(-1,1)
aii.set_xticks([-1, 0, 1])
aii.set_xticklabels([-1,'',1])
aii.set_yticks([])
aii.set_title("Corr(Fisher)", fontsize = 4, position = (0.5, 0.9))
aii.set_xlabel("$R^2$", fontsize = 4, labelpad = -2.4)







##PLOT 3#################################################################################################################
# x y split
gs = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = outer[1], wspace = 0.5, hspace = 0.5)
title2 = ['AD', 'Post-S']
for e, i in zip(['ADn','Pos'],range(2)):			
	ax = subplot(gs[i])
	simpleaxis(ax)	
	[ax.axvline(l, alpha = 0.1, color = colors_[i], linewidth = 0.1) for l in thresholds['xy'][e]['f0']]
	[ax.axhline(l, alpha = 0.1, color = colors_[i], linewidth = 0.1) for l in thresholds['xy'][e]['f1']]	
	ax.plot(data['x'].values[0:20000], data['y'].values[0:20000], '-', color = 'black', alpha = 1, linewidth = 0.5)				
	ax.set_xlabel('x pos', labelpad = 0.4)
	ax.set_ylabel('y pos')
	ax.set_xticks([])
	ax.set_yticks([])	
	ax.set_xlim(thresholds['xy'][e]['f0'].min(), thresholds['xy'][e]['f0'].max())
	ax.set_ylim(thresholds['xy'][e]['f1'].min(), thresholds['xy'][e]['f1'].max())
	ax.set_title(title2[i], fontsize = 5)

for e, i in zip(['ADn','Pos'],range(2,4)):			
	ax = subplot(gs[i])
	simpleaxis(ax)	
	im = ax.imshow(twod['3.'+e].transpose(), origin = 'lower', interpolation = 'nearest', aspect=  'equal', cmap = 'viridis')
	ax.set_xlabel('x pos', labelpad = 0.4)
	ax.set_ylabel('y pos')
	ax.set_xticks([])
	ax.set_yticks([])		
#  	cbar = colorbar(im, orientation = 'horizontal', fraction = 0.05, pad = 0.4, ticks=[np.min(twod['3.'+e]), np.max(twod['3.'+e])])
# 	cbar.set_ticklabels([np.min(twod['3.'+e]), np.max(twod['3.'+e])])
# 	cbar.ax.tick_params(labelsize = 4)
# 	cbar.update_ticks()

ax.set_title('Density of (x,y) splits $(\%)$', fontsize=5, position = (-0.35, 0.95))

##PLOT 4#################################################################################################################
# ang x y density
gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[3])

subplot(gs[0])
simpleaxis(gca())
x = np.arange(2, dtype = float)
for g, i in zip(ratio.iterkeys(), xrange(2)):
	tmp = []
	for k in ratio[g].iterkeys():
		# plot(x, ratio[g][k], 'o', alpha = 0.5, color = colors_[i], markersize = 2)
		tmp.append(ratio[g][k])
	tmp = np.array(tmp)
	mean = [np.mean(tmp[:,0]), np.mean(tmp[:,1:])]	
	sem = [np.std(tmp[:,0])/np.sqrt(np.size(tmp[:,0])),np.std(tmp[:,1:])/np.sqrt(np.size(tmp[:,1:].flatten()))]
	bar(x, mean, 0.4, yerr = sem, align='center',
        ecolor='k', alpha=.9, color=colors_[i], ec='w')
	# xticks([])
	# yticks([])

	x += 0.41

locator_params(axis='y', nbins = 5)
ylabel('Density of splits $(\%)$')
xticks(np.arange(2)+0.205, ('Angle','Position'))



# # COunt 
# for i in xrange(2):
# 	gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[i+2])		
# 	ax = subplot(gs[0])
# 	simpleaxis(ax)
# 	group = order[i][0].split(".")[0]
# 	for neuron in thresholds[group].iterkeys():
# 		count = np.array([len(thresholds[group][neuron][f]) for f in thresholds[group][neuron].iterkeys()])
# 		name = np.array([trans[f] for f in thresholds[group][neuron].keys()])
# 		ax.plot(np.arange(len(count)), count, 'o--', 
# 			color = colors_[i], 
# 			markersize = 1.8, 
# 			linewidth = 0.3,
# 			markerfacecolor = colors_[i],
# 			markeredgecolor = colors_[i],
# 			alpha = 0.6)
# 	for j in xrange(3):
# 		neuron = order[i][j]
# 		count = np.array([len(thresholds[group][neuron][f]) for f in thresholds[group][neuron].keys()])
# 		name = np.array([trans[f] for f in thresholds[neuron.split(".")[0]][neuron].keys()])
# 		ax.plot(np.arange(len(count)), count, 'o-', 
# 			color = colors_[i], 
# 			markersize = 3, 
# 			linewidth = 1.2,
# 			markerfacecolor = colors_[i],
# 			markeredgecolor = colors_[i]
# 			)

	
# 	ax.set_ylabel('Number of split')
# 	ax.set_xticks(np.arange(3))
# 	ax.set_xticklabels(tuple(name))
# 	ax.set_xlabel('Feature')
# 	ax.set_xlim(-0.2, 2.2)
# 	ax.locator_params(axis='y', nbins = 3)


savefig("../../figures/fig2.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig2.pdf &")



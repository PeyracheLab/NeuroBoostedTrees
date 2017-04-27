#!/usr/bin/env python

'''
	File name: main_tsne.py
	Author: Guillaume Viejo
	Date created: 25/04/2017    
	Python Version: 2.7

test of t-sne 



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
luke_data = {}
maindir = os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HPC-NAc/')
folder = os.listdir(maindir)

animals = np.unique([f.split("_")[0] for f in folder])


toload = {}
for a in animals:    
	subfold = [f for f in folder if a in f]
	# where are pre and post file
	pre = []
	post = []
	for b in subfold:
		files = os.listdir(maindir+b)
		for i in files:
			if 'data_hipp_nac_' in i:
				if 'pre' in i:
					pre.append(maindir+b+'/'+i)
				elif 'post' in i:
					post.append(maindir+b+'/'+i)    

	toload[a] = {
		'pre':pre[0],
		'post':post[0]
	}

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

def test_features(features, targets, learners = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens']):
	'''
		Main function of the script
		Return : dictionnary with for each models the score PR2 and Yt_hat
	'''
	X = data[features].values
	Y = np.vstack(data[targets].values)
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
# MAIN LOOP
########################################################################
methods = ['xgb_run']
bsts = {}
combination = {}
params = {'objective': "count:poisson", #for poisson output
    'eval_metric': "logloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.05,
    'min_child_weight': 2, 'n_estimators': 10,
    'subsample': 0.6, 'max_depth': 400, 'gamma': 0.4}        
num_round = 400

# for a in animals:
for a in ['BF3-3']:
	bsts[a] = {}
	combination[a] = {}
	for s in ['pre', 'post']:
		bsts[a][s] = {}
		combination[a][s] = {}
		#####################################################################
		# DATA ENGINEERING
		#####################################################################
		luke_data = scipy.io.loadmat(toload[a][s])
		data = 	pd.DataFrame()
		# Firing data
		for i in xrange(luke_data['hpc'].shape[1]): data['hpc'+'.'+str(i)] = luke_data['hpc'][:,i]
		for i in xrange(luke_data['nac'].shape[1]): data['nac'+'.'+str(i)] = luke_data['nac'][:,i]
		#####################################################################
		# COMBINATIONS DEFINITION
		#####################################################################		
		targets = [i for i in list(data) if i.split(".")[0] == 'nac']
		features = [i for i in list(data) if i.split(".")[0] == 'hpc']
		for k in targets:
			combination[a][s][k] = 	{ 	'features'	: features,
									'targets'	: k
								}									
		
		#####################################################################
		# LEARNING BOOSTER
		#####################################################################        
		for k in combination[a][s].iterkeys():				
			features = combination[a][s][k]['features']
			target =  combination[a][s][k]['targets'] 					
			X = data[features].values
			Yall = data[target].values		
			dtrain = xgb.DMatrix(X, label=Yall)
			bst = xgb.train(params, dtrain, num_round)
			bsts[a][s][k] = bst

#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
thresholds = {}
for a in bsts.iterkeys():
	thresholds[a] = {}
	for s in ['pre', 'post']:
		thresholds[a][s] = {}
		for k in bsts[a][s].iterkeys():	
			thresholds[a][s][k] = extract_tree_threshold(bsts[a][s][k])		

# need to sort the features by the number of splits
sorted_features = {}
for a in bsts.iterkeys():
	sorted_features[a] = {}
	for s in ['pre', 'post']:
		sorted_features[a][s] = {}
		for k in bsts[a][s].iterkeys():
			count = np.array([len(thresholds[a][s][k][f]) for f in thresholds[a][s][k].iterkeys()])
			name = np.array([combination[a][s][k]['features'][int(f[1:])] for f in thresholds[a][s][k].iterkeys()])            			
			sorted_features[a][s][k] = np.array([name[np.argsort(count)], np.sort(count)])

#####################################################################
# T-SNE
#####################################################################
# need to build a matrix N * N (neurons * number of splits)
# need to assign to each neuron an index
# first hpc then nac
group = {}
labels = {}
for a in sorted_features.iterkeys():
	group[a] = {}
	labels[a] = {}
	for s in ['pre', 'post']:
		allneurons = combination[a][s].keys()
		allneurons += combination[a][s][combination[a][s].keys()[0]]['features']
		index = {}		
		for n in allneurons:
			if 'hpc' in n:
				index[n] = int(n.split(".")[1])
		tmp = np.max(index.values())+1
		for n in allneurons:
		 	if 'nac' in n:
		 		index[n] = int(n.split(".")[1])+tmp

		N = len(index.keys())
		distance = np.zeros((N,N))
		for n in allneurons:
			if 'nac' in n:		
				for j,i in zip(sorted_features[a][s][n][0],sorted_features[a][s][n][1]):
					distance[index[n],index[j]] = float(i)

		distance = 1.0 - distance/distance.max()
		group[a][s] = tsne(distance, no_dims = 2, initial_dims = distance.shape[1], perplexity = 100.0)
		labels[a][s] = np.ones(N)
		labels[a][s][tmp:] = 2.0



figure()
subplot(121)
plot(group[a]['pre'][labels[a]['pre']==1,0], group[a]['pre'][labels[a]['pre']==1,1], 'o')
plot(group[a]['pre'][labels[a]['pre']==2,0], group[a]['pre'][labels[a]['pre']==2,1], '*')
subplot(122)
plot(group[a]['post'][labels[a]['post']==1,0], group[a]['post'][labels[a]['post']==1,1], 'o')
plot(group[a]['post'][labels[a]['post']==2,0], group[a]['post'][labels[a]['post']==2,1], '*')

show()

		
sys.exit()



#####################################################################
# PLOTTING
#####################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean              # height in inches
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
	"axes.labelsize": 10,               # LaTeX default is 10pt font.
	"font.size": 8,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 7,
	"ytick.labelsize": 7,
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




figure(figsize = figsize(0.5))

simpleaxis(gca())
y = []
err = []
x = [0.0]
color = []

for g in ['cros']:
	for m in ['xgb_run', 'nn']:		
		PR2_art = []    		
		for n in final_data.iterkeys():
			PR2_art.append(final_data[n][g][m]['PR2'])
		y.append(np.mean(PR2_art))
		err.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))
		x.append(x[-1]+0.42)
		x[-1] += 0.2
	x[-1] += 0.5
		
x = np.array(x)
y = np.array(y)
err = np.array(err)		


bar(x[0:-1], y, 0.4, align='center',
			ecolor='k', color = 'grey', alpha=.9, ec='w', yerr=err, label = 'Nucleus Accumbeuns ')

plot(x[0:-1], y, 'k.', markersize=3)         

xlim(np.min(x)-0.5, np.max(x[0:-1])+0.5)
ylabel('Pseudo-R2')
xticks(x[0:-1], ['XGB', 'NN']*2)

legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=2, frameon = False)

figtext(0.4, -0.1, "Hpc $\Rightarrow$ Acc \n \scriptsize{(Features $\Rightarrow$ Target)}")


savefig("../../figures/fig4_bis.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig4_bis.pdf &")

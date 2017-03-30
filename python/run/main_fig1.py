#!/usr/bin/env python

'''
    File name: main_fig1.py
    Author: Guillaume Viejo
    Date created: 27/03/2017    
    Python Version: 2.7

Fig 1 of the article

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
data            =   pd.DataFrame()
data['time']    =   np.arange(len(adrien_data['Ang']))      # TODO : import real time from matlab script
data['ang']     =   adrien_data['Ang'].flatten()            # angular direction of the animal head
data['x']       =   adrien_data['X'].flatten()              # x position of the animal 
data['y']       =   adrien_data['Y'].flatten()              # y position of the animal
data['vel']     =   adrien_data['speed'].flatten()          # velocity of the animal 
# Engineering features
data['cos']     =   np.cos(adrien_data['Ang'].flatten())    # cosinus of angular direction
data['sin']     =   np.sin(adrien_data['Ang'].flatten())    # sinus of angular direction
# Firing data
for i in xrange(adrien_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = adrien_data['Pos'][:,i]
for i in xrange(adrien_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = adrien_data['ADn'][:,i]


methods = ['mb_10', 'mb_60', 'mb_360', 'xgb_run', 'nn', 'lin_comb']
colors=['#F5A21E', '#02A68E', '#EF3E34', '#134B64', '#FF07CD','b']

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
    'Pos':  {
            'features'  :   ['ang'],
            'targets'   :   [i for i in list(data) if i.split(".")[0] == 'Pos'], 
            },          
    'ADn':  {
            'features'  :   ['ang'],
            'targets'   :   [i for i in list(data) if i.split(".")[0] == 'ADn'],
            }
}

# ########################################################################
# # MAIN LOOP
# ########################################################################
# for k in np.sort(combination.keys()):
#     features = combination[k]['features']
#     targets = combination[k]['targets'] 

#     results = test_features(features, targets, methods)
    
#     final_data[k] = results

########################################################################
# PLOTTING
########################################################################
with open("../data/fig1.pickle", 'rb') as f:
    final_data = pickle.load(f)

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


labels = {'mb_10':'MB \n 10 bins', 
            'mb_60':'MB \n 60 bins', 
            'mb_360':'MB \n 360 bins', 
            'lin_comb':'Lin', 
            'nn':'NN', 
            'xgb_run':'XGB'}

colors = ['#F5A21E']*3 + ['#134B64', '#02A68E', '#FF07CD']

figure(figsize = figsize(0.5))

simpleaxis(gca())

# plot([-1, len(methods)], [0,0],'--k', alpha=0.4)


labels_plot = [labels[m] for m in methods[0:-1]]


mean_pR2 = list()
sem_pR2 = list()
for model in methods[0:-1]:        
    PR2_art = final_data['ADn'][model]['PR2']
    mean_pR2.append(np.mean(PR2_art))
    sem_pR2.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))        

bar(np.arange(np.size(mean_pR2)), mean_pR2, 0.4, align='center',
        ecolor='k', alpha=.9, color='#134B64', ec='w', yerr=np.array(sem_pR2), label = 'Antero-dorsal nucleus')
plot(np.arange(np.size(mean_pR2)), mean_pR2, 'k.', markersize=5)

mean_pR2 = list()
sem_pR2 = list()
for model in methods[0:-1]:        
    PR2_art = final_data['Pos'][model]['PR2']
    mean_pR2.append(np.mean(PR2_art))
    sem_pR2.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))        

bar(np.arange(np.size(mean_pR2))+0.41, mean_pR2, 0.4, align='center',
        ecolor='k', alpha=.9, color='#F5A21E', ec='w', yerr=np.array(sem_pR2), label = 'Post-subiculum')
plot(np.arange(np.size(mean_pR2))+0.41, mean_pR2, 'k.', markersize=5)

legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=2, frameon = False)
xlim(-0.5, 4.5)
# ylim(0.0, 0.8)
xticks(np.arange(np.size(mean_pR2))+0.205, labels_plot)
ylabel('Pseudo-R2')


savefig("../../figures/fig1.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig1.pdf &")
    




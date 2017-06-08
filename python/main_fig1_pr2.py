#!/usr/bin/env python
"""
    Fig 1 of the article : pseudo-R2 score for head-direction signal
    File name: main_fig1_pr2.py
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
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.linear_model import LinearRegression

#####################################################################
# DATA LOADING
#####################################################################
hd_data = scipy.io.loadmat('data_test_boosted_tree_20ms.mat')


#####################################################################
# DATA ENGINEERING
#####################################################################
data            =   pd.DataFrame()
data['time']    =   np.arange(len(hd_data['Ang']))      
data['ang']     =   hd_data['Ang'].flatten()                                            # angular direction of the animal head
data['x']       =   hd_data['X'].flatten()                                              # x position of the animal 
data['y']       =   hd_data['Y'].flatten()                                              # y position of the animal
data['vel']     =   hd_data['speed'].flatten()                                          # velocity of the animal 
# Engineering features
data['cos']     =   np.cos(hd_data['Ang'].flatten())                                    # cosinus of angular direction
data['sin']     =   np.sin(hd_data['Ang'].flatten())                                    # sinus of angular direction
# Firing data
for i in xrange(hd_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = hd_data['Pos'][:,i]  
for i in xrange(hd_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = hd_data['ADn'][:,i]  

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def kernel(Xr, Yr, Xt):
    """ Kernel model, cos and sin of the angle to the linear model
        Return the predicted firing rate
    """
    newX = np.zeros((Xr.shape[0], 12)) # up to six order    
    newXt = np.zeros((Xt.shape[0], 12))
    for i, j in zip(xrange(1,7), xrange(0, 12, 2)):
        newX[:,j] = np.cos(i*Xr).flatten()
        newX[:,j+1] = np.sin(i*Xr).flatten()
        newXt[:,j] = np.cos(i*Xt).flatten()
        newXt[:,j+1] = np.sin(i*Xt).flatten()
    Yt = lin_comb(newX, Yr, newXt)    
    return Yt

def xgb_run(Xr, Yr, Xt):
    """ XGBoost model
        Return the predicted firing rate
    """
    params = {'objective': "count:poisson", #for poisson output
    'eval_metric': "poisson-nloglik", #loglikelihood loss
    'seed': 2925,
    'silent': 1,
    'learning_rate': 0.1,
    'min_child_weight': 2, 
    # 'subsample': 0.6, 
    'max_depth': 5, 
    'gamma': 0.5
    }    
    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)
    num_round = 120
    bst = xgb.train(params, dtrain, num_round)
    Yt = bst.predict(dtest)
    return Yt

def lin_comb(Xr, Yr, Xt):
    """ Linear model
        Return the predicted firing rate
    """
    lr = LinearRegression()
    lr.fit(Xr, Yr)
    Yt = lr.predict(Xt)    
    #rectify outputs
    Yt = np.maximum(Yt,np.zeros(Yt.shape))
    return Yt 

def mb(Xr, Yr, Xt):
    """ Compute the model-based tuning curve of the hd cells
        And return the predicted firing rate        
    """
    nb_bins = 60
    bins = np.linspace(np.vstack((Xr, Xt)).min(), np.vstack((Xr, Xt)).max()+1e-8, nb_bins+1)
    index = np.digitize(Xr, bins).flatten()    
    tcurve = np.array([np.sum(Yr[index == i]) for i in xrange(1, nb_bins+1)])
    occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
    tcurve = (tcurve/occupancy)*40.0  
    new_index = np.digitize(Xt, bins).flatten()    
    return tcurve[new_index-1]/40.0 

def poisson_pseudoR2(y, yhat, ynull):
    """ Compute the pseudo R2 score
        y : real firing rate
        yhat : predicted firing rate
        ynull : mean of the real firing rate
    """    
    yhat = yhat.reshape(y.shape)
    eps = np.spacing(1)
    L1 = np.sum(y*np.log(eps+yhat) - yhat)
    L1_v = y*np.log(eps+yhat) - yhat
    L0 = np.sum(y*np.log(eps+ynull) - ynull)
    LS = np.sum(y*np.log(eps+y) - y)
    R2 = 1-(LS-L1)/(LS-L0)
    return R2

def fit_cv(X, Y, algorithm, n_cv=10):
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
    Y_hat=np.zeros(len(Y))
    pR2_cv = list()
    for idx_r, idx_t in skf:
        Xr = X[idx_r, :]
        Yr = Y[idx_r]
        Xt = X[idx_t, :]
        Yt = Y[idx_t]           
        Yt_hat = eval(algorithm)(Xr, Yr, Xt)        
        Y_hat[idx_t] = Yt_hat
        pR2 = poisson_pseudoR2(Yt, Yt_hat, np.mean(Yr))
        pR2_cv.append(pR2)

    return Y_hat, pR2_cv

def test_features(features, targets, learners):
    '''
    features : a list of features, for example ['ang', 'x', 'y']
    targets :  the list of neurons to predict ['ADn.0', 'ADn.1']
    learners : the list of model to use, for example ['xgb']        
    Return : dictionnary with for each models the score PR2 and Yt_hat (the prediction of the firing rate)
    '''
    X = data[features].values
    Y = data[targets].values    
    output = {method:{'PR2':[],'Yt_hat':[]} for method in learners}
    learners_ = list(learners)
    for i in xrange(Y.shape[1]):
        y = Y[:,i]        
        for method in learners_:        
            print('Running '+method+'...')                              
            Yt_hat, PR2 = fit_cv(X, y, algorithm = method, n_cv=8)       
            output[method]['Yt_hat'].append(Yt_hat)
            output[method]['PR2'].append(PR2)           
    for m in output.iterkeys():
        output[m]['Yt_hat'] = np.array(output[m]['Yt_hat'])
        output[m]['PR2'] = np.array(output[m]['PR2'])        
    return output

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

########################################################################
# MAIN LOOP
########################################################################
methods = ['mb', 'xgb_run', 'lin_comb', 'kernel']
final_data = {}
for g in np.sort(combination.keys()):
    features = combination[g]['features']
    targets = combination[g]['targets'] 

    results = test_features(features, targets, methods)    
    final_data[g] = results

########################################################################
# PLOTTING
########################################################################
def figsize(scale):
    fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean             # height in inches
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


labels = {  'mb':'MB \n 60 bins', 
            'lin_comb':'Linear', 
            'nn':'NN', 
            'xgb_run':"$\mathbf{XGB}$",
            'h1':"Linear \n (cos$\\theta$, sin$\\theta$)",
            'kernel':"$6^{rd}$ order \n kernel"}

colors=['#F5A21E', '#02A68E', '#EF3E34', '#134B64', '#FF07CD','b']

figure(figsize = figsize(1))
subplots_adjust()
# subplots_adjust(hspace = 0.2, right = 0.999)
subplot(1,1,1)
simpleaxis(gca())
methods_to_plot = ['mb', 'xgb_run', 'lin_comb', 'kernel']
labels_plot = [labels[m] for m in methods_to_plot]
mean_pR2 = list()
sem_pR2 = list()
for model in methods_to_plot:
    PR2_art = final_data['ADn'][model]['PR2']
    mean_pR2.append(np.mean(PR2_art))
    sem_pR2.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))        
bar(np.arange(np.size(mean_pR2)), mean_pR2, 0.4, align='center',
        ecolor='k', alpha=1, color='#EE6C4D', ec='w', yerr=np.array(sem_pR2), label = 'Antero-dorsal nucleus')
plot(np.arange(np.size(mean_pR2)), mean_pR2, 'k.', markersize=5)
mean_pR2 = list()
sem_pR2 = list()

for model in methods_to_plot:
    PR2_art = final_data['Pos'][model]['PR2']
    mean_pR2.append(np.mean(PR2_art))
    sem_pR2.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))
bar(np.arange(np.size(mean_pR2))+0.405, mean_pR2, 0.4, align='center',
        ecolor='k', alpha=1, color='#3D5A80', ec='w', yerr=np.array(sem_pR2), label = 'Post-subiculum')
plot(np.arange(np.size(mean_pR2))+0.41, mean_pR2, 'k.', markersize=5)
plot([-1, len(methods_to_plot)], [0,0],'--k', alpha=0.4)
legend(loc='upper center', ncol=2, frameon = False)
xlim(-0.5,)
# ylim(0.0, 0.8)
xticks(np.arange(np.size(mean_pR2))+0.205, labels_plot)
ylabel("$Pseudo-R^2$")


savefig("fig1_pr2.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince fig1_pr2.pdf &")
    




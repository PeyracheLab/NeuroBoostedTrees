#!/usr/bin/env python

'''
    File name: main_ang_grid_search.py
    Author: Guillaume Viejo
    Date created: 02/05/2017    
    Python Version: 2.7

Grid search of the best parameters

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
import math

#####################################################################
# DATA LOADING
#####################################################################
adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree.mat'))
# m1_imported = scipy.io.loadmat('/home/guillaume/spykesML/data/m1_stevenson_2011.mat')

grid = {}


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
def poisson_nloglik(y, yp):
    tmp = np.array([math.lgamma(i+1.0) for i in y])
    eps = 1e-16
    yp[yp<eps] = eps
    return np.sum(tmp + yp - np.log(yp) * y)

def xgb_run_param(X, Y, max_depth, num_round, n_cv=8, verbose=1):
    params = {  'objective': "count:poisson", #for poisson output
                'eval_metric': "poisson-nloglik", #loglikelihood loss
                'seed': 2925, #for reproducibility
                'silent': 1,
                'learning_rate': 0.05,
                'min_child_weight': 2, 'n_estimators': 580,
                'subsample': 0.6, 'max_depth': max_depth, 'gamma': 0.4}    

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

        dtrain = xgb.DMatrix(Xr, label=Yr)
        dtest = xgb.DMatrix(Xt)

        bst = xgb.train(params, dtrain, num_round)
        
        
        Y_hat[idx_t] = bst.predict(dtest)

    return Y_hat

def grid_search(features, targets):
    '''
        Main function of the script
        Return : dictionnary with for each models the score PR2 and Yt_hat
    '''
    X = data[features].values
    Y = data[targets].values    
    
    max_depth_step = 2**np.arange(1,11)
    max_trees_step = np.array([5,20,40,80,100,150,200,250,300,350,400,500])
    # max_depth_step = 2**np.arange(1,5)
    # max_trees_step = np.array([5,10,20,30])    

    grid_results = np.zeros((Y.shape[1],len(max_depth_step),len(max_trees_step)))
    

    # for i in xrange(Y.shape[1]):
    for i in [2]:
        y = Y[:,i]        
        for j in xrange(len(max_depth_step)):
            for k in xrange(len(max_trees_step)):
                
                d = max_depth_step[j]
                t = max_trees_step[k]
                Yt_hat = xgb_run_param(X, y, d, t, n_cv=8, verbose=1)
                
                LogL = poisson_nloglik(y, Yt_hat)
                bic = 2.0*LogL + np.log(float(y.shape[0]))*d*t 

                print LogL, np.log(float(y.shape[0]))*d*t , bic
                
                grid_results[i,j,k] = bic
    
    return grid_results



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
grid = {'ADn':[], 'Pos':[]}
for g in combination.iterkeys():
    for n in combination[g]['targets']:
        grid[g].append(pickle.load(open("../data/results_grid/grid_search_ang_"+n+".pickle", 'rb'))[n])
    grid[g] = np.array(grid[g])
    

max_depth_step = 2**np.arange(1,11)
max_trees_step = np.array([5,20,40,80,100,150,200,250,300,350,400,500])

dt = (np.vstack(max_depth_step)*max_trees_step).astype('float')
penalty = np.log(22863.0)*dt
dtnew = (np.vstack(max_depth_step)+max_trees_step).astype('float')
newpenalty = np.log(22863.0)*dtnew

allbic = {} 
best = []
for g in grid.iterkeys():
    bic = grid[g]
    log = bic - penalty
    bic = log + newpenalty

    for i in xrange(len(bic)):
        d, t = np.where(bic[i] == bic[i].min())
        best.append([max_depth_step[d[0]], max_trees_step[t[0]]])

    log_ = log.mean(0)
    bic_ = bic.mean(0)
    allbic[g] = bic_

best= np.array(best)
allbic['best'] = best

with open("../data/grid_search_ang_adn_pos.pickle", 'wb') as f:
    pickle.dump(allbic, f)

from matplotlib import *
from pylab import *

imshow(allbic['Pos'], origin = 'lower', interpolation = 'nearest', aspect = 'auto')
yticks(np.arange(len(max_depth_step)), max_depth_step)
xticks(np.arange(len(max_trees_step)), max_trees_step)
ylabel("depth")
xlabel("n trees")
show()

sys.exit()



#!/usr/bin/env python

'''
    File name: cluster_ang_grid_search.py
    Author: Guillaume Viejo
    Date created: 04/05/2017    
    Python Version: 2.7

Grid search of the best parameters
To use on guillimin
minimal import
'''

import os, sys
import pandas as pd
import scipy.io
import numpy as np
import cPickle as pickle
from sklearn.model_selection import KFold
import xgboost as xgb
import math
from optparse import OptionParser

#######################################################################
# ARGUMENT MANAGER                                                                    
#######################################################################
if not sys.argv[1:]:
   sys.stdout.write("Sorry: you must specify at least 1 argument")
   sys.stdout.write("More help avalaible with -h or --help option")
   sys.exit(0)
parser = OptionParser()
parser.add_option("-n", "--neuron", action="store", help="The name of the target neuron", default=False)
parser.add_option("-s", "--session", action="store", help="The name of the session", default=False)
(options, args) = parser.parse_args()
#######################################################################



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
                'learning_rate': 0.1,
                'min_child_weight': 2, 
                'n_estimators':  num_round,
                # 'subsample': 0.6, 
                'max_depth': max_depth, 
                'gamma': 0.5}    

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
    
    # max_depth_step = 2**np.arange(1,11)
    # max_depth_step = np.arange(0, 100, 10)
    max_depth_step = [3,5,8,10,12,15,20,25,30,40]    
    max_trees_step = [10, 50, 100, 150, 200, 500, 1000, 1500, 2000, 3000]

    grid_results = np.zeros((Y.shape[1],len(max_depth_step),len(max_trees_step)))
    

    for i in xrange(Y.shape[1]):    
        y = Y[:,i]        
        for j in xrange(len(max_depth_step)):
            for k in xrange(len(max_trees_step)):
                
                d = max_depth_step[j]
                t = max_trees_step[k]
                Yt_hat = xgb_run_param(X, y, d, t, n_cv=8, verbose=1)                
                LogL = poisson_nloglik(y, Yt_hat)
                bic = 2.0*LogL + np.log(float(y.shape[0]))*(d+t)                
                grid_results[i,j,k] = bic
                
    return grid_results

#####################################################################
# DATA LOADING
#####################################################################
adrien_data = scipy.io.loadmat(os.path.expanduser('~/sessions/wake/'+options.session))

grid = {}


#####################################################################
# DATA ENGINEERING
#####################################################################
data            =   pd.DataFrame(index=np.arange(len(adrien_data['Ang'])))
data['time']    =   np.vstack(np.arange(len(adrien_data['Ang'])))      # TODO : import real time from matlab script
data['ang']     =   adrien_data['Ang'].flatten()            # angular direction of the animal head
data['x']       =   adrien_data['X'].flatten()              # x position of the animal 
data['y']       =   adrien_data['Y'].flatten()              # y position of the animal
# data['vel']     =   adrien_data['speed'].flatten()          # velocity of the animal 
# Engineering features
# data['cos']     =   np.cos(adrien_data['Ang'].flatten())    # cosinus of angular direction
# data['sin']     =   np.sin(adrien_data['Ang'].flatten())    # sinus of angular direction
# Firing data
for i in xrange(adrien_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = adrien_data['Pos'][:,i]
for i in xrange(adrien_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = adrien_data['ADn'][:,i]




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
features = combination[options.neuron.split(".")[0]]['features']
# targets = combination[k]['targets'] 
grid[options.neuron] = grid_search(features, np.array([options.neuron]))[0]
    

    
with open("/home/viejo/results_grid_sessions/grid_search_ang_"+options.session.split(".")[1]+"_"+options.neuron+".pickle", 'wb') as f:
    pickle.dump(grid, f)



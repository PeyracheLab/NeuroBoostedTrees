#!/usr/bin/env python

'''
    File name: cluster_ang_pr2.py
    Author: Guillaume Viejo
    Date created: 04/05/2017    
    Python Version: 2.7

cluster_ang_pr2
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
from sklearn.linear_model import LinearRegression

#######################################################################
# ARGUMENT MANAGER                                                                    
#######################################################################
if not sys.argv[1:]:
   sys.stdout.write("Sorry: you must specify at least 1 argument")
   sys.stdout.write("More help avalaible with -h or --help option")
   sys.exit(0)
parser = OptionParser()
parser.add_option("-e", "--episode", action="store", help="The episode type", default=False)
parser.add_option("-s", "--session", action="store", help="The name of the session", default=False)
(options, args) = parser.parse_args()
#######################################################################


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
    tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)])    
    occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
    tcurve = (tcurve/occupancy)*40.0
    x = bins[0:-1] + (bins[1]-bins[0])/2.    
    return (x, tcurve)

#####################################################################
# DATA LOADING
#####################################################################
adrien_data = scipy.io.loadmat(os.path.expanduser('~/sessions/'+options.episode+'/'+options.session))

#####################################################################
# DATA ENGINEERING
#####################################################################
data            =   pd.DataFrame(index=np.arange(len(adrien_data['Ang'])))
data['time']    =   np.arange(len(adrien_data['Ang']))      # TODO : import real time from matlab script
data['ang']     =   adrien_data['Ang'].flatten()            # angular direction of the animal head
data['x']       =   adrien_data['X'].flatten()              # x position of the animal 
data['y']       =   adrien_data['Y'].flatten()              # y position of the animal

# Firing data
for i in xrange(adrien_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = adrien_data['Pos'][:,i]
for i in xrange(adrien_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = adrien_data['ADn'][:,i]

########################################################################
# COMBINATIONS DEFINITIONS
########################################################################
combination = {
    '1.ADn':    {
            'features'  :   ['ang'],
            'targets'   :   [i for i in list(data) if i.split(".")[0] == 'ADn']
        },
    '1.Pos':    {
            'features'  :   ['ang'],
            'targets'   :   [i for i in list(data) if i.split(".")[0] == 'Pos']
        },      
    '2.ADn':    {
            'features'  :   ['x', 'y', 'ang'],
            'targets'   :   [i for i in list(data) if i.split(".")[0] == 'ADn']
        },
    '2.Pos':    {
            'features'  :   ['x', 'y', 'ang'],
            'targets'   :   [i for i in list(data) if i.split(".")[0] == 'Pos']
        },  
    '3.ADn':    {
            'features'  :   ['x', 'y'],
            'targets'   :   [i for i in list(data) if i.split(".")[0] == 'ADn']
        },
    '3.Pos':    {
            'features'  :   ['x', 'y'],
            'targets'   :   [i for i in list(data) if i.split(".")[0] == 'Pos']
    }
}

#####################################################################
# LEARNING XGB
#####################################################################   
params = {'objective': "count:poisson", #for poisson output
    'eval_metric': "poisson-nloglik", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.1,
    'min_child_weight': 2, 
    'n_estimators': 1000,
    # 'subsample': 0.5, 
    'max_depth': 5, 
    'gamma': 0.5}        
num_round = 1000
bsts = {}
session = options.session.split(".")[1]
for k in combination.keys():
    features = combination[k]['features']
    targets = combination[k]['targets'] 
    X = data[features].values
    Yall = data[targets].values 
    bsts[k] = {}
    for i in xrange(Yall.shape[1]):
        print k, options.session+"."+targets[i]
        dtrain = xgb.DMatrix(X, label=Yall[:,i])
        bst = xgb.train(params, dtrain, num_round)
        bsts[k][session+"."+targets[i]] = bst

#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
thresholds = {}
for i in bsts.iterkeys():
    thresholds[i] = {}
    for j in bsts[i].iterkeys():
        thresholds[i][j] = extract_tree_threshold(bsts[i][j])       


#####################################################################
# TUNING CURVE
#####################################################################
X = data['ang'].values
alln = [i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']]
tuningc = {}
for t in alln:
    Y = data[t].values
    tuningc[session+"."+t] = tuning_curve(X, Y, nb_bins = 60)


all_data = {'thr':thresholds,
            'tuni':tuningc}

with open("/home/viejo/results_density_fig2/"+options.episode+"/dens."+options.session.split('.')[1]+".pickle", 'wb') as f:
    pickle.dump(all_data, f)

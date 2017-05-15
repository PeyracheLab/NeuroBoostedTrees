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
def kernel(Xr, Yr, Xt):
    newX = np.zeros((Xr.shape[0], 6)) # up to third order    
    newXt = np.zeros((Xt.shape[0], 6))
    for i, j in zip(xrange(1,4), xrange(0, 6, 2)):
        newX[:,j] = np.cos(i*Xr).flatten()
        newX[:,j+1] = np.sin(i*Xr).flatten()
        newXt[:,j] = np.cos(i*Xt).flatten()
        newXt[:,j+1] = np.sin(i*Xt).flatten()
    Yt = lin_comb(newX, Yr, newXt)    
    return Yt

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
    'min_child_weight': 2, 'n_estimators': 150,
    'subsample': 0.6, 'max_depth': 4, 'gamma': 0.0}
    
    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)

    num_round = 150
    bst = xgb.train(params, dtrain, num_round)

    Yt = bst.predict(dtest)
    return Yt

def lin_comb(Xr, Yr, Xt):
    lr = LinearRegression()
    lr.fit(Xr, Yr)
    Yt = lr.predict(Xt)
    
    #rectify outputs
    Yt = np.maximum(Yt,np.zeros(Yt.shape))
    return Yt 

def mb_60(Xr, Yr, Xt):
    nb_bins = 60
    bins = np.linspace(np.vstack((Xr, Xt)).min(), np.vstack((Xr, Xt)).max()+1e-8, nb_bins+1)
    index = np.digitize(Xr, bins).flatten()    
    tcurve = np.array([np.sum(Yr[index == i]) for i in xrange(1, nb_bins+1)])
    occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
    tcurve = (tcurve/occupancy)*200.0  
    new_index = np.digitize(Xt, bins).flatten()    
    return tcurve[new_index-1]/200.0 

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

def test_features(features, targets, learners = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens']):
    '''
        Main function of the script
        Return : dictionnary with for each models the score PR2 and Yt_hat
    '''
    X = data[features].values
    Y = data[targets].values    
    Models = {method:{'PR2':[],'Yt_hat':[]} for method in learners}
    learners_ = list(learners)
    for i in xrange(Y.shape[1]):
        y = Y[:,i]        
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
# DATA LOADING
#####################################################################
# adrien_data = scipy.io.loadmat(os.path.expanduser('../data/sessions/'+options.episode+'/'+options.session))
adrien_data = scipy.io.loadmat(os.path.expanduser('~/sessions/'+options.episode+'/'+options.session))
#####################################################################
# DATA ENGINEERING
#####################################################################
data            =   pd.DataFrame(index=np.arange(len(adrien_data['Ang'])))
data['time']    =   np.arange(len(adrien_data['Ang']))      # TODO : import real time from matlab script
data['ang']     =   adrien_data['Ang'].flatten()            # angular direction of the animal head
data['x']       =   adrien_data['X'].flatten()              # x position of the animal 
data['y']       =   adrien_data['Y'].flatten()              # y position of the animal
# Engineering features
data['cos']     =   np.cos(adrien_data['Ang'].flatten())    # cosinus of angular direction
data['sin']     =   np.sin(adrien_data['Ang'].flatten())    # sinus of angular direction

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

methods = ['mb_60', 'xgb_run', 'lin_comb', 'kernel']
final_data = {}

for k in np.sort(combination.keys()):
    features = combination[k]['features']
    targets = combination[k]['targets'] 

    results = test_features(features, targets, methods)
    
    final_data[k] = results    


with open("/home/viejo/results_pr2_fig1/"+options.episode+"/pr2."+options.session.split('.')[1]+".pickle", 'wb') as f:
    pickle.dump(final_data, f)



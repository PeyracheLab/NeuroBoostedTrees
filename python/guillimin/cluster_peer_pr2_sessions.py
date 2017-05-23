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
# FONCTIONS DEFINITIONS
#######################################################################
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
    'learning_rate': 0.1,
    'min_child_weight': 2, 
    'n_estimators': 100,
    # 'subsample': 0.5,
    'max_depth': 5, 
    'gamma': 0.5}
    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    Yt = bst.predict(dtest)
    return Yt

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

    i=0
    Y_hat=np.zeros(len(Y))
    pR2_cv = list()    


    for idx_r, idx_t in skf:
        if verbose > 1:
            print( '...runnning cv-fold', i, 'of', n_cv)        
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
        i+=1

    if verbose > 0:
        print("pR2_cv: %0.6f (+/- %0.6f)" % (np.mean(pR2_cv),
                                             np.std(pR2_cv)/np.sqrt(n_cv)))

    return Y_hat, pR2_cv

def test_features(features, targets, learners = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens']):  
    X = data[features].values
    Y = np.vstack(data[targets].values)
    Models = {method:{'PR2':[],'Yt_hat':[]} for method in learners}
    learners_ = list(learners)
    # print learners_

    real_corr = np.zeros(X.shape[1])
    pred_corr = np.zeros(X.shape[1])

    for i in xrange(Y.shape[1]):
        y = Y[:,i]
        for method in learners_:
            # print('Running '+method+'...')                              
            Yt_hat, PR2 = fit_cv(X, y, algorithm = method, n_cv=8, verbose=0)       
            Models[method]['Yt_hat'].append(Yt_hat)
            Models[method]['PR2'].append(PR2)           
            # calling for correlation 
            # correlation between each Xt column and Yt/Yt_hat
            for n in xrange(X.shape[1]):
                real_corr[n] = scipy.stats.pearsonr(X[:,n], y)[0]
                pred_corr[n] = scipy.stats.pearsonr(X[:,n], Yt_hat)[0]

    diffcorre = real_corr - pred_corr


    for m in Models.iterkeys():
        Models[m]['Yt_hat'] = np.array(Models[m]['Yt_hat'])
        Models[m]['PR2'] = np.array(Models[m]['PR2'])
        Models[m]['corr'] = np.vstack([features, diffcorre])
        
    return Models


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

#####################################################################
# DATA LOADING | ALL SESSIONS WAKE
#####################################################################
final_data = {g:{
    k:{'PR2':[], 'Yt_hat':[], 'corr':{}} for k in ['peer', 'cros']
} for g in ['ADn', 'Pos']}
bsts = {g:{k:{} for k in ['peer', 'cros']} for g in ['ADn', 'Pos']}
adrien_data = scipy.io.loadmat(os.path.expanduser('~/sessions/'+options.episode+'/'+options.session))
#####################################################################
# DATA ENGINEERING
#####################################################################
data            =   pd.DataFrame(index=np.arange(len(adrien_data['Ang'])))
data['time']    =   np.arange(len(adrien_data['Ang']))      # TODO : import real time from matlab script

# Firing data
for i in xrange(adrien_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = adrien_data['Pos'][:,i]
for i in xrange(adrien_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = adrien_data['ADn'][:,i]

#####################################################################
# COMBINATIONS DEFINITION
#####################################################################
combination = {}
targets = [i for i in list(data) if i.split(".")[0] in ['Pos', 'ADn']]

for n in ['Pos', 'ADn']:            
    combination[n] = {'peer':{},'cros':{}}  
    sub = [i for i in list(data) if i.split(".")[0] == n]
    for k in sub:
        combination[n]['peer'][k] = {   'features'  : [i for i in sub if i != k],
                                        'targets'   : k
                                    }
        combination[n]['cros'][k] = {   'features'  : [i for i in targets if i.split(".")[0] != k.split(".")[0]],
                                        'targets'   : k
                                    }       
########################################################################
# NEED TO WRITE NAME OF features
########################################################################
names = pickle.load(open("../z620/fig3_names.pickle", 'rb'))
for g in ['ADn', 'Pos']:
    for w in ['peer', 'cros']:        
        for k in combination[g][w].keys():
            ses = options.session.split(".")[1]
            names[g][w][ses+"."+k] = [ses+"."+n for n in combination[g][w][k]['features']]
names = pickle.dump(names, open("../z620/fig3_names.pickle", 'wb'))

########################################################################
# MAIN LOOP FOR R2
########################################################################
methods = ['xgb_run']
for g in combination.iterkeys():            
    for w in combination[g].iterkeys():             
        for k in combination[g][w].iterkeys():
            print options.session, g, w, k
            features = combination[g][w][k]['features']
            targets =  combination[g][w][k]['targets']             
            results = test_features(features, targets, methods)
            for i in xrange(len(results['xgb_run']['corr'][0])):
                results['xgb_run']['corr'][0][i] = options.session.split(".")[1]+"."+results['xgb_run']['corr'][0][i]
                
            final_data[g][w]['PR2'].append(results['xgb_run']['PR2'][0])
            final_data[g][w]['Yt_hat'].append(results['xgb_run']['Yt_hat'][0])
            final_data[g][w]['corr'][options.session.split(".")[1]+"."+k] = results['xgb_run']['corr']
        

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
    'max_depth': 6, 
    'gamma': 0.5}   
num_round = 95

for g in combination.iterkeys():            
    for w in combination[g].iterkeys():             
        for k in combination[g][w].iterkeys():
            features = combination[g][w][k]['features']
            targets =  combination[g][w][k]['targets']  
            X = data[features].values
            Yall = data[targets].values     
            dtrain = xgb.DMatrix(X, label=Yall)
            bst = xgb.train(params, dtrain, num_round)
            bsts[g][w][options.session.split(".")[1]+"."+k] = bst

for g in final_data.iterkeys():
    for w in final_data[g].iterkeys():
        for s in ['PR2', 'Yt_hat']:
            final_data[g][w][s] = np.array(final_data[g][w][s])

pickle.dump(final_data, open("/home/guillaume/results_peer_fig3/"+options.episode+"/peer_pr2."+options.session.split(".")[1]+".pickle", 'wb'))
pickle.dump(bsts, open("/home/guillaume/results_peer_fig3/"+options.episode+"/peer_bsts."+options.session.split(".")[1]+".pickle", 'wb'))            
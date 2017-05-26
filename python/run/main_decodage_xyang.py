#!/usr/bin/env python

'''
    File name: main_fig5.py
    Author: Guillaume Viejo
    Date created: 12/05/2017    
    Python Version: 2.7

Plot figure 5 for decodage

'''

import warnings
import pandas as pd
import scipy.io
import numpy as np
# Should not import fonctions if already using tensorflow for something else
import sys, os
import itertools
import cPickle as pickle
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.linear_model import LinearRegression

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def xgb_decodage(Xr, Yr, Xt):      
    # order is [ang, x, y]
    nbins_ang = 10
    nbins_xy = 10
    index = np.arange(nbins_ang*nbins_xy*nbins_xy).reshape(nbins_ang,nbins_xy,nbins_xy)
    # binning ang in nbins_ang classes
    angbins = np.linspace(0, 2*np.pi+1e-8, nbins_ang+1)
    angindex = np.digitize(Yr[:,0], angbins).flatten()-1
    # binning pos in 20 classes
    posbins = np.linspace(0, 1+1e-8, nbins_xy+1)
    xposindex = np.digitize(Yr[:,1], posbins).flatten()-1
    yposindex = np.digitize(Yr[:,2], posbins).flatten()-1
    # setting class from index
    clas = np.zeros(Yr.shape[0])
    for i in xrange(Yr.shape[0]):
        clas[i] = index[angindex[i],xposindex[i],yposindex[i]]
    
    dtrain = xgb.DMatrix(Xr, label=clas)
    dtest = xgb.DMatrix(Xt)

    params = {'objective': "multi:softprob",
    'eval_metric': "mlogloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.01,
    'min_child_weight': 2, 
    'n_estimators': 1000,
    # 'subsample': 0.5,
    'max_depth': 5, 
    'gamma': 0.5,
    'num_class':index.max()+1}

    num_round = 40
    bst = xgb.train(params, dtrain, num_round)
    
    ymat = bst.predict(dtest)

    pclas = np.argmax(ymat, 1)
    x, y, z = np.mgrid[0:nbins_ang,0:nbins_xy,0:nbins_xy]
    clas_to_index = np.vstack((x.flatten(), y.flatten(), z.flatten())).transpose()
    Yp = clas_to_index[pclas]
    # returning real position
    real = np.zeros(Yp.shape)
    angx = angbins[0:-1] + (angbins[1]-angbins[0])/2.    
    xy = posbins[0:-1] + (posbins[1]-posbins[0])/2.    
    real[:,0] = angx[Yp[:,0]]
    real[:,1] = xy[Yp[:,1]]
    real[:,2] = xy[Yp[:,2]]
    return real
    # return x[np.argmax(ymat,1)]

def fit_cv(X, Y, algorithm, n_cv=10, verbose=1):
    if np.ndim(X)==1:
        X = np.transpose(np.atleast_2d(X))
    cv_kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
    skf  = cv_kf.split(X)    
    Y_hat=np.zeros((len(Y),Y.shape[1]))
    
    for idx_r, idx_t in skf:        
        Xr = X[idx_r, :]
        Yr = Y[idx_r]
        Xt = X[idx_t, :]
        Yt = Y[idx_t]           
        Yt_hat = eval(algorithm)(Xr, Yr, Xt)         
        Y_hat[idx_t] = Yt_hat
        
    return Y_hat


def test_decodage(features, targets, learners):
    '''
        Main function of the script
        Return : dictionnary with for each models the score PR2 and Yt_hat
    '''
    X = data[features].values
    Y = data[targets].values    # ang
    Models = {method:{'PR2':[],'Yt_hat':[]} for method in learners}
    learners_ = list(learners)
    print learners_

        
    for method in learners_:    
        print('Running '+method+'...')                              
        Models[method] = fit_cv(X, Y, method, n_cv = 3)


    return Models


final_data = {}
for ses in os.listdir("../data/sessions_nosmoothing_200ms/wake/"):
#####################################################################
# DATA LOADING
#####################################################################
    wake_data = scipy.io.loadmat(os.path.expanduser('../data/sessions_nosmoothing_200ms/wake/'+ses))
    adn = wake_data['ADn'].shape[1]
    pos = wake_data['Pos'].shape[1]
        
    if adn >= 7 and pos >= 7:   
#####################################################################
# DATA ENGINEERING
#####################################################################
        data            =   pd.DataFrame()
        data['time']    =   np.arange(len(wake_data['Ang']))      # TODO : import real time from matlab script
        data['ang']     =   wake_data['Ang'].flatten()            # angular direction of the animal head
        data['x']       =   wake_data['X'].flatten()              # x position of the animal 
        data['y']       =   wake_data['Y'].flatten()              # y position of the animal
        data['vel']     =   wake_data['speed'].flatten()          # velocity of the animal 
        # Firing data
        for i in xrange(wake_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = wake_data['Pos'][:,i]
        for i in xrange(wake_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = wake_data['ADn'][:,i]
        # Let's normalize x and y
        for i in ['x', 'y']:
            data[i] = data[i]-np.min(data[i])
            data[i] = data[i]/np.max(data[i])
########################################################################
# COMBINATIONS DEFINITIONS
########################################################################
        combination = {
            'Pos':  {
                    'targets'  :   ['ang', 'x', 'y'],
                    'features'   :   [i for i in list(data) if i.split(".")[0] == 'Pos'], 
                    },          
            'ADn':  {
                    'targets'  :   ['ang', 'x', 'y'],
                    'features'   :   [i for i in list(data) if i.split(".")[0] == 'ADn'],
                    }
        }

# ########################################################################
# # MAIN LOOP FOR SCORE
# ########################################################################

        methods = ['xgb_decodage']
        results = {}
        score = {}
        for k in np.sort(combination.keys()):
            features = combination[k]['features']
            targets = combination[k]['targets']     

            y_hat = test_decodage(features, targets, methods)            

            # y_hat = pickle.load(open("test_xyang.pickle", 'rb'))
            sys.exit()
            from pylab import *
            figure()
            plot(data['ang'].values, label = 'real')
            plot(y_hat[:,0], label = 'pred')
            legend()

            figure()
            plot(data['x'].values, data['y'].values, label = 'real')
            plot(y_hat[:,1], y_hat[:,2], label = 'pred')
            legend()

            show()

            sys.exit()
            y_hat = test_decodage(features, targets, methods)            
            results[k] = y_hat
            score[k] = {}
            y = data['ang'].values
            for m in methods:
                tmp = np.abs(y_hat[m]-y)
                tmp[tmp>np.pi] = 2*np.pi - tmp[tmp>np.pi]
                score[k][m] = np.sum(tmp)

            
        final_data[ses] = {}
        final_data[ses]['wake'] = {'score':score, 'output':results}


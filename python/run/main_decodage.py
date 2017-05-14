#!/usr/bin/env python

'''
    File name: main_fig1.py
    Author: Guillaume Viejo
    Date created: 12/05/2017    
    Python Version: 2.7

Test of decodage

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
adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree_200ms.mat'))
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


methods = ['xgb_decodage', 'lin_decodage']

colors=['#F5A21E', '#02A68E', '#EF3E34', '#134B64', '#FF07CD','b']

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def tuning_curve(x, f, nb_bins):    
    bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
    index = np.digitize(x, bins).flatten()    
    tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)])    
    occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
    tcurve = (tcurve/occupancy)*5.0
    x = bins[0:-1] + (bins[1]-bins[0])/2.    
    return (x, tcurve)

def bayesian_decoding(X, Y, nb_bins = 60):
    # X firing rate
    # Y angular    
    tau = 0.005
    pattern = np.zeros((nb_bins,X.shape[1]))
    for k in xrange(X.shape[1]):
        theta, tuning = tuning_curve(Y, X[:,k], nb_bins)
        pattern[:,k] = tuning
    Yhat = np.zeros((X.shape[0], nb_bins))
    
    tmp = np.exp(-tau*pattern.sum(1))    
    for i in xrange(Yhat.shape[0]):
        Yhat[i] = tmp * np.prod(pattern**(np.tile(X[i], (nb_bins, 1))), 1)

    index = np.argmax(Yhat, 1)
    Yt = theta[index]
    return Yt


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
            'targets'  :   ['ang'],
            'features'   :   [i for i in list(data) if i.split(".")[0] == 'Pos'], 
            },          
    'ADn':  {
            'targets'  :   ['ang'],
            'features'   :   [i for i in list(data) if i.split(".")[0] == 'ADn'],
            }
}

# ########################################################################
# # MAIN LOOP
# ########################################################################

for k in np.sort(combination.keys()):
    features = combination[k]['features']
    targets = combination[k]['targets']     
    results = test_features(features, targets, methods)
    final_data[k] = results
    final_data[k]['bay_decodage'] = {'Yt_hat':bayesian_decoding(data[combination[k]['features']].values, data[targets].values)}

# bayerror = []
# bax = np.arange(30, 2000, 30)
# for b in bax:
#     yt = bayesian_decoding(data[combination['ADn']['features']].values, data[targets].values, b)
#     e = np.sum(np.power(yt.flatten() - data['ang'].values, 2))
#     print e
#     bayerror.append(e)
# bayerror = np.array(bayerror)

error = {}
meth = ['xgb', 'lin', 'bay']
for k in final_data.iterkeys():
    error[k] = []
    for m in meth:
        e = np.sum(np.power(final_data[k][m+'_decodage']['Yt_hat'].flatten() - data['ang'].values, 2))
        error[k].append(e)


from pylab import *
figure()
bar([0,1,2], error['ADn'], 0.4, color = 'red', label = 'adn')
bar([0.45,1.45,2.45], error['Pos'], 0.4, color = 'blue', label = 'pos')
xticks([0.5, 1.5, 2.5], ['xgboost', 'linear regression', 'bayesian'])
ylabel("Mean square error")
legend()


# figure()
# plot(bax, bayerror, 'o-')
# xticks(bax)

show()


sys.exit()
figure()
plot(final_data['ADn']['xgb_decodage']['Yt_hat'].flatten(), label = 'ADn')
plot(final_data['ADn']['lin_decodage']['Yt_hat'].flatten(), label = 'Pos')
plot(data['ang'].values, label = 'real')
legend()
title('xgb')

figure()
plot(final_data['ADn']['lin_decodage']['Yt_hat'].flatten(), label = 'ADn')
plot(final_data['Pos']['lin_decodage']['Yt_hat'].flatten(), label = 'Pos')
plot(data['ang'].values, label = 'real')
title('linear')
legend()

show()


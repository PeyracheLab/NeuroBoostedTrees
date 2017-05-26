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
    'num_class':60}

    # params = {'objective': "reg:linear",
    # # 'eval_metric': "mlogloss", #loglikelihood loss
    # 'seed': 2925, #for reproducibility
    # 'silent': 1,
    # 'learning_rate': 0.1,
    # 'min_child_weight': 2, 
    # 'n_estimators': 1000,
    # # 'subsample': 0.5,
    # 'max_depth': 5, 
    # 'gamma': 0.5}
    

    # binning Yr in 60 classes
    bins = np.linspace(0, 2*np.pi+1e-8, 61)
    clas = np.digitize(Yr, bins).flatten()-1
    x = bins[0:-1] + (bins[1]-bins[0])/2.    
    dtrain = xgb.DMatrix(Xr, label=clas)
    dtest = xgb.DMatrix(Xt)

    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    
    ymat = bst.predict(dtest)

    return x[np.argmax(ymat,1)]

def lin_decodage(Xr, Yr, Xt):
    lr = LinearRegression()
    lr.fit(Xr, Yr)
    Yt = lr.predict(Xt)    
    return Yt     

def tuning_curve(x, f, nb_bins):    
    bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
    index = np.digitize(x, bins).flatten()    
    tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)])    
    occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
    tcurve = (tcurve/occupancy)*5.0
    x = bins[0:-1] + (bins[1]-bins[0])/2.    
    return (x, tcurve)

def bayesian_decoding(Xr, Yr, Xt, nb_bins = 60):
    # X firing rate
    # Y angular    
    tau = 0.200
    pattern = np.zeros((nb_bins,Xr.shape[1]))
    for k in xrange(Xr.shape[1]):
        theta, tuning = tuning_curve(Yr.flatten(), Xr[:,k], nb_bins)
        pattern[:,k] = tuning

    Yhat = np.zeros((Xt.shape[0], nb_bins))        
    tmp = np.exp(-tau*pattern.sum(1))    
    for i in xrange(Yhat.shape[0]):
        Yhat[i] = tmp * np.prod(pattern**(np.tile(Xt[i], (nb_bins, 1))), 1)

    index = np.argmax(Yhat, 1)
    Yt = theta[index]
    return Yt

def fit_cv(X, Y, algorithm, n_cv=10, verbose=1):
    if np.ndim(X)==1:
        X = np.transpose(np.atleast_2d(X))
    cv_kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
    skf  = cv_kf.split(X)    
    Y_hat=np.zeros(len(Y))    
    
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
        Models[method] = fit_cv(X, Y, method, n_cv = 8)


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
# # MAIN LOOP FOR SCORE
# ########################################################################

        methods = ['xgb_decodage', 'bayesian_decoding']
        results = {}
        score = {}
        for k in np.sort(combination.keys()):
            features = combination[k]['features']
            targets = combination[k]['targets']     
            y_hat = test_decodage(features, targets, methods)            
            results[k] = y_hat
            score[k] = {}
            y = data['ang'].values
            for m in methods:
                tmp = np.abs(y_hat[m]-y)
                tmp[tmp>np.pi] = 2*np.pi - tmp[tmp>np.pi]
                score[k][m] = np.mean(tmp)
                

            
        final_data[ses] = {}
        final_data[ses]['real'] = y
        final_data[ses]['wake'] = {'score':score, 'output':results}

# # ########################################################################
# # # TESTING WITH REM 30 min
# # ########################################################################
        rem_data = scipy.io.loadmat(os.path.expanduser('../data/sessions_nosmoothing_100ms/rem/'+ses))
        remdata            =   pd.DataFrame()        
        # Firing data
        for i in xrange(rem_data['Pos'].shape[1]): remdata['Pos'+'.'+str(i)] = rem_data['Pos'][:18000,i]
        for i in xrange(rem_data['ADn'].shape[1]): remdata['ADn'+'.'+str(i)] = rem_data['ADn'][:18000,i]
        for g in ['ADn', 'Pos']:
            features = combination[g]['features']
            target = combination[g]['targets']
            Xr = data[features].values
            Yr = data[target].values
            Xt = remdata[features].values
            Yt = xgb_decodage(Xr, Yr, Xt)

        final_data[ses]['rem'] = {'output':Yt}

# ########################################################################
# # TESTING WITH SWS 30 min
# ########################################################################
        sws_data = scipy.io.loadmat(os.path.expanduser('../data/sessions_nosmoothing_100ms/sws/'+ses))
        swsdata            =   pd.DataFrame()        
        # Firing data
        for i in xrange(sws_data['Pos'].shape[1]): swsdata['Pos'+'.'+str(i)] = sws_data['Pos'][:18000,i]
        for i in xrange(sws_data['ADn'].shape[1]): swsdata['ADn'+'.'+str(i)] = sws_data['ADn'][:18000,i]
        for g in ['ADn', 'Pos']:
            features = combination[g]['features']
            target = combination[g]['targets']
            Xr = data[features].values
            Yr = data[target].values
            Xt = swsdata[features].values
            Yt = xgb_decodage(Xr, Yr, Xt)

        final_data[ses]['sws'] = {'output':Yt}        

# final_data = pickle.load(open("../data/fig4_200t.pickle", 'rb'))

wakescore = {}
for g in ['Pos', 'ADn']:
    wakescore[g] = []
    for s in final_data.iterkeys(): # SESSION
        print s        
        tmp = []
        for m in ['bayesian_decoding', 'xgb_decodage']:
            tmp.append(final_data[s]['wake']['score'][g][m])
        tmp = np.array(tmp)        
        wakescore[g].append(tmp)
    wakescore[g] = np.array(wakescore[g])

# sys.exit()

########################################################################
# PLOTTING
########################################################################

def figsize(scale):
    fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean*0.3              # height in inches
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
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
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


labels = {  'lin_decodage':'Linear', 
            'xgb_decodage':"$\mathbf{XGB}$",
            'bayesian_decoding':"Bayesian"}

colors_ = {'ADn':'#EE6C4D', 'Pos':'#3D5A80'}




figure(figsize = figsize(1))
subplots_adjust()
# subplots_adjust(hspace = 0.2, right = 0.999)
outer = gridspec.GridSpec(1,2, wspace = 0.25,  width_ratios = [0.5,1.3])
## PLOT 1 ##############################################################################################################
gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[0])        
subplot(gs[0])
simpleaxis(gca())

methods_to_plot = ['bayesian_decoding', 'xgb_decodage']
labels_plot = [labels[m] for m in methods_to_plot]


mean_mse = list()
sem_mse = list()
for i in xrange(len(methods_to_plot)):
    PR2_art = wakescore['ADn'][:,i]
    mean_mse.append(np.mean(PR2_art))
    sem_mse.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))
bar(np.arange(np.size(mean_mse)), mean_mse, 0.4, align='center',
        ecolor='k', alpha=.9, color=colors_['ADn'], ec='w', yerr=np.array(sem_mse), label = 'Antero-dorsal nucleus')
plot(np.arange(np.size(mean_mse)), mean_mse, 'k.', markersize=5)
mean_mse = list()
sem_mse = list()
for i in xrange(len(methods_to_plot)):
    PR2_art = wakescore['Pos'][:,i]
    mean_mse.append(np.mean(PR2_art))
    sem_mse.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))
bar(np.arange(np.size(mean_mse))+0.405, mean_mse, 0.4, align='center',
        ecolor='k', alpha=.9, color=colors_['Pos'], ec='w', yerr=np.array(sem_mse), label = 'Post-subiculum')
plot(np.arange(np.size(mean_mse))+0.41, mean_mse, 'k.', markersize=5)
plot([-1, len(methods_to_plot)], [0,0],'--k', alpha=0.4)
legend(bbox_to_anchor=(0.55, 1.3), loc='upper center', ncol=1, frameon = False)
xlim(-0.5,)
# ylim(0.0, 0.8)
xticks(np.arange(np.size(mean_mse))+0.205, labels_plot)
ylabel("Mean error (rad)")


## PLOT 2 ##############################################################################################################
gs = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outer[1])        

ses_ex = 'boosted_tree.Mouse28-140313.mat'

title_ = ['XGB(AD)', 'XGB(Post-S)']



for i,s,g  in zip(xrange(2), ses_ex, ['ADn', 'Pos']):
    subplot(gs[i])
    simpleaxis(gca())
    timestep = np.arange(150) + 9060 # 10 seconds
    plot((timestep-np.min(timestep))*0.2, final_data[ses_ex]['real'][timestep], color = 'black', linewidth = 2)            
    plot((timestep-np.min(timestep))*0.2, final_data[ses_ex]['wake']['output'][g]['xgb_decodage'][timestep], '-', color = colors_[g], linewidth = 1)
    if i == 0:
        ylabel("Angle prediction (rad)")
    xlabel("Time (s)")
    ylim(0, 2*np.pi)
    yticks([0, np.pi, 2*np.pi], ['0', "$\pi$", "$2\pi$"])
    title(title_[i])
    locator_params(axis='x', nbins = 4)


savefig("../../figures/fig4.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig4.pdf &")
    




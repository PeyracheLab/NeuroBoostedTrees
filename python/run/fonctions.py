import numpy as np
from pylab import *
from sklearn.model_selection import KFold
from pyglmnet import GLM
import xgboost as xgb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.regularizers import l1l2
from keras.optimizers import Nadam
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import multiprocessing

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

def poisson_pseudoR2(y, yhat, ynull):
    # This is our scoring function. Implements pseudo-R2
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

def plot_model_comparison(models, labels, color, models_for_plot = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens'], title=None, fs=12):
    """Just makes a comparision bar plot."""
    rcParams.update({   'backend':'pdf',
                        'savefig.pad_inches':0.1})
    figure(figsize = (7,5))
    subplots_adjust(hspace = 0.1, wspace = 0.1)
    plot([-1, len(models_for_plot)], [0,0],'--k', alpha=0.4)

    mean_pR2 = list()
    sem_pR2 = list()
    
    for model in models_for_plot:        
        PR2_art = models[model]['PR2']
        mean_pR2.append(np.mean(PR2_art))
        sem_pR2.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))

    bar(np.arange(np.size(mean_pR2)), mean_pR2, 0.8, align='center',
            ecolor='k', alpha=.9, color=color, ec='w', yerr=np.array(sem_pR2),
            tick_label=labels)
    plot(np.arange(np.size(mean_pR2)), mean_pR2, 'k.', markersize=15)

    ylabel('Pseudo-R2',fontsize=fs)

    simpleaxis(gca())
    if title:
        title(title)


def glm_pyglmnet(Xr, Yr, Xt):
    glm = GLM(	distr='softplus', 
    			alpha=0.1, 
    			tol=1e-8, 
    			verbose=0,
              	reg_lambda=np.logspace(np.log(0.05), np.log(0.0001), 4, base=np.exp(1)),
              	learning_rate=0.1, 
              	max_iter=60000, 
              	eta=2.0, 
              	random_state=1)

    
    glm.fit(Xr, Yr)
    Yt = glm[-1].predict(Xt)
    
    return Yt

def xgb_run(Xr, Yr, Xt):
    params = {'objective': "count:poisson", #for poisson output
    'eval_metric': "logloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.05,
    'min_child_weight': 2, 'n_estimators': 580,
    'subsample': 0.6, 'max_depth': 5, 'gamma': 0.4}        
     

    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)

    num_round = 200
    bst = xgb.train(params, dtrain, num_round)

    Yt = bst.predict(dtest)
    return Yt

def nn(Xr, Yr, Xt):

    if np.ndim(Xr)==1:
        Xr = np.transpose(np.atleast_2d(Xr))

    model = Sequential()

    model.add(  Dense(1980, 
                input_dim=np.shape(Xr)[1], 
                init='glorot_normal',
                activation='relu', 
                W_regularizer=l1l2(0.0, 0.0))
            )
    model.add(  Dropout(0.5)
            )
    model.add(  Dense(18, 
                init='glorot_normal',
                activation='relu',
                W_regularizer=l1l2(0.0, 0.0))
            )
    model.add(  Dense(1,
                activation='softplus')
                )

    optim = Nadam()

    model.compile(loss='poisson', optimizer=optim,)

    hist = model.fit(Xr, Yr, batch_size = 32, nb_epoch=5, verbose=1, validation_split=0.0)
    Yt = model.predict(Xt)[:,0]
    return Yt

def rf(Xr, Yr, Xt):
    params = {'max_depth': 15,
             'min_samples_leaf': 4,
             'min_samples_split': 5,
             'min_weight_fraction_leaf': 0.0,
             'n_estimators': 471}
    
    clf = RandomForestRegressor(**params)
    clf.fit(Xr, Yr)
    Yt = clf.predict(Xt)
    return Yt    

def knn(Xr, Yr, Xt):
    neigh = KNeighborsRegressor(n_neighbors=5,weights='distance')
    neigh.fit(Xr, Yr) 
    Yt = neigh.predict(Xt)
    #returns list of probabilities for each category
    return Yt

def lin_comb(Xr, Yr, Xt):
    lr = LinearRegression()
    lr.fit(Xr, Yr)
    Yt = lr.predict(Xt)
    
    #rectify outputs
    Yt = np.maximum(Yt,np.zeros(Yt.shape))
    return Yt    

def mb_100(Xr, Yr, Xt, nb_bins = 100):
    '''
        Build a tuning curve Yr = hist(Xr) 
        and predict Yt for each Xt point
        TODO : Xr in several dimensions or other than angular 
        TODO : find the function in numpy to do it        
    '''    
    bins = np.linspace(np.vstack((Xr, Xt)).min(), np.vstack((Xr, Xt)).max()+1e-8, nb_bins+1)
    index = np.digitize(Xr, bins).flatten()    
    tcurve = np.array([np.mean(Yr[index == i]) for i in xrange(1, nb_bins+1)])  
    new_index = np.digitize(Xt, bins).flatten()    
    return tcurve[new_index-1]

# TOO LAZY TO RECHANGE LET'S NAME MB WITH DIFFERENT NAMES

def mb_1000(Xr, Yr, Xt, nb_bins = 1000):
    '''
        Build a tuning curve Yr = hist(Xr) 
        and predict Yt for each Xt point
        TODO : Xr in several dimensions or other than angular 
        TODO : find the function in numpy to do it        
    '''    
    bins = np.linspace(np.vstack((Xr, Xt)).min(), np.vstack((Xr, Xt)).max()+1e-8, nb_bins+1)
    index = np.digitize(Xr, bins).flatten()    
    tcurve = np.array([np.mean(Yr[index == i]) for i in xrange(1, nb_bins+1)])  
    new_index = np.digitize(Xt, bins).flatten()    
    return tcurve[new_index-1]

def mb_10(Xr, Yr, Xt, nb_bins = 10):
    '''
        Build a tuning curve Yr = hist(Xr) 
        and predict Yt for each Xt point
        TODO : Xr in several dimensions or other than angular 
        TODO : find the function in numpy to do it        
    '''    
    bins = np.linspace(np.vstack((Xr, Xt)).min(), np.vstack((Xr, Xt)).max()+1e-8, nb_bins+1)
    index = np.digitize(Xr, bins).flatten()    
    tcurve = np.array([np.mean(Yr[index == i]) for i in xrange(1, nb_bins+1)])  
    new_index = np.digitize(Xt, bins).flatten()    
    return tcurve[new_index-1]        
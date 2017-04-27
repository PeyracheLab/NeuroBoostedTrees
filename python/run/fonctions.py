import imp
import numpy as np
# from pylab import *
from sklearn.model_selection import KFold
import xgboost as xgb
try:
    imp.find_module('pyglmnet')
    from pyglmnet import GLM
except ImportError:
    pass
try:
    imp.find_module('keras')
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Lambda
    from keras.regularizers import l1l2
    from keras.optimizers import Nadam
except ImportError:
    pass

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

def plot_model_comparison(models, labels, color, models_for_plot = ['glm_pyglmnet', 'nn', 'xgb_run', 'ens'], title=None, fs=12):
    figure(figsize = figsize(1.0))
    subplots_adjust(hspace = 0.1, wspace = 0.1)

    simpleaxis(gca())

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
    # params = {'objective': "count:poisson", #for poisson output
    # 'eval_metric': "logloss", #loglikelihood loss
    # 'seed': 2925, #for reproducibility
    # 'silent': 1,
    # 'learning_rate': 0.05,
    # 'min_child_weight': 2, 'n_estimators': 580,
    # 'subsample': 0.6, 'max_depth': 5, 'gamma': 0.4}        
    params = {'objective': "count:poisson", #for poisson output
    'eval_metric': "logloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.05,
    'min_child_weight': 2, 'n_estimators': 580,
    'subsample': 0.6, 'max_depth': 400, 'gamma': 0.4}
    
    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)

    num_round = 400
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

    hist = model.fit(Xr, Yr, batch_size = 32, nb_epoch=5, verbose=0, validation_split=0.0)
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

def mb_60(Xr, Yr, Xt):
    '''
        Build a tuning curve Yr = hist(Xr) 
        and predict Yt for each Xt point
        TODO : Xr in several dimensions or other than angular 
        TODO : find the function in numpy to do it        
    '''    
    nb_bins = 60
    bins = np.linspace(np.vstack((Xr, Xt)).min(), np.vstack((Xr, Xt)).max()+1e-8, nb_bins+1)
    index = np.digitize(Xr, bins).flatten()    
    tcurve = np.array([np.mean(Yr[index == i]) for i in xrange(1, nb_bins+1)])  
    new_index = np.digitize(Xt, bins).flatten()    
    return tcurve[new_index-1]      

def mb_360(Xr, Yr, Xt):
    '''
        Build a tuning curve Yr = hist(Xr) 
        and predict Yt for each Xt point
        TODO : Xr in several dimensions or other than angular 
        TODO : find the function in numpy to do it        
    '''    
    nb_bins = 360
    bins = np.linspace(np.vstack((Xr, Xt)).min(), np.vstack((Xr, Xt)).max()+1e-8, nb_bins+1)
    index = np.digitize(Xr, bins).flatten()    
    tcurve = np.array([np.mean(Yr[index == i]) for i in xrange(1, nb_bins+1)])  
    new_index = np.digitize(Xt, bins).flatten()    
    return tcurve[new_index-1]          

def Hbeta(D = np.array([]), beta = 1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta);
    sumP = sum(P);
    H = np.log(sumP) + beta * np.sum(D * P) / sumP;
    P = P / sumP;
    return H, P;


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print "Computing pairwise distances..."
    (n, d) = X.shape;
    sum_X = np.sum(np.square(X), 1);
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
    P = np.zeros((n, n));
    beta = np.ones((n, 1));
    logU = np.log(perplexity);

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print "Computing P-values for point ", i, " of ", n, "..."

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf;
        betamax =  np.inf;
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
        (H, thisP) = Hbeta(Di, beta[i]);

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy();
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2;
                else:
                    beta[i] = (beta[i] + betamax) / 2;
            else:
                betamax = beta[i].copy();
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2;
                else:
                    beta[i] = (beta[i] + betamin) / 2;

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i]);
            Hdiff = H - logU;
            tries = tries + 1;

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

    # Return final P-matrix
    print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta));
    return P;


def pca(X = np.array([]), no_dims = 50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print "Preprocessing the data using PCA..."
    (n, d) = X.shape;
    X = X - np.tile(np.mean(X, 0), (n, 1));
    (l, M) = np.linalg.eig(np.dot(X.T, X));
    Y = np.dot(X, M[:,0:no_dims]);
    return Y;


def tsne(X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if isinstance(no_dims, float):
        print "Error: array X should have type float.";
        return -1;
    if round(no_dims) != no_dims:
        print "Error: number of dimensions should be an integer.";
        return -1;

    # Initialize variables
    X = pca(X, initial_dims).real;
    (n, d) = X.shape;
    max_iter = 1000;
    initial_momentum = 0.5;
    final_momentum = 0.8;
    eta = 500;
    min_gain = 0.01;
    Y = np.random.randn(n, no_dims);
    dY = np.zeros((n, no_dims));
    iY = np.zeros((n, no_dims));
    gains = np.ones((n, no_dims));

    # Compute P-values
    P = x2p(X, 1e-5, perplexity);
    P = P + np.transpose(P);
    P = P / np.sum(P);
    P = P * 4;                                  # early exaggeration
    P = np.maximum(P, 1e-12);

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1);
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
        num[range(n), range(n)] = 0;
        Q = num / np.sum(num);
        Q = np.maximum(Q, 1e-12);

        # Compute gradient
        PQ = P - Q;
        for i in range(n):
            dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
        gains[gains < min_gain] = min_gain;
        iY = momentum * iY - eta * (gains * dY);
        Y = Y + iY;
        Y = Y - np.tile(np.mean(Y, 0), (n, 1));

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q));
            print "Iteration ", (iter + 1), ": error is ", C

        # Stop lying about P-values
        if iter == 100:
            P = P / 4;

    # Return solution
    return Y;

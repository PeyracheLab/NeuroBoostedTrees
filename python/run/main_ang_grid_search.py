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

# max_depth_step = 2**np.arange(1,11)
# max_trees_step = np.array([5,20,40,80,100,150,200,250,300,350,400,500])
# max_depth_step = [3,5,8,10,12,15,20,25,30,40]    
# max_trees_step = [10, 50, 100, 150, 200, 500, 1000, 1500, 2000]
max_depth_step = [3,5,8,10,15,20,30,60,100]    
max_trees_step = [30, 60, 90, 120, 150, 180, 210, 240, 400, 800]

# # automatic fetching | transfer in ../data/results_grid_sessions
os.system("scp -r viejo@guillimin.hpc.mcgill.ca:~/results_grid_sessions/* ../data/results_grid_sessions/")
bic = {}
for file in os.listdir("../data/results_grid_sessions/"):    
    tmp = pickle.load(open("../data/results_grid_sessions/"+file, 'rb'))
    print file, tmp[tmp.keys()[0]].shape
    if tmp[tmp.keys()[0]].shape == (len(max_depth_step),len(max_trees_step)):
        bic[file.split("_")[-2]+"."+tmp.keys()[0]] = tmp[tmp.keys()[0]]


best = []
for k in bic.iterkeys():
    d, t = np.where(bic[k] == bic[k].min())
    best.append([max_depth_step[d[0]], max_trees_step[t[0]]])

best = np.array(best)

bic['best'] = best

with open("../data/grid_search_ang_adn_pos.pickle", 'wb') as f:
    pickle.dump(bic, f)

sys.exit()
# session size
ssize = {}
for f in os.listdir("../data/sessions/wake/"):
    adrien_data = scipy.io.loadmat(os.path.expanduser('../data/sessions/wake/'+f))
    ssize[f.split(".")[1]] = len(adrien_data['Ang'])

llog = {}
for k in bic.keys():
    if k != 'best':
        dt = np.log((np.vstack(max_depth_step)+max_trees_step).astype('float'))
        penalty = np.log(float(ssize[k.split(".")[0]]))*dt
        llog[k] = bic[k] - penalty        

from pylab import *
figure()
for i in xrange(25):    
    subplot(5,5,i+1)
    imshow(bic[bic.keys()[i]], origin = 'lower', interpolation = 'nearest', aspect = 'auto')
    title(bic.keys()[i])
    yticks(np.arange(len(max_depth_step)), max_depth_step)
    xticks(np.arange(len(max_trees_step)), max_trees_step)

show()

# penalty = np.log(22863.0)*dt
# dtnew = (np.vstack(max_depth_step)+max_trees_step).astype('float')
# newpenalty = np.log(22863.0)*dtnew

# allbic = {} 
# best = []
# for g in grid.iterkeys():
#     bic = grid[g]
#     log = bic - penalty
#     bic = log + newpenalty

#     for i in xrange(len(bic)):
#         d, t = np.where(bic[i] == bic[i].min())
#         best.append([max_depth_step[d[0]], max_trees_step[t[0]]])

#     log_ = log.mean(0)
#     bic_ = bic.mean(0)
#     allbic[g] = bic_


from matplotlib import *
from pylab import *


figure()

imshow(allbic['Pos'], origin = 'lower', interpolation = 'nearest', aspect = 'auto')
yticks(np.arange(len(max_depth_step)), max_depth_step)
xticks(np.arange(len(max_trees_step)), max_trees_step)
ylabel("depth")
xlabel("n trees")
show()

sys.exit()



#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""                                                                                   
submit_job.py                                                                         
                                                                                      
to launch sferes job on cluster                                                       
                                                                                      
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.                              
"""

import sys,os
from optparse import OptionParser
import pandas as pd
import scipy.io
import numpy as np
import cPickle as pickle
from time import sleep
# # -----------------------------------                                                 
# # ARGUMENT MANAGER                                                                    
# # -----------------------------------                                                 
# if not sys.argv[1:]:
#    sys.stdout.write("Sorry: you must specify at least 1 argument")
#    sys.stdout.write("More help avalaible with -h or --help option")
#    sys.exit(0)
# parser = OptionParser()
# parser.add_option("-m", "--model", action="store", help="The name of the model to optimize", default=False)
# parser.add_option("-t", "--time", action="store", help="Time of execution", default=False)
# parser.add_option("-d", "--data", action="store", help="The data to fit", default=False)
# parser.add_option("-n", "--n_run", action="store", help="Number of run per subject", default=1)
# (options, args) = parser.parse_args()
# # -----------------------------------                                                 

#####################################################################
# DATA LOADING
#####################################################################
adrien_data = scipy.io.loadmat(os.path.expanduser('~/data_test_boosted_tree.mat'))
# adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree.mat'))

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

########################################################################
# CREATE DIRECTORY RESULTS
########################################################################
os.system("rm -r /home/viejo/results_grid")
os.system("mkdir /home/viejo/results_grid")

########################################################################
# GENERATE BASH SCRIPTS                                                               
########################################################################
all_neurons = []
for k in combination.keys():
	for n in combination[k]['targets']:
		all_neurons.append(n)

all_neurons = np.array(all_neurons)

for n in all_neurons:
# for n in all_neurons[0:1]:	
    filename = "submit_"+n+".sh"
    f = open(filename, "w")
    f.writelines("#!/bin/bash\n")
    f.writelines("#PBS -l nodes=1:ppn=12\n")
    f.writelines("#PBS -l walltime=03:00:00\n")
    f.writelines("#PBS -A exm-690-aa\n")
    f.writelines("#PBS -j oe\n")
    f.writelines("#PBS -N gridsearch_"+n+"\n")    
    f.writelines("\n")                                                  
    f.writelines("python /home/viejo/Prediction_ML_GLM/python/guillimin/cluster_ang_grid_search.py -n "+n+"\n")
    f.close()
    #-----------------------------------

    # ------------------------------------                                                                                            
    # SUBMIT                                                                                                                          
    # ------------------------------------                                                                                            
    os.system("chmod +x "+filename)
    os.system("qsub "+filename+" -q sw")
    os.system("rm "+filename)

    sleep(30)
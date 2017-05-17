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

########################################################################
# CREATE DIRECTORY RESULTS
########################################################################
# os.system("rm -r /home/viejo/results_grid_sessions")
# os.system("mkdir /home/viejo/results_grid_sessions")

#####################################################################
# SESSION LOADING
#####################################################################
all_sessions = os.listdir(os.path.expanduser("~/sessions/wake/"))
for ses in all_sessions:

    #####################################################################
    # DATA LOADING
    #####################################################################
    adrien_data = scipy.io.loadmat(os.path.expanduser('~/sessions/wake/'+ses))    
    
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
    # GENERATE BASH SCRIPTS                                                               
    ########################################################################
    all_neurons = []
    for k in combination.keys():
    	for n in combination[k]['targets']:
    		all_neurons.append(n)

    all_neurons = np.array(all_neurons)
    for n in all_neurons:    
        filename = "submit_"+n+"_"+ses.split(".")[1]+".sh"
        f = open(filename, "w")
        f.writelines("#!/bin/bash\n")
        f.writelines("#PBS -l nodes=1:ppn=12\n")
        f.writelines("#PBS -l walltime=06:00:00\n")
        f.writelines("#PBS -A exm-690-aa\n")
        f.writelines("#PBS -j oe\n")
        f.writelines("#PBS -N gridsearch_"+ses+"_"+n+"\n")    
        f.writelines("\n")                                                  
        f.writelines("python /home/viejo/Prediction_ML_GLM/python/guillimin/cluster_ang_grid_search_all_sessions.py -n "+n+" -s "+ses+"\n")
        f.close()
        #-----------------------------------        
        # ------------------------------------                                                                                            
        # SUBMIT                                                                                                                          
        # ------------------------------------                                                                                            
        os.system("chmod +x "+filename)
        os.system("qsub "+filename+" -q sw")
        os.system("rm "+filename)
        print filename
        sleep(15)
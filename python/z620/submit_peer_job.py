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
names = {}
for g in ['ADn', 'Pos']:
    names[g] = {}
    for w in ['peer', 'cros']:
        names[g][w] = {}
pickle.dump(names, open("fig3_names.pickle", 'wb'))

#####################################################################
# SESSION LOADING
#####################################################################
all_sessions = {}
# for ep in ['wake', 'rem', 'sws']:
for ep in ['rem', 'sws']:
    all_sessions[ep] = os.listdir(os.path.expanduser("~/sessions/"+ep+"/"))
    for ses in all_sessions[ep]:        
        adrien_data = scipy.io.loadmat("/home/guillaume/sessions/"+ep+"/"+ses)
        session = ses.split(".")[1]
        adn = adrien_data['ADn'].shape[1]
        pos = adrien_data['Pos'].shape[1]
########################################################################
# RUNNING PYTHON SCRIPTS ONE BY ONE
########################################################################        
        if adn >= 7 and pos >= 7:               
            os.system("python /home/guillaume/Prediction_ML_GLM/python/guillimin/cluster_peer_pr2_sessions.py -e "+ep+" -s "+ses+"\n")

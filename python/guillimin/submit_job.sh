#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=12:00:00
#PBS -A exm-690-aa
#PBS -j oe
#PBS -N gridsearch_xgboost

python /home/viejo/Prediction_ML_GLM/python/guillimin/cluster_ang_grid_search.py
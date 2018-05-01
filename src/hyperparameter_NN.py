"""
author: Florian Krach

this file is used to do grid search to find optimal hyperparameters for the neural network
"""

# import instruments as inst
import neural_network as nn
import settings_NN as settings
import send_email
import time
from sklearn.model_selection import ParameterGrid
import pandas as pd
from subprocess import call
import os
import data_utils as du
# import numpy as np
# import multiprocessing as mp
# import QuantLib as ql
# import matplotlib.pyplot as plt


# -----------------------------------------------
# send email when starting:
if settings.send:
    send_email.send_email(body='started hyperparameter_NN.py')


if settings.measure_time:
    starttime = time.time()  # to get time


# -----------------------------------------------
# create grid using ParamGrid of sklearn (added by FK)
grid_to_use = settings.param_grid_test
if settings.param_grid_name == 'grid1_g2':
    grid_to_use = settings.param_grid1_g2
elif settings.param_grid_name == 'grid2_g2':
    grid_to_use = settings.param_grid2_g2
elif settings.param_grid_name == 'grid_g2':
    grid_to_use = settings.param_grid_g2
elif settings.param_grid_name == 'grid_hw':
    grid_to_use = settings.param_grid_hw
elif settings.param_grid_name == 'grid_test':
    grid_to_use = settings.param_grid_test


parameters = list(ParameterGrid(grid_to_use))  # FK: this gives a list of dictionaries
param = []
nb_NN_totest = len(parameters)
print
print 'number of NN to be tested:', nb_NN_totest


for i, p in enumerate(parameters):
    param += [[p['exp'], p['layer'], p['dof'], p['dom'], p['dol'], p['lr'], p['activation'], p['alpha'],
               p['residual_cells'], p['batch_size'], p['epochs'], p['train_file'], i]]

# print param
# for exp, layer, dof, dom, dol, lr, activation, alpha, residual_cells, batch_size, epochs, train_file, i in param:
#     print exp, layer, dof, dom, dol, lr, activation, alpha, residual_cells, batch_size, epochs, train_file, i


# -----------------------------------------------
# set the number of parallel jobs:
nb_requested_cpus = settings.hp_requested_cpus
nb_jobs = min(nb_NN_totest, nb_requested_cpus)
print 'number of parallel jobs:', nb_jobs
print

# -----------------------------------------------
# check whethter the directory settings.data_dir_hp exists, otherwise create it
if not os.path.isdir(settings.data_dir_hp):
    print 'make new directory:', settings.data_dir_hp
    print
    call(['mkdir', settings.data_dir_hp])


# -----------------------------------------------
# test all different neural networks generated from hyperparams of the grid:
results_df = nn.test_fnn(nn.hullwhite_fnn, parameters=param, nb_jobs=nb_jobs)  # FK: nb_jobs=-1 means all CPUs are used
results_df = results_df.sort_values(by=['test_loss'])

if settings.hp_print_results:
    print
    print results_df
    print

if settings.save_hp:
    if settings.append_hp:
        try:  # try to load old file and concatenate old results with the new results, afterwards save
            old_df = pd.read_csv(filepath_or_buffer=settings.hp_file, index_col=0, header=0)  # load old results
            results_df = pd.concat([old_df, results_df], ignore_index=True)  # concatenate old and new results
            results_df = results_df.sort_values(by=['test_loss'])  # sort the dataframe s.t. smallest loss on top
            print 'file %s is appended with new data' %settings.hp_file
        except Exception as e:  # if file doesnt yet exist, simple create it with new results
            print e
            print 'New file is generated with the hyperparameter optimization data'
        results_df.to_csv(path_or_buf=settings.hp_file)
    else:  # if we dont append the existing file, save to file with '_new' in the end, so that old file isnt overwritten
        results_df.to_csv(path_or_buf=settings.hp_file[:-4]+'_new.csv')


# -----------------------------------------------
# delete the files in settings.data_dir_hp which were just used as meantime storage:
for file_n in os.listdir(settings.data_dir_hp):  # listdir list of all names of the entries in the given path
    # print file
    call(['rm', settings.data_dir_hp+file_n])


# -----------------------------------------------
if settings.measure_time:
    print 'total time needed:', time.time() - starttime


# send email when done:
if settings.send:
    send_email.send_email(body='successfully terminated hyperparameter_NN.py')




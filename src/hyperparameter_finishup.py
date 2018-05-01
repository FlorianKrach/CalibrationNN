"""
author: Florian Krach

file to finish up hyperparameter testing of the file hyperparameter_NN.py in the case that this was interrupted at some
point while running, so that we can go one at the point where it was interrupted (e.g. if the time on the cluster is too
short)
"""


# import instruments as inst
import data_utils as du
import neural_network as nn
import settings_NN as settings
import send_email
import time
from sklearn.model_selection import ParameterGrid
import pandas as pd
from subprocess import call
import os
import numpy as np


# -----------------------------------------------
# send email when starting:
if settings.send:
    send_email.send_email(body='started hyperparameter_finishup.py')


if settings.measure_time:
    starttime = time.time()  # to get time


# -----------------------------------------------
# hyperparameter grid
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
# print 'number of NN to be tested:', nb_NN_totest
# print


for i, p in enumerate(parameters):
    param += [[p['exp'], p['layer'], p['dof'], p['dom'], p['dol'], p['lr'], p['activation'], p['alpha'],
               p['residual_cells'], p['batch_size'], p['epochs'], p['train_file'], i]]


# -----------------------------------------------
# go through all existing files in settings.data_dir_hp, and find out which parameters out of the grid are missing
# also add all the data together to one dataframe, which can then be appended with the new data
missing_indices = []
df = pd.DataFrame(columns=['test_loss', 'layer', 'exponent', 'learning_rate', 'dropout_first', 'dropout_middle',
                           'dropout_last', 'activation_function', 'alpha', 'residual_cells', 'batch_size', 'epochs'])

for i in range(nb_NN_totest):
    try:
        file_name = '%s.csv' % i
        df_i = pd.read_csv(filepath_or_buffer=settings.data_dir_hp+file_name, index_col=0, header=0)
        df = pd.concat([df, df_i], ignore_index=True)
    except Exception as e:
        missing_indices += [i]

if settings.hp_reversed:
    missing_indices = list(reversed(missing_indices))
    print 'we do the testing with reversed indices'
    # print missing_indices

if settings.hp_shuffle:
    np.random.shuffle(missing_indices)
    print 'we do the testing with shuffled indices:'
    print missing_indices
    print

# print df
# print
# print missing_indices
# print

nb_NN_missing = len(missing_indices)
print
print 'number of missing NN to be tested now / out of original number: %s / %s ' % (nb_NN_missing, nb_NN_totest)
print

missing_param = [param[i] for i in missing_indices]
# print missing_param

if settings.hp_print_results and nb_NN_missing < nb_NN_totest:
    print 'the results that were obtained before:'
    print df.sort_values(by=['test_loss'])
    print


# -----------------------------------------------
# set the number of parallel jobs:
nb_requested_cpus = settings.hp_requested_cpus
nb_jobs = min(nb_NN_missing, nb_requested_cpus)
print 'number of parallel jobs:', nb_jobs
print

# -----------------------------------------------
# check whethter the directory settings.data_dir_hp exists, otherwise create it
if not os.path.isdir(settings.data_dir_hp):
    print 'make new directory:', settings.data_dir_hp
    print
    call(['mkdir', settings.data_dir_hp])

# -----------------------------------------------
# test all different neural networks which haven't been tested yet:
results_df = nn.test_fnn(nn.hullwhite_fnn, parameters=missing_param, nb_jobs=nb_jobs)  # nb_jobs=-1 means all CPUs are used


# -----------------------------------------------
# concatenate new results with old ones (where process was interrupted) and then save them:
results_df = pd.concat([df, results_df], ignore_index=True)
results_df = results_df.sort_values(by=['test_loss'])  # sort the dataframe s.t. smallest loss on top

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
# delete the files which were just used as meantime storage:
for file_n in os.listdir(settings.data_dir_hp):
    # print file
    call(['rm', settings.data_dir_hp+file_n])


# -----------------------------------------------
if settings.measure_time:
    print 'total time needed:', time.time() - starttime


# send email when done:
if settings.send:
    send_email.send_email(body='successfully terminated hyperparameter_finishup.py')



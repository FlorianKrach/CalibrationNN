"""
author: Florian Krach

File to train a neural network with previously generated training data
"""

import instruments as inst
import neural_network as nn
import settings_NN as settings
import send_email
import time
from subprocess import call
import os
import tensorflow as tf

# import numpy as np
# import multiprocessing as mp
# import QuantLib as ql
# import matplotlib.pyplot as plt
# import pandas as pd


# -----------------------------------------------
# send email when starting:
if settings.send:
    send_email.send_email(body='started train_neural_network.py')


if settings.measure_time:
    starttime = time.time()  # to get time


# -----------------------------------------------
# swo definition and filename:
if settings.modelname == 'g2':
    model_dict = inst.g2
else:
    model_dict = inst.hullwhite_analytic
swo = inst.get_swaptiongen(model_dict)

nb_samples_total = settings.nb_samples_total
nb_samples = settings.nb_samples
with_error = settings.with_error
history_part = settings.history_part
history_start = settings.history_start
history_end = settings.history_end
threshold = settings.threshold

train_file = inst.sample_file_name(swo,size=nb_samples_total, with_error=with_error, history_part=history_part,
                                  history_end=history_end, history_start=history_start)


# -----------------------------------------------
# neural net definition and training:
epochs = settings.epochs
batch_size = settings.batch_size
prefix = settings.prefix
dropout_first = settings.dropout_first
dropout_middle = settings.dropout_middle
dropout_last = settings.dropout_last
dropout = settings.dropout
earlyStopPatience = settings.earlyStopPatience
reduceLRPatience = settings.reduceLRPatience
reduceLRFactor = settings.reduceLRFactor
reduceLRMinLR = settings.reduceLRMinLR
layers = settings.layers
lr = settings.lr
exponent = settings.exponent
residual_cells = settings.residual_cells
do_transform = settings.do_transform
loss = settings.loss
postfix = settings.postfix
activation = settings.activation


net = nn.hullwhite_fnn(exponent=exponent, batch_size=batch_size, layers=layers, lr=lr,
                       prefix=prefix, postfix=postfix,
                       dropout=dropout,
                       dropout_first=dropout_first,
                       dropout_middle=dropout_middle,
                       dropout_last=dropout_last,
                       early_stop=earlyStopPatience,
                       lr_patience=reduceLRPatience,
                       reduce_lr=reduceLRFactor,
                       reduce_lr_min=reduceLRMinLR,
                       model_dict=model_dict,
                       residual_cells=residual_cells,
                       train_file=train_file,
                       do_transform=do_transform,
                       activation=activation,
                       valid_size=0.2, test_size=0)


file_name = net.file_name()

# set number of tensorflow threads: (this is needed, since otherwise tensorflow will by default use all available
# threads, which makes the program slower, since the computations are quite small)
config = tf.ConfigProto()
config.intra_op_parallelism_threads = settings.train_NN_requested_cpus
config.inter_op_parallelism_threads = settings.train_NN_requested_cpus
sess = tf.Session(config=config)


# train and save (write) model:
if settings.train:
    net.train(epochs)

# todo: add a fine-tuning on the historical data
if settings.fine_tune:
    tune_data = swo.train_history()

    # load the model: ------------------------------
    # first move the saved model to the right dir from where it can be loaded:
    call(['mv', settings.model_dir + file_name[len(settings.data_dir):] + '.h5', file_name + '.h5'])
    call(['mv', settings.model_dir + file_name[len(settings.data_dir):] + '.p', file_name + '.p'])

    # now load the saved model
    net = nn.read_model(file_name + '.p')

    # move the saved model back:
    call(['mv', file_name + '.h5', settings.model_dir + file_name[len(settings.data_dir):] + '.h5'])
    call(['mv', file_name + '.p', settings.model_dir + file_name[len(settings.data_dir):] + '.p'])

    # do fine tuning
    net.fine_tune(nb_epochs=20, tune_data=tune_data)


if settings.save_NN:
    nn.write_model(net)

    # move the model to the right folder:
    if not os.path.isdir(settings.model_dir):
        print
        print 'make new directory:', settings.model_dir
        print
        call(['mkdir', settings.model_dir])

    call(['mv', file_name+'.h5', settings.model_dir+file_name[len(settings.data_dir):]+'.h5'])
    call(['mv', file_name+'.p', settings.model_dir+file_name[len(settings.data_dir):]+'.p'])


# -----------------------------------------------
if settings.measure_time:
    print 'total time needed:', time.time() - starttime


# send email when done:
if settings.send:
    send_email.send_email(body='successfully terminated train_neural_network.py')



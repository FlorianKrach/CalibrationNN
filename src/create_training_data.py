"""
author: Florian Krach

File to create training data
"""


import instruments as inst
import send_email
import settings_NN as settings
import time
import numpy as np

# import data_utils as du
# import neural_network as nn
# import QuantLib as ql
# import matplotlib.pyplot as plt
# import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------------
# # create the training data (using append and loops, so that we never use to much memory at once):

# send email when starting:
if settings.send:
    send_email.send_email(body='started create_training_data.py')


if settings.measure_time:
    starttime = time.time()  # to get time


multiprocessing = settings.multiprocessing  # use multiple processing (doesnt work on macbook, but it works on cluster (euler))
modelname = settings.modelname


if modelname == 'g2':
    swo = inst.get_swaptiongen(inst.g2)
else:
    swo = inst.get_swaptiongen(inst.hullwhite_analytic)

nb_samples_total = settings.nb_samples_total
nb_samples = settings.nb_samples
with_error = settings.with_error
history_part = settings.history_part
history_start = settings.history_start
history_end = settings.history_end
threshold = settings.threshold
file_name = inst.sample_file_name(swo, size=nb_samples_total, with_error=with_error, history_part=history_part,
                                  history_end=history_end, history_start=history_start)

total_throwouts = 0

# -------------------------------------
# without multiple processing:
if not multiprocessing:
    _, _, _, throwouts = swo.training_data(nb_samples=nb_samples, with_error=with_error, save=True, threshold=threshold,
                                           history_part=history_part, plot=False, ir_pca=True, file_name=file_name)
    total_throwouts += throwouts

    for i in range(nb_samples_total//nb_samples):
        _, _, _, throwouts = swo.training_data(nb_samples=nb_samples, with_error=with_error, save=True, threshold=threshold,
                                               history_part=history_part, plot=False, ir_pca=True, append=True,
                                               file_name=file_name)
        total_throwouts += throwouts

    print
    print 'wanted total number samples:\n', nb_samples + nb_samples*(nb_samples_total//nb_samples)
    print 'total number of throw outs:\n', total_throwouts
    print 'real total number samples:\n', nb_samples + nb_samples*(nb_samples_total//nb_samples) - total_throwouts


# -------------------------------------
# with multiprocessing: '''this works only on cluster (euler cloud computing), macbook always crashes'''
else:
    x_swo, x_ir, y, throwouts = swo.training_data(nb_samples=nb_samples, with_error=with_error, save=False, threshold=threshold,
                                           history_part=history_part, plot=False, ir_pca=True, file_name=file_name)
    total_throwouts += throwouts

    nb_iterations = nb_samples_total // nb_samples
    x_swo, x_ir, y, total_throwouts = inst.call_func_data_gen(nb_iterations, nb_samples, with_error, history_part,
                                                              threshold, modelname, x_swo, x_ir, y, total_throwouts,
                                                              nb_jobs=-1)

    np.save(file_name + '_x_swo', x_swo)
    np.save(file_name + '_x_ir', x_ir)
    np.save(file_name + '_y', y)

    # print x_swo
    # print x_swo.shape
    # print
    # print x_ir
    # print x_ir.shape
    # print
    # print y
    # print y.shape

    print
    print 'wanted total number samples:\n', nb_samples + nb_samples*(nb_samples_total//nb_samples)
    print 'total number of throw outs:\n', total_throwouts
    print 'real total number samples:\n', nb_samples + nb_samples*(nb_samples_total//nb_samples) - total_throwouts


if settings.measure_time:
    print 'total time needed:', time.time() - starttime


# # send email when done:
if settings.send:
    send_email.send_email(body='successfully terminated create_training_data.py')


# ---------------------------------------------------------------------------
# # load saved data:
# import numpy as np
# x_swo = np.load(file_name + '_x_swo.npy')
# x_ir = np.load(file_name + '_x_ir.npy')
# y = np.load(file_name + '_y.npy')
#
# print 'x_swo:\n', x_swo, '\nshape:', x_swo.shape
# print 'x_ir:\n', x_ir, '\nshape:', x_ir.shape
# print 'y:\n', y, '\nshape:', y.shape
# print
# print 'sample 60: x_swo\n', x_swo[60, :].reshape(12,13)
#
# ---------------------------------------------------------------------------
# # volatility surface plot:
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# mgx, mgy = np.meshgrid(range(13), range(12))
# mg = np.meshgrid( swo.axis(1).values, swo.axis(0).values)
# print mg[0], mg[0].shape
# print mg[1], mg[1].shape
# Z = x_swo[90, :].reshape((12,13))
# volas = swo.__getitem__(swo._dates[10])
# print volas, volas.shape
# # px, py = np.meshgrid(range(3), range(4))
# # z = np.random.normal(0,1,(4,3))
# # print px
# # print py
# # print z
# ax.plot_surface(mg[0], mg[1], Z)
# ax.set_xlabel('tenor')
# ax.set_ylabel('maturity')
# ax.set_zlabel('vol')
# plt.show()

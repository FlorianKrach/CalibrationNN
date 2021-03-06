"""
author: Florian Krach
"""

import instruments as inst
import neural_network as nn
import data_utils as du
import settings_NN as settings
import send_email
import time
import numpy as np
from keras.utils import plot_model
from subprocess import call
from joblib import Parallel, delayed


# ----------------------------------------------
# functions:

def error(date, swo, net):  # not uptodate, see error_less_memory
    """
    :param date: date of the wanted day in the panda datetime format
    :param swo: a member of the SwaptionGen class, with the corresponding data
    :param net: a member of the NeuralNetwork class, which has already be trained to compute parameters for the given swo
    :return: errors: NPV and volatility mean absolute error over all swaption instruments of the date
    """
    swo.set_date(date)
    data = du.from_hdf5(swo.key_model)  # calibrated history data
    nb_instruments = len(swo.helpers)  # the numbur of swaptions/instruments
    volas_true = swo.values  # true volatilities of the day
    OrigParams = [data.loc[date]['OrigParam%s' % i] for i in range(len(swo._default_params))]  # params gained with optimization method
    HistParams = [data.loc[date]['HistParam%s' % i] for i in range(len(swo._default_params))]  # params gained with optimization method with start point from previous day
    NetParams = net.predict((swo.values, swo._ircurve.values))
    # print NetParams
    # print OrigParams

    # compute the NPVs and implied volas and errors for each instrument
    NPV_true = np.zeros(nb_instruments)
    NPV_net = np.zeros(nb_instruments)
    NPV_orig = np.zeros(nb_instruments)
    NPV_hist = np.zeros(nb_instruments)
    implVol_net = np.zeros(nb_instruments)
    implVol_orig = np.zeros(nb_instruments)
    implVol_hist = np.zeros(nb_instruments)
    NPV_error_net = np.zeros(nb_instruments)
    NPV_error_orig = np.zeros(nb_instruments)
    NPV_error_hist = np.zeros(nb_instruments)
    Vol_error_net = np.zeros(nb_instruments)
    Vol_error_orig = np.zeros(nb_instruments)
    Vol_error_hist = np.zeros(nb_instruments)
    NPV_total_error_orig = 0
    NPV_total_error_hist = 0
    NPV_total_error_net = 0
    Vol_total_error_orig = 0
    Vol_total_error_hist = 0
    Vol_total_error_net = 0
    with_exception_orig = 0
    with_exception_hist = 0
    with_exception_net = 0

    for i in range(nb_instruments):
        NPV_true[i] = swo.helpers[i].marketValue()

        swo.model.setParams(OrigParams)
        try:
            NPV_orig[i] = swo.helpers[i].modelValue()
            NPV_error_orig[i] = NPV_true[i] - NPV_orig[i]
            NPV_total_error_orig += abs(NPV_error_orig[i])
            implVol_orig[i] = swo.helpers[i].impliedVolatility(NPV_orig[i], 1.0e-6, 1000, 0.0001, 2.50)  # or with: (NPV, 1.0e-4, 1000, 0.001, 1.80)
            Vol_error_orig[i] = volas_true[i] - implVol_orig[i]
            Vol_total_error_orig += abs(Vol_error_orig[i])
        except RuntimeError:
            with_exception_orig += 1

        swo.model.setParams(HistParams)
        try:
            NPV_hist[i] = swo.helpers[i].modelValue()
            NPV_error_hist[i] = NPV_true[i] - NPV_hist[i]
            NPV_total_error_hist += abs(NPV_error_hist[i])
            implVol_hist[i] = swo.helpers[i].impliedVolatility(NPV_hist[i], 1.0e-6, 1000, 0.0001, 2.50)
            Vol_error_hist[i] = volas_true[i] - implVol_hist[i]
            Vol_total_error_hist += abs(Vol_error_hist[i])
        except RuntimeError:
            with_exception_hist += 1

        swo.model.setParams(NetParams[0].tolist())
        try:
            NPV_net[i] = swo.helpers[i].modelValue()
            NPV_error_net[i] = NPV_true[i] - NPV_net[i]
            NPV_total_error_net += abs(NPV_error_net[i])
            implVol_net[i] = swo.helpers[i].impliedVolatility(NPV_net[i], 1.0e-6, 1000, 0.0001, 2.50)
            Vol_error_net[i] = volas_true[i] - implVol_net[i]
            Vol_total_error_net += abs(Vol_error_net[i])
        except RuntimeError:
            with_exception_net += 1

    # compute the average errors for each instrument
    denom_orig = nb_instruments - with_exception_orig
    if denom_orig == 0:
        Vol_average_error_orig = float('inf')
    else:
        Vol_average_error_orig = Vol_total_error_orig / denom_orig
        NPV_average_error_orig = NPV_total_error_orig / denom_orig

    denom_hist = nb_instruments - with_exception_hist
    if denom_hist == 0:
        Vol_average_error_hist = float('inf')
    else:
        Vol_average_error_hist = Vol_total_error_hist / denom_hist
        NPV_average_error_hist = NPV_total_error_hist / denom_hist

    denom_net = nb_instruments - with_exception_net
    if denom_net == 0:
        Vol_average_error_net = float('inf')
    else:
        Vol_average_error_net = Vol_total_error_net / denom_net
        NPV_average_error_net = NPV_total_error_net / denom_net

    # print with_exception_orig, with_exception_hist, with_exception_net
    return NPV_average_error_orig, NPV_average_error_hist, NPV_average_error_net, \
           Vol_average_error_orig, Vol_average_error_hist, Vol_average_error_net


def error_less_memory1(date, swo, net):
    """
    :param date: date of the wanted day in the panda datetime format
    :param swo: a member of the SwaptionGen class, with the corresponding data
    :param net: a member of the NeuralNetwork class, which has already be trained to compute parameters for the given swo
    :return: errors: NPV and volatility mean error over all swaption instruments of the date
    """
    swo.set_date(date)
    data = du.from_hdf5(swo.key_model)  # calibrated history data
    nb_instruments = len(swo.helpers)  # the numbur of swaptions/instruments
    volas_true = swo.values  # true volatilities of the day
    OrigParams = [data.loc[date]['OrigParam%s' % i] for i in range(len(swo._default_params))]  # params gained with optimization method
    HistParams = [data.loc[date]['HistParam%s' % i] for i in range(len(swo._default_params))]  # params gained with optimization method with start point from previous day
    NetParams = net.predict((swo.values, swo._ircurve.values))
    # print NetParams[0].tolist()
    # print OrigParams

    # compute the NPVs and implied volas and errors for each instrument
    NPV_total_error_orig = 0
    NPV_total_error_hist = 0
    NPV_total_error_net = 0
    Vol_total_error_orig = 0
    Vol_total_error_hist = 0
    Vol_total_error_net = 0
    with_exception_orig_npv = 0
    with_exception_hist_npv = 0
    with_exception_net_npv = 0
    with_exception_orig_vol = 0
    with_exception_hist_vol = 0
    with_exception_net_vol = 0

    for i in range(nb_instruments):
        NPV_true = swo.helpers[i].marketValue()

        swo.model.setParams(OrigParams)
        try:
            NPV_orig = swo.helpers[i].modelValue()
            NPV_error_orig = NPV_true - NPV_orig
            NPV_total_error_orig += abs(NPV_error_orig)
            try:
                implVol_orig = swo.helpers[i].impliedVolatility(NPV_orig, 1.0e-6, 1000, 0.0001,
                                                                2.50)  # or with: (NPV, 1.0e-4, 1000, 0.001, 1.80)
                Vol_error_orig = volas_true[i] - implVol_orig
                Vol_total_error_orig += abs(Vol_error_orig)
            except RuntimeError:
                with_exception_orig_vol += 1
        except RuntimeError:
            with_exception_orig_npv += 1
            with_exception_orig_vol += 1


        swo.model.setParams(HistParams)
        try:
            NPV_hist = swo.helpers[i].modelValue()
            NPV_error_hist = NPV_true - NPV_hist
            NPV_total_error_hist += abs(NPV_error_hist)
            try:
                implVol_hist = swo.helpers[i].impliedVolatility(NPV_hist, 1.0e-6, 1000, 0.0001, 2.50)
                Vol_error_hist = volas_true[i] - implVol_hist
                Vol_total_error_hist += abs(Vol_error_hist)
            except RuntimeError:
                with_exception_hist_vol += 1
        except RuntimeError:
            with_exception_hist_npv += 1
            with_exception_hist_vol += 1

        swo.model.setParams(NetParams[0].tolist())
        try:
            NPV_net = swo.helpers[i].modelValue()
            NPV_error_net = NPV_true - NPV_net
            NPV_total_error_net += abs(NPV_error_net)
            try:
                implVol_net = swo.helpers[i].impliedVolatility(NPV_net, 1.0e-6, 1000, 0.0001, 2.50)
                Vol_error_net = volas_true[i] - implVol_net
                Vol_total_error_net += abs(Vol_error_net)
            except RuntimeError:
                with_exception_net_vol += 1
        except RuntimeError:
            with_exception_net_npv += 1
            with_exception_net_vol += 1

    # vol average errors
    denom_orig = nb_instruments - with_exception_orig_vol
    if denom_orig == 0:
        Vol_average_error_orig = float('inf')
    else:
        Vol_average_error_orig = Vol_total_error_orig / denom_orig

    denom_hist = nb_instruments - with_exception_hist_vol
    if denom_hist == 0:
        Vol_average_error_hist = float('inf')
    else:
        Vol_average_error_hist = Vol_total_error_hist / denom_hist

    denom_net = nb_instruments - with_exception_net_vol
    if denom_net == 0:
        Vol_average_error_net = float('inf')
    else:
        Vol_average_error_net = Vol_total_error_net / denom_net

    # npv average errors
    denom_orig = nb_instruments - with_exception_orig_npv
    if denom_orig == 0:
        NPV_average_error_orig = float('inf')
    else:
        NPV_average_error_orig = NPV_total_error_orig / denom_orig

    denom_hist = nb_instruments - with_exception_hist_npv
    if denom_hist == 0:
        NPV_average_error_hist = float('inf')
    else:
        NPV_average_error_hist = NPV_total_error_hist / denom_hist

    denom_net = nb_instruments - with_exception_net_npv
    if denom_net == 0:
        NPV_average_error_net = float('inf')
    else:
        NPV_average_error_net = NPV_total_error_net / denom_net

    return NPV_average_error_orig, NPV_average_error_hist, NPV_average_error_net, Vol_average_error_orig,\
           Vol_average_error_hist, Vol_average_error_net


def error_less_memory(date, swo, net):
    """
    :param date: date of the wanted day in the panda datetime format
    :param swo: a member of the SwaptionGen class, with the corresponding data
    :param net: a member of the NeuralNetwork class, which has already be trained to compute parameters for the given swo
    :return: errors: NPV and volatility mean error over all swaption instruments of the date
    """
    swo.set_date(date)
    data = du.from_hdf5(swo.key_model)  # calibrated history data
    nb_instruments = len(swo.helpers)  # the numbur of swaptions/instruments
    volas_true = swo.values  # true volatilities of the day
    OrigParams = [data.loc[date]['OrigParam%s' % i] for i in range(len(swo._default_params))]  # params gained with optimization method
    HistParams = [data.loc[date]['HistParam%s' % i] for i in range(len(swo._default_params))]  # params gained with optimization method with start point from previous day
    NetParams = net.predict((swo.values, swo._ircurve.values))
    # print NetParams[0].tolist()
    # print OrigParams

    # compute the NPVs and implied volas and errors for each instrument
    NPV_total_error_orig = 0
    NPV_total_error_hist = 0
    NPV_total_error_net = 0
    Vol_total_error_orig = 0
    Vol_total_error_hist = 0
    Vol_total_error_net = 0

    for i in range(nb_instruments):
        NPV_true = swo.helpers[i].marketValue()

        swo.model.setParams(OrigParams)
        try:
            NPV_orig = swo.helpers[i].modelValue()
            NPV_error_orig = NPV_true - NPV_orig
            NPV_total_error_orig += abs(NPV_error_orig**2)
        except RuntimeError:
            NPV_orig = 0
            NPV_error_orig = NPV_true - NPV_orig
            NPV_total_error_orig += abs(NPV_error_orig**2)
        try:
            implVol_orig = swo.helpers[i].impliedVolatility(NPV_orig, 1.0e-6, 1000, 0.0001,
                                                            2.50)  # or with: (NPV, 1.0e-4, 1000, 0.001, 1.80)
            Vol_error_orig = volas_true[i] - implVol_orig
            Vol_total_error_orig += abs(Vol_error_orig)
        except RuntimeError:
            implVol_orig = 0
            Vol_error_orig = volas_true[i] - implVol_orig
            Vol_total_error_orig += abs(Vol_error_orig)


        swo.model.setParams(HistParams)
        try:
            NPV_hist = swo.helpers[i].modelValue()
            NPV_error_hist = NPV_true - NPV_hist
            NPV_total_error_hist += abs(NPV_error_hist**2)
        except RuntimeError:
            NPV_hist = 0
            NPV_error_hist = NPV_true - NPV_hist
            NPV_total_error_hist += abs(NPV_error_hist**2)
        try:
            implVol_hist = swo.helpers[i].impliedVolatility(NPV_hist, 1.0e-6, 1000, 0.0001, 2.50)
            Vol_error_hist = volas_true[i] - implVol_hist
            Vol_total_error_hist += abs(Vol_error_hist)
        except RuntimeError:
            implVol_hist = 0
            Vol_error_hist = volas_true[i] - implVol_hist
            Vol_total_error_hist += abs(Vol_error_hist)

        swo.model.setParams(NetParams[0].tolist())
        try:
            NPV_net = swo.helpers[i].modelValue()
            NPV_error_net = NPV_true - NPV_net
            NPV_total_error_net += abs(NPV_error_net**2)
        except RuntimeError:
            NPV_net = 0
            NPV_error_net = NPV_true - NPV_net
            NPV_total_error_net += abs(NPV_error_net**2)
        try:
            implVol_net = swo.helpers[i].impliedVolatility(NPV_net, 1.0e-6, 1000, 0.0001, 2.50)
            Vol_error_net = volas_true[i] - implVol_net
            Vol_total_error_net += abs(Vol_error_net)
        except RuntimeError:
            implVol_net = 0
            Vol_error_net = volas_true[i] - implVol_net
            Vol_total_error_net += abs(Vol_error_net)

    Vol_average_error_orig = Vol_total_error_orig / nb_instruments
    Vol_average_error_hist = Vol_total_error_hist / nb_instruments
    Vol_average_error_net = Vol_total_error_net / nb_instruments

    NPV_average_error_orig = NPV_total_error_orig / nb_instruments
    NPV_average_error_hist = NPV_total_error_hist / nb_instruments
    NPV_average_error_net = NPV_total_error_net / nb_instruments

    return NPV_average_error_orig, NPV_average_error_hist, NPV_average_error_net, Vol_average_error_orig,\
           Vol_average_error_hist, Vol_average_error_net


def error_helper(date, index):  # todo: edit, since it doesn't work on mac, also not on cluster
    """
    :param date:
    :param index:
    :return:
    """

    if settings.modelname == 'g2':
        model_dict = inst.g2
    else:
        model_dict = inst.hullwhite_analytic
    swo = inst.get_swaptiongen(model_dict)
    print 'swo'

    print 'day:', index
    if index % 100 == 0:
        print 'day:', index
    return np.append(error_less_memory(date, swo, net), index)


# ---------------------------------------------
# send email when starting:
if settings.send:
    send_email.send_email(body='started performance_NN.py')


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

train_file = inst.sample_file_name(swo, size=nb_samples_total, with_error=with_error, history_part=history_part,
                                   history_end=history_end, history_start=history_start)


# -----------------------------------------------
# neural net definition and loading of the trained network:
batch_size = settings.batch_size
epochs = settings.epochs
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


# load the model: ------------------------------
# first move the saved model to the right dir from where it can be loaded:
call(['mv', settings.model_dir+file_name[len(settings.data_dir):]+'.h5', file_name+'.h5'])
call(['mv', settings.model_dir+file_name[len(settings.data_dir):]+'.p', file_name+'.p'])

# now load the saved model
net = nn.read_model(file_name+'.p')

# move the saved model back:
call(['mv', file_name+'.h5', settings.model_dir+file_name[len(settings.data_dir):]+'.h5'])
call(['mv', file_name+'.p', settings.model_dir+file_name[len(settings.data_dir):]+'.p'])


# -----------------------------------------------
# compute errors with error function, save them and plot results:

print net.model.summary()
# print error(swo._dates[600], swo, net)

# plot the model:
if settings.plot_model:
    plot_model(net.model, settings.model_dir+'model_plot.png', show_shapes=True)

dates = settings.dates
nb_dates = len(dates)
NPV_error = np.zeros((nb_dates, 3))
Vol_error = np.zeros((nb_dates, 3))



# without parallalism:
if not settings.multiprocessing_performance:
    for i, date in enumerate(dates):
        if i%100 == 0:
            print 'day:', i
        errors = error_less_memory(date, swo, net)
        NPV_error[i, :] = errors[0:3]
        Vol_error[i, :] = errors[3:6]


# parallelize error computation:
else:
    nb_jobs = settings.performance_requested_cpus

    call(['mv', settings.model_dir + file_name[len(settings.data_dir):] + '.h5', file_name + '.h5'])
    call(['mv', settings.model_dir + file_name[len(settings.data_dir):] + '.p', file_name + '.p'])
    results = Parallel(n_jobs=nb_jobs)(delayed(error_helper)(date, i) for i, date in enumerate(dates))
    call(['mv', file_name + '.h5', settings.model_dir + file_name[len(settings.data_dir):] + '.h5'])
    call(['mv', file_name + '.p', settings.model_dir + file_name[len(settings.data_dir):] + '.p'])

    print results
    errors = sorted(results, key=lambda x: x[6])
    print errors
    NPV_error = errors[:, 0:3]
    Vol_error = errors[:, 3:6]

# print
# print NPV_error
# print
# print Vol_error
# print


# save errors:
np.save(settings.error_filename, np.concatenate((NPV_error, Vol_error), axis=1))


# -----------------------------------------------
# plot:
if settings.save_plot:
    save_file = settings.model_dir+'plot_NPV_mean_squared_error.png'
else:
    save_file = None

du.plot_data(dates, NPV_error, labels=('Orig', 'Hist', 'FNN'), figsize=settings.figsize, frame_lines=False,
             yticks_format=None, save=save_file, min_x_ticks=5, max_x_ticks=8,
             interval_multiples=True, colors=None, title='NPV mean squared error',
             legend_fontsize=None, legend_color=du.almost_black,
             title_fontsize=None, title_color=du.almost_black,
             xlabel=None, ylabel=None, xmin=0, xmax=None,
             xlabel_color=du.almost_black, ylabel_color=du.almost_black,
             xlabel_fontsize=14, ylabel_fontsize=14,
             xtick_fontsize=14, ytick_fontsize=14,
             xtick_color=du.almost_black, ytick_color=du.almost_black,
             out_of_sample=int(nb_dates*history_part), ytick_range=None, xtick_labels=None, show=settings.show_plot)

if settings.save_plot:
    save_file = settings.model_dir+'plot_Vol_absolute_mean_error.png'
else:
    save_file = None

du.plot_data(dates, Vol_error*100, labels=('Orig', 'Hist', 'FNN'), figsize=settings.figsize, frame_lines=False,
             yticks_format=None, save=save_file, min_x_ticks=5, max_x_ticks=8,
             interval_multiples=True, colors=None, title='Vol absolute mean error',
             legend_fontsize=None, legend_color=du.almost_black,
             title_fontsize=None, title_color=du.almost_black,
             xlabel=None, ylabel='%', xmin=0, xmax=None,
             xlabel_color=du.almost_black, ylabel_color=du.almost_black,
             xlabel_fontsize=14, ylabel_fontsize=14,
             xtick_fontsize=14, ytick_fontsize=14,
             xtick_color=du.almost_black, ytick_color=du.almost_black,
             out_of_sample=int(nb_dates*history_part), ytick_range=None, xtick_labels=None, show=settings.show_plot)


# -----------------------------------------------
# measure time
if settings.measure_time:
    print 'total time needed:', time.time() - starttime


# send email when done:
if settings.send:
    send_email.send_email(body='successfully terminated performance_NN.py')






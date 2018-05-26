"""
author: Florian Krach

File to test out all things
"""


import instruments as inst
import data_utils as du
import neural_network as nn
import QuantLib as ql
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import send_email
import time
import settings_NN as settings
from subprocess import call
import os
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def error(swo):
    nb_instruments = len(swo.helpers)  # the numbur of swaptions/instruments
    volas_true = swo.values  # true volatilities of the day

    # compute the NPVs and implied volas and errors for each instrument
    NPV_total_error = 0
    Vol_total_error = 0

    for i in range(nb_instruments):
        NPV_true = swo.helpers[i].marketValue()

        try:
            NPV_model = swo.helpers[i].modelValue()
            NPV_error = NPV_true - NPV_model
            NPV_total_error += abs(NPV_error**2)
        except RuntimeError:
            NPV_model = 0
            NPV_error = NPV_true - NPV_model
            NPV_total_error += abs(NPV_error**2)
        try:
            implVol = swo.helpers[i].impliedVolatility(NPV_model, 1.0e-6, 1000, 0.0001,
                                                            2.50)  # or with: (NPV, 1.0e-4, 1000, 0.001, 1.80)
            Vol_error = volas_true[i] - implVol
            Vol_total_error += abs(Vol_error)
        except RuntimeError:
            implVol = 0
            Vol_error = volas_true[i] - implVol
            Vol_total_error += abs(Vol_error)

    Vol_average_error = Vol_total_error / nb_instruments
    NPV_average_error = NPV_total_error / nb_instruments

    return NPV_average_error, Vol_average_error


# ---------------------------------------------------------------------------
# get_swaptiongen:
# index = ql.GBPLibor(ql.Period(6, ql.Months))
# print index


# ---------------------------------------------------------------------------
# calibrate history
# swo = inst.get_swaptiongen(inst.hullwhite_analytic)
# swo.calibrate_history()
# print swo
'''this works'''

# ---------------------------------------------------------------------------
# train history: this generates the data from the history that are needed to generate the random training data
# swo = inst.get_swaptiongen(inst.hullwhite_analytic)
# # swo.train_history(save=False)
''' this works '''
#
# # load the saved data:
# file_name = inst.sample_file_name(swo,0, True, None, None, 0.4)
# x_swo = np.load(file_name + '_x_swo.npy')
# y = np.load(file_name + '_y.npy')
# print x_swo
# print y


# ---------------------------------------------------------------------------
# training_data: this generates the random training data for the neural network
swo = inst.get_swaptiongen(inst.hullwhite_analytic)

nb_samples = 100
with_error = True
history_part = 0.4
history_start = None
history_end = None
file_name = inst.sample_file_name(swo,size=nb_samples, with_error=with_error, history_part=history_part, history_end=history_end, history_start=history_start)

# swo.training_data(nb_samples=nb_samples, with_error=with_error, seed=0, save=True, history_part=history_part, plot=False, ir_pca=True)

# # load saved data:
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

# --------------------------------------------
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


# ---------------------------------------------------------------------------
# evaluate function: evaluates the model for given parameters, IRCurve values, date and returns error
# swo = inst.get_swaptiongen(inst.hullwhite_analytic)
data = du.from_hdf5(swo.key_model) #calibrated history data
df_error = du.from_hdf5(swo.key_error, du.h5file)
iDate = 100
date = swo._dates[iDate]

# print date
# print swo.model.params()
# params = [data.iloc[iDate]['OrigParam%s' %i] for i in range(len(swo.model.params()))]
# print params
# print swo._ircurve.values  # here the values are still from day 0
# swo.set_date(date)
# print swo._ircurve.values  # now after setting the date, we get the correct values from day iDate
# IRValues = swo._ircurve.__getitem__(date).data()  # gives the same as swo._ircurve.values, if the IRCurve is linked to the right date
# print IRValues
#
# average_error, errors = swo.evaluate(params, IRValues, date)
# # compare whether we get the same errors as stored in the error file:
# print errors
# print -df_error.loc[date]


# ---------------------------------------------------------------------------
# compare_history:
# print swo._default_params
# print np.array(swo._default_params).shape


# ---------------------------------------------------------------------------
# using pipeline:
# funcTrm = inst.FunctionTransformerWithInverse(func=np.exp, inv_func=np.log)
# scaler = inst.MinMaxScaler((1,2))  # for scaler see: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# pipeline = inst.Pipeline([('funcTrm', funcTrm), ('scaler', scaler)])
# # pipeline = inst.Pipeline([('funcTrm', funcTrm)])
# X = np.array(range(10)).reshape((10, 1))
# print X
# Y = pipeline.fit_transform(X)
# print Y
# Z = pipeline.inverse_transform(Y)
# print Z
# print


# ---------------------------------------------------------------------------
# retrieve_swo_train_set:
# file_name = inst.sample_file_name(swo,size=nb_samples, with_error=with_error, history_part=history_part, history_end=history_end, history_start=history_start)
#
# dict = inst.retrieve_swo_train_set(file_name, transform=True, func=None, inv_func=None,
#                            valid_size=0.2, test_size=0.2, total_size=1.0,
#                            scaler=inst.MinMaxScaler(), concatenated=True)
# for k,v in dict.iteritems():
#     print k+':'
#     if hasattr(v, 'shape'):
#         print 'shape:', v.shape
#     print v
#     print


# ---------------------------------------------------------------------------
# local_hw_map:
# pointA = [0.0001, 0.0001]
# pointB = [0.1,0.01]
# mgx, mgy, objective_values = inst.local_hw_map(swo, date, pointA, pointB, off_x=0.0, off_y=0.0,
#                  low_x=1e-8, low_y=1e-8, nb_points=10)
#
# print objective_values
# print np.unravel_index(np.argmin(objective_values), objective_values.shape)
# print

# -----------------------------------------------------------------------------
# # play with data
#
# swo = inst.get_swaptiongen(inst.hullwhite_analytic)  # other models: g2 (=HW-2factor model)
# swo1 = inst.get_swaptiongen(inst.g2)
# # print swo.key_model
# # print
# # print swo.helpers
# print
#
# # get model price and volatility of swaption
# #  all calculated with default params instead of the right params, see below
# NPV = swo.helpers[0].modelValue()
# vola = swo.helpers[0].impliedVolatility(NPV, 1.0e-6, 1000, 0.0001, 2.50)
# date = swo._dates[0]
# #
# # NPV1 = swo1.helpers[0].modelValue()
# # vola1 = swo1.helpers[0].impliedVolatility(NPV1, 1.0e-6, 1000, 0.0001, 2.50)
# # date1 = swo1._dates[0]
# #
# # print date, date1
# # print NPV, NPV1
# # print vola, vola1
# # print
# # print 'length helpers:', len(swo.helpers)
# # print
#
#
# # term structure
# # print 'term structure:\n', swo._term_structure
#
#
# # IRCurve:
# # ir = swo._ircurve.to_matrix()
# # print 'IR Curve'
# # print ir
# # print ir.shape
# # print
#
#
# # quotes:
# # print 'quotes:'
# # print swo._quotes
#
# # --- get the market volatilities at t=0
# volas = swo.__getitem__(swo._dates[0])
# # print volas
# # print volas.shape
# # print
# # print swo._data.loc[swo._dates[0]]  # other way to get the data from the swo model, does in fact exactly the same as swo.__getitem__(...), unless that it is not transformed to matrix
# # print
#
# # test whethter they are same for all models:
# # volas1 = swo1.__getitem__(swo1._dates[0])
# # print volas1
# # print volas1.shape
# # print
# # print swo1._data.loc[swo1._dates[0]]
# # print
# # print volas == volas1
# # print
#
#
# # calibrated history data (includes the optimized parameters, errors etc.) and stuff
# data = du.from_hdf5(swo.key_model) #calibrated history data
# df_error = du.from_hdf5(swo.key_error, du.h5file)
# # print data
# # print
# # print swo.model.params()
# # print data.iloc[0]
# # print data.iloc[0]['OrigParam0']
# # print data.iloc[0].OrigParam0
# # print
#
#
# # see documentation: http://quantlib.sourcearchive.com/documentation/1.1-1/classQuantLib_1_1SwaptionHelper.html
# # if we want to compute prices etc for other dates, use 'set_date' first
# swo.model.setParams([data.iloc[0]['OrigParam0'], data.iloc[0]['OrigParam1']])  # set the model parameters to first day
# NPV_new = swo.helpers[0].modelValue() # price of the swaption according to model
# vola_new = swo.helpers[0].impliedVolatility(NPV_new, 1.0e-6, 1000, 0.0001, 2.50) # black volatility implied by the model, -> the model volatility
# market_val = swo.helpers[0].marketValue() # market price of the swaption, using black pricing formula and the true market volatility
# black_price = swo.helpers[0].blackPrice(volas[0,0]) # black swaption price, for a given volatility, here I used the true market volatility, so as to get the market value
# print 'NPV: default params: ', NPV, ' calibrated params: ', NPV_new
# print 'Vol.: default params: ', vola, ' calibrated params: ', vola_new, ' true market vol: ', volas[0,0]
# print 'market value:', market_val, 'black price:', black_price
# print 'error:', df_error.iloc[0][0], 'implied vol - error =', vola_new-df_error.iloc[0][0], 'true market vol:', volas[0,0] # implied_vol - error gives agein the market volatility
# print
#
# # do same for 2nd day, without setting date before:
# # !!!! this gives wrong prices, since the quote values (volatilities) were not updated in the swaptionHelpers
# # iDate = 1
# # volas = swo.__getitem__(swo._dates[iDate])
# # swo.model.setParams([data.iloc[iDate]['OrigParam0'], data.iloc[iDate]['OrigParam1']])  # set the model parameters to first day
# # NPV_new = swo.helpers[0].modelValue() # price of the swaption according to model
# # vola_new = swo.helpers[0].impliedVolatility(NPV_new, 1.0e-6, 1000, 0.0001, 2.50) # black volatility implied by the model, -> the model volatility
# # market_val = swo.helpers[0].marketValue() # market price of the swaption, using black pricing formula and the true market volatility
# # black_price = swo.helpers[0].blackPrice(volas[0,0]) # black swaption price, for a given volatility, here I used the true market volatility, so as to get the market value
# # print 'NPV: (model price)', NPV_new
# # print 'Vol.: model:', vola_new, ' true market vol: ', volas[0,0]
# # print 'market value:', market_val, 'black price (computed with market vol):', black_price
# # print 'error:', df_error.iloc[iDate][0], 'implied vol - error =', vola_new-df_error.iloc[iDate][0], 'true market vol:', volas[0,0]  # implied_vol - error gives agein the market volatility
# # print
#
# # do same for 2nd day, with setting date
# iDate = 1
# volas = swo.__getitem__(swo._dates[iDate])
# swo.set_date(swo._dates[iDate])
# swo.model.setParams([data.iloc[iDate]['OrigParam0'], data.iloc[iDate]['OrigParam1']])  # set the model parameters to first day
# NPV_new = swo.helpers[0].modelValue() # price of the swaption according to model (of the first swaption_instrument)
# vola_new = swo.helpers[0].impliedVolatility(NPV_new, 1.0e-6, 1000, 0.0001, 2.50)  # black volatility implied by the model, -> the model volatility
# market_val = swo.helpers[0].marketValue()  # market price of the swaption, using black pricing formula and the true market volatility
# black_price = swo.helpers[0].blackPrice(volas[0,0])  # black swaption price, for a given volatility, here I used the true market volatility, so as to get the market value
# print 'NPV: (model price)', NPV_new
# print 'Vol.: model:', vola_new, ' true market vol: ', volas[0,0]
# print 'market value:', market_val, 'black price (computed with market vol):', black_price
# print 'error:', df_error.iloc[iDate][0], 'implied vol - error =', vola_new-df_error.iloc[iDate][0], 'true market vol:', volas[0,0]  # implied_vol - error gives agein the market volatility
# print


# with pd.HDFStore(du.h5file) as store:
# 	#print store.keys()
# 	data = store['/Models/IR/SWO/GBP/Hull_White_analytic_formulae']
# 	store.close()
# print data
print


# ---------------------------------------------------------------------------------------------------------------------
# Neural Network testing:
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# # with dict we can access all class variables and functions, as if the class was a dictionary:
# for k,v in swo.__dict__.items():
#     print k


# ---------------------------------------------------------------------------
# # play with class: NeuralNetwork:
# net = nn.NeuralNetwork(inst.hullwhite_analytic, train_file=file_name)
# for k,v in net._data.items():
#     print k
#     print v
#     print


# ---------------------------------------------------------------------------
# # try out partial:
# def dummy_func(x,y):
#     print x, y
#
# partial_dummy_func = nn.partial(dummy_func, y=4)
#
# print partial_dummy_func(x=7)
# print partial_dummy_func(x=6,y=10)


# ---------------------------------------------------------------------------
# # test NeuralNets:


# ---------------------------------------------------------------------------
# # send email when done:
# send_email.send_email(password='')


# ---------------------------------------------------------------------------
# # test_fnn: grid search to find best hyperparams, using multiprocessing (package: joblib):
''' this works'''

# # create grid using ParamGrid of sklearn (added by FK)
# from sklearn.model_selection import ParameterGrid
# param_grid = {'exp': range(5,7), 'layer': range(3,5), 'dof': [0.2], 'dom': [0.2], 'dol': [0.2],
#               'lr': [0.1, 0.01, 0.001], 'alpha': [1.0], 'epochs': [20], 'train_file': [train_file]}
# parameters = list(ParameterGrid(param_grid))  # FK: this gives a list of dictionaries
# param = []
#
# for i,p in enumerate(parameters):
#     param += [[p['exp'], p['layer'], p['dof'], p['dom'], p['dol'], p['lr'], p['alpha'], p['epochs'], p['train_file']]]
#
# # print param
# #
# # for exp, layer, dof, dom, dol, lr, alpha, epochs, train_file in param:
# #     print exp, layer, dof, dom, dol, lr, alpha, epochs, train_file
#
#
# # # count number of CPUs:
# # import multiprocessing
# # num_core=multiprocessing.cpu_count()
# # print num_core
#
# # test all neural networks generated from hyperparams of the grid:
# start_time = time.time()
# nn.test_fnn(nn.hullwhite_fnn , parameters=param, nb_jobs=-1)  # FK: nb_jobs=-1 means all CPUs are used
# print
# print 'time:', time.time() - start_time


# ---------------------------------------------------------------------------
# # test network training with multiprocessing via Theano:

swo = inst.get_swaptiongen(inst.g2)

nb_samples_total = 250000  # 100
nb_samples = 1000  # 10
with_error = True
history_part = 0.4
history_start = None
history_end = None
threshold = 0

train_file = inst.sample_file_name(swo,size=nb_samples_total, with_error=with_error, history_part=history_part,
                                  history_end=history_end, history_start=history_start)

compare = False
epochs = 4
prefix = ''
postfix = '_s'+str(nb_samples_total)+'_'+str(history_part)
dropout_first = None
dropout_middle = None
dropout_last = None
dropout = 0.2
earlyStopPatience = 125
reduceLRPatience = 40
reduceLRFactor = 0.5
reduceLRMinLR = 0.000009
save = True
layers = 4
lr = 0.001
exponent = 6
load = False
model_dict = inst.g2
residual_cells = 1
train_file = train_file
do_transform = True
loss = 'mean_squared_error'


net = nn.hullwhite_fnn(exponent=exponent, layers=layers, lr=lr,
                                 prefix=prefix, postfix=postfix,
                                 dropout=dropout,
                                 dropout_first=dropout_first,
                                 dropout_middle=dropout_middle,
                                 dropout_last=dropout_last,
                                 earlyStopPatience=earlyStopPatience,
                                 reduceLRPatience=reduceLRPatience,
                                 reduceLRFactor=reduceLRFactor,
                                 reduceLRMinLR=reduceLRMinLR,
                                 model_dict=model_dict,
                                 residual_cells=residual_cells,
                                 train_file=train_file,
                                 do_transform=do_transform,
                                 activation="elu")

# file_name = net.file_name()
#
# starttime = time.time()
# # # train and write model:
# net.train(epochs)
# # nn.write_model(net)
#
# print 'time:', time.time() - starttime


# ------------------------------------------
# testing for own error function in performance_NN.py:
nb_instruments = len(swo.helpers)  # the numbur of swaptions/instruments
volas = swo._data.loc[date].as_matrix().reshape((nb_instruments,))
swo.set_date(date)
# print volas, volas.shape
# print swo.values
# print swo._ircurve.values
# print data
# print len(swo._dates)


# ------------------------------------------
# testing to find out nominal N and type of used swaptions
# print
# print swo.helpers[0].swaptionNominal()
# print swo.helpers[0].marketValue()
# print swo.helpers[0].swaptionStrike()


# ------------------------------------------
# # testing for output of parallel hyperparemeter optimization
# df = pd.DataFrame(columns=['test loss', 'layer', 'exponent', 'Learning Rate', 'dropout_first', 'dropout_middle',
#                            'dropout_last', 'activation function', 'alpha', 'residual_cells', 'batch_size', 'epochs'],
#                   data=np.random.normal(size=(4,12)))
#
# print df

df = pd.read_csv(filepath_or_buffer= settings.hp_file, index_col=0, header=0)
print df

# df = df.sort_values(by=['test_loss'])
# print df

# res = pd.concat([df,df], ignore_index=True)
# print res

# try:
#     df = pd.read_csv(filepath_or_buffer= 'file1.csv', index_col=0, header=0)
# except Exception as e:
#     print(e)

# array = (1,2,3,4,5)
# np_array = np.asarray([array])
# print np_array
# df = pd.DataFrame(columns=['test loss', 'layer', 'exponent', 'Learning Rate', 'dropout_first'],
#                   data=np_array)
#
# print df

# df = pd.read_csv(filepath_or_buffer= du.data_dir_hp+'0.csv',
#                  index_col=0, header=0)
# print
# print df
# print


# ------------------------------------------
# # testing for hyperparemeter_finishup:
# for file in os.listdir(du.data_dir_hp):
#     print file
#     call(['rm', du.data_dir_hp+file])


# ------------------------------------------
# testing for plot_performance:
# print
# print np.load(settings.error_filename)

# start = int(settings.history_part * len(settings.dates))
# end = start + 43
# print swo._dates[start], swo._dates[end]


# ------------------------------------------
# dummy joblib example
#
# def dummy_func(index, start_time):
#     time1 = int(time.time() - start_time)
#     time.sleep(2)
#     time2 = int(time.time() - start_time)
#     return index, time1, time2
#
# start_time = time.time()
# results = Parallel(n_jobs=1)(delayed(dummy_func)(i, start_time) for i in range(10))
#
# print results


# ------------------------------------------
# # try to find out whether models are calibrated to fit volas or npv
# swo = inst.get_swaptiongen(inst.hullwhite_analytic)
# dates = swo._dates
# swo.set_date(dates[0])
# print
# print 'before calibration:', swo.model.params()
# swo.model.calibrate(swo.helpers, swo.method, swo.end_criteria, swo.constraint)
# vol_params = swo.model.params()
# print 'calibrate with VolErrorType:', vol_params
# error1 = error(swo)
#
# swo1 = inst.SwaptionGen(index=ql.GBPLibor(ql.Period(6, ql.Months)), model_dict=inst.hullwhite_analytic,
#                         error_type=ql.CalibrationHelper.PriceError)
# swo1.set_date(dates[0])
# swo1.model.calibrate(swo1.helpers, swo1.method, swo1.end_criteria, swo1.constraint)
# rel_price_params = swo1.model.params()
# print 'calibrated with RelativePriceError:', rel_price_params
# error2 = error(swo1)
# print
# print 'errors:                             NPV mean squared error, absolute volatility error'
# print 'calibrate with VolErrorType:       ', error1
# print 'calibrated with RelativePriceError:', error2
'''result: hernandez uses an optimization method that minimizes to volatility error, not the npv error'''

#
# print dir(swo.helpers[0])
# print swo.helpers[0].swaptionStrike()
# print swo.helpers[0].swaptionNominal()


# ------------------------------------------
# # test for new pipeline for neural net input
funcTrm = inst.FunctionTransformerWithInverse(func=np.log, inv_func=np.exp)
input_transformer = Pipeline([('funcTrm', funcTrm), ('scaler', MinMaxScaler())])

x = range(1,10)
x = np.asarray(x)
x = np.array([x,2*x])
print x
input_transformer.fit(x.transpose())
x = input_transformer.transform(x.transpose())
print x
x = input_transformer.inverse_transform(x)
print x



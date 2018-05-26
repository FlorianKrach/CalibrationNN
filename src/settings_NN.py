"""
author: Florian Krach

this file is used to set all the variables for: create_training_data.py, train_neural_network.py, performance_NN.py
"""
import data_utils as du
import instruments as inst

# TODO: try out neural network, where the input data is first transformed with log and a MinMaxScaler
# DONE: try out much deeper networks, with residuals (compare paper)
# DONE: try out batch size 8 and other alpha

# ----------------------------------------------------------------------------------------------------------------------
# general settings:
send = False  # send emails when starting, terminating
measure_time = True
modelname = 'g2'  # options: 'g2', 'hullwhite_analytic'
which_model = 'model1'  # other possibilities: see below


model_dir = du.data_dir+which_model+'/'
data_dir = du.data_dir


# ----------------------------------------------------------------------------------------------------------------------
# settings for the SwapGen and create_training_data:
with_error = True
history_part = 0.4
history_start = None
history_end = None
threshold = 0
multiprocessing_CTD = True  # use multiple processing for creating training data (doesnt work on macbook, but it
# works on cluster (euler))


# ----------------------------------------------------------------------------------------------------------------------
# settings for train_NN:
train = True
fine_tune = False
save_NN = True
train_NN_requested_cpus = 1

# ----------------------------------------------------------------------------------------------------------------------
# settings for performance_NN:
plot_model = True  # whether to plot an overview of the model structure
show_plot = False  # whether to show the error plots
save_plot = True  # whether to save the error plots
figsize = (15, 10)  # (20, 15)
multiprocessing_performance = False
performance_requested_cpus = 2

error_filename_name = 'performance_errors.npy'
error_filename = model_dir+error_filename_name
swo = inst.get_swaptiongen(inst.g2)
dates = swo._dates[:]
end_plotting = int(history_part*len(dates)) + 132

# ----------------------------------------------------------------------------------------------------------------------
# settings for plot_performance:
# performance_models = ['hw_model1', 'model1_smaller_train_size', 'model1', 'model2']
performance_models = ['hw_model1','model1', 'model2']
performance_index = [0,4,5,8]


# ----------------------------------------------------------------------------------------------------------------------
# settings for NN general
prefix = ''
do_transform = True
loss = 'mean_squared_error'
batch_size = 16

if modelname == 'g2':
    if which_model == 'model_test':
        epochs = 1
        dropout_first = None
        dropout_middle = None
        dropout_last = None
        dropout = 0.2
        earlyStopPatience = 125
        reduceLRPatience = 40
        reduceLRFactor = 0.1
        reduceLRMinLR = 0.00000001
        layers = 4
        residual_cells = 0
        exponent = 4
        lr = 0.000001
        activation = 'elu'

        nb_samples_total = 250000
        nb_samples = 1000

        model_dict = inst.g2

    elif which_model == 'model1' or 'model1_smaller_train_size' or 'model1_fine_tuned':
        epochs = 500
        dropout_first = None
        dropout_middle = None
        dropout_last = None
        dropout = 0.2
        earlyStopPatience = 125
        reduceLRPatience = 40
        reduceLRFactor = 0.1
        reduceLRMinLR = 0.00000001
        layers = 9
        residual_cells = 1
        exponent = 6
        lr = 0.000001
        activation = 'elu'

        nb_samples_total = 250000
        nb_samples = 1000

        model_dict = inst.g2

    elif which_model == 'model2':
        epochs = 500
        dropout_first = 0.2
        dropout_middle = 0.0
        dropout_last = 0.5
        dropout = 0.2
        earlyStopPatience = 125
        reduceLRPatience = 40
        reduceLRFactor = 0.1
        reduceLRMinLR = 0.00000001
        layers = 12
        residual_cells = 2
        exponent = 8
        lr = 0.000001
        activation = 'elu'

        nb_samples_total = 250000
        nb_samples = 1000

        model_dict = inst.g2

else:
    if which_model == 'hw_model1':
        epochs = 500
        dropout_first = None
        dropout_middle = None
        dropout_last = None
        dropout = 0.2
        earlyStopPatience = 125
        reduceLRPatience = 40
        reduceLRFactor = 0.5
        reduceLRMinLR = 0.000009
        layers = 4
        residual_cells = 1
        exponent = 6
        lr = 0.001
        activation = 'elu'

        nb_samples_total = 100000
        nb_samples = 1000

        model_dict = inst.hullwhite_analytic

postfix = '_s'+str(nb_samples_total)+'_h'+str(history_part)+'_epochs'+str(epochs)
postfix_old1 = '_s'+str(nb_samples_total)+'_'+str(history_part)
postfix_old2 = '_s'+str(nb_samples_total)+'_'+str(history_part)+'_epochs'+str(epochs)


# ----------------------------------------------------------------------------------------------------------------------
# settings for hyperparameter optimization:
save_hp = True
append_hp = True
hp_print_results = True
hp_file = du.data_dir + 'hyperparameter_optimization_' + modelname + '_nn.csv'
hp_reversed = False
hp_shuffle = False
hp_requested_cpus = 18  # -1 means all cpus are used
param_grid_name = 'grid5_g2'

# parameter grids for hyperparameter optimization
swo = inst.get_swaptiongen(model_dict)
train_file = inst.sample_file_name(swo, size=nb_samples_total, with_error=with_error, history_part=history_part,
                                   history_end=history_end, history_start=history_start)

data_dir_hp = du.data_dir+'hyperparameter_optimization_index_files_'+param_grid_name+'/'

param_grid1_g2 = {'exp': [6, 8], 'layer': [9, 12], 'dof': [0.2, 0.5], 'dom': [0, 0.2, 0.5], 'dol': [0.2, 0.5],
                  'lr': [0.000001], 'activation': ['tanh', 'rbf', 'elu'], 'alpha': [1.0], 'residual_cells': [0, 1, 2],
                  'batch_size': [16, 32], 'epochs': [200], 'train_file': [train_file]}

param_grid2_g2 = {'exp': [6, 8], 'layer': [6, 9, 12], 'dof': [0.2], 'dom': [0.2], 'dol': [0.2],
                  'lr': [0.000001], 'activation': ['elu'], 'alpha': [1.0], 'residual_cells': [0, 1, 2],
                  'batch_size': [16], 'epochs': [1000], 'train_file': [train_file]}

param_grid_g2 = {'exp': [6, 8], 'layer': [6, 9, 12], 'dof': [0, 0.2, 0.5], 'dom': [0, 0.2, 0.5], 'dol': [0, 0.2, 0.5],
                 'lr': [0.00001, 0.000001, 0.0000001], 'activation': ['tanh', 'rbf', 'elu'],
                 'alpha': [1.0], 'residual_cells': [0, 1, 2], 'batch_size': [16, 32, 64],
                 'epochs': [200], 'train_file': [train_file]}

param_grid_hw = {'exp': [6, 8], 'layer': [4], 'dof': [0.2], 'dom': [0.2], 'dol': [0.2],
                 'lr': [0.1, 0.01, 0.001],
                 'activation': ['tanh'], 'alpha': [1.0], 'residual_cells': [0,1], 'batch_size': [16],
                 'epochs': [200], 'train_file': [train_file]}

param_grid3_g2 = [[8, 12, 0, 0, 0, 0.000001, 'elu', 1.0, 2, 16, 200, train_file, 0],
                  [8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 1.0, 2, 64, 200, train_file, 1],
                  [8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 1.0, 2, 128, 200, train_file, 2],
                  [8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 1.0, 2, 256, 200, train_file, 3],
                  [8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 1.0, 2, 512, 200, train_file, 4]]

param_grid4_g2 = [[8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 1.0, 2, 8, 200, train_file, 0],
                  [8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 2.0, 2, 16, 200, train_file, 1],
                  [8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 0.5, 2, 16, 200, train_file, 2],
                  [8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 0.1, 2, 16, 200, train_file, 3],
                  [8, 12, 0.2, 0, 0.5, 0.000001, 'elu', 0.0, 2, 16, 200, train_file, 4]]

param_grid5_g2 = {'exp': [8], 'layer': [12, 18, 34], 'dof': [0], 'dom': [0], 'dol': [0, 0.2, 0.5],
                 'lr': [0.000001], 'activation': ['elu'],
                 'alpha': [1.0], 'residual_cells': [1, 2], 'batch_size': [16],
                 'epochs': [200], 'train_file': [train_file]}


param_grid_test = {'exp': [4,3], 'layer': [6,3], 'dof': [0], 'dom': [0], 'dol': [0],
                   'lr': [0.00001],
                   'activation': ['tanh'], 'alpha': [1.0], 'residual_cells': [0],
                   'batch_size': [16, 64],
                   'epochs': [1], 'train_file': [train_file]}




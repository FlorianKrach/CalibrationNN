"""
author: Florian Krach
"""
import settings_NN as settings
import data_utils as du
import numpy as np
import pandas as pd


models = settings.performance_models
errors_NPV_total = np.zeros((len(settings.dates), 3*len(models)))
errors_Vol_total = np.zeros((len(settings.dates), 3*len(models)))
labels = []

average_2_month_error = np.zeros((3*len(models), 2))
start = int(settings.history_part * len(settings.dates))
end = start + 43
start_date = settings.dates[start]
end_date = settings.dates[end]

# go through all folders and extract the error data:
for i, model in enumerate(models):
    file_name = settings.data_dir+model+'/'+settings.error_filename_name
    errors = np.load(file_name)
    errors_NPV_total[:, 3*i:(3*i+3)] = errors[:, 0:3]
    errors_Vol_total[:, 3*i:(3*i+3)] = errors[:, 3:6]
    average_2_month_error[3*i:(3*i+3), 0] = np.mean(errors[start:(end+1), 0:3], axis=0)
    average_2_month_error[3*i:(3*i+3), 1] = np.mean(errors[start:(end+1), 3:6], axis=0)

    labels += ['Orig_'+model, 'Hist_'+model, 'FNN_'+model]

df_average_2_month_error = pd.DataFrame(data=average_2_month_error, index=labels,
                                        columns=['NPV_2m_av_err', 'Vol_2m_av_err'])
print
print 'The average NPV and Vol errors over 2 months after the training sample are:'
print 'start-date:', start_date
print 'end-date:  ', end_date
print df_average_2_month_error
print 'NPV:', df_average_2_month_error.iloc[4, 0], df_average_2_month_error.iloc[5, 0], df_average_2_month_error.iloc[8, 0]
print 'vol:', df_average_2_month_error.iloc[4, 1], df_average_2_month_error.iloc[5, 1], df_average_2_month_error.iloc[8, 1]
print
print 'plotting end date', settings.dates[settings.end_plotting]
print

labels[0] = 'local optimizer'
labels[4] = 'global optimizer'
labels[5] = 'model 1'
labels[8] = 'model 2'

# plot:
# plot NPV:
end_plotting = settings.end_plotting
index = settings.performance_index

if settings.save_plot:
    save_file = settings.data_dir+'plot_NPV_mean_squared_error.png'
else:
    save_file = None

du.plot_data(settings.dates[:end_plotting], errors_NPV_total[:end_plotting, index], labels=[labels[i] for i in index],
             figsize=settings.figsize, frame_lines=False,
             yticks_format=None, save=save_file, min_x_ticks=5, max_x_ticks=8,
             interval_multiples=True, colors=None, title='NPV mean squared error',
             legend_fontsize=14, legend_color=du.almost_black,
             title_fontsize=None, title_color=du.almost_black,
             xlabel=None, ylabel=None, xmin=0, xmax=None,
             xlabel_color=du.almost_black, ylabel_color=du.almost_black,
             xlabel_fontsize=14, ylabel_fontsize=14,
             xtick_fontsize=14, ytick_fontsize=14,
             xtick_color=du.almost_black, ytick_color=du.almost_black,
             out_of_sample=int(len(settings.dates)*settings.history_part), ytick_range=None, xtick_labels=None,
             show=settings.show_plot)

# plot Vol:
if settings.save_plot:
    save_file = settings.data_dir+'plot_Vol_absolute_mean_error.png'
else:
    save_file = None

du.plot_data(settings.dates[:end_plotting], errors_Vol_total[:end_plotting, index]*100, labels=[labels[i] for i in index],
             figsize=settings.figsize, frame_lines=False,
             yticks_format=None, save=save_file, min_x_ticks=5, max_x_ticks=8,
             interval_multiples=True, colors=None, title='Vol mean absolute error',
             legend_fontsize=14, legend_color=du.almost_black,
             title_fontsize=None, title_color=du.almost_black,
             xlabel=None, ylabel='%', xmin=0, xmax=None,
             xlabel_color=du.almost_black, ylabel_color=du.almost_black,
             xlabel_fontsize=14, ylabel_fontsize=14,
             xtick_fontsize=14, ytick_fontsize=14,
             xtick_color=du.almost_black, ytick_color=du.almost_black,
             out_of_sample=int(len(settings.dates)*settings.history_part), ytick_range=None, xtick_labels=None,
             show=settings.show_plot)





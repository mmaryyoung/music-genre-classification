"""
Trains a CRNN with a limited amount of data and saves the temporary model.

The training data is read from the result of the .raw_wave_to_melspects.py script as the melspects.npz.
The model is created by the .keras_crnn.py script.
The training is done iteratively through multiple optimizer configs, and cut off early when the accuracy
no longer improves.
Each training's learning curve will be printed via the .process_history.py script.
"""

import atexit
import numpy as np
import tensorflow as tf
from data_sanity_check import checkData
from keras_crnn import createCRNNModel
from keras.utils import np_utils
from operator import itemgetter
from process_history import plot_history
from tabulate import tabulate
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adamax, Adam, Nadam, RMSprop, SGD, Ftrl

import itertools
import datetime
import sys

# Where the learning curve figures go.
FIG_DIR_PATH = './learning_curve_figs/'
MELSPECTS_SOURCE_PATH = './gtzan/10secs/melspects.npz'
    
# Trains the global data with variable optimizer specs.
def train_with_config(x_tr, y_tr, x_te, y_te, opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=7)
    opt = _get_optimizer(opt_type, learning_rate)
    model = createCRNNModel(
        x_tr.shape[1:],
        conv_num=conv_num,
        conv_filter=conv_filter,
        conv_kernel_size=conv_kernel_size,
        conv_stride=conv_stride)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    history = model.fit(
        x=x_tr, y=y_tr,
        validation_data=(x_te, y_te),
        batch_size=20,
        epochs=100,
        callbacks=[early_stopping_callback],
        verbose=2,
        shuffle=True)
    return history

# Setup the fail-safe to always print out the results so far before exiting the program.
results_table = []
def handle_exit():
    # Prints the efficacy of different model configs in a table.
    print(tabulate(results_table, headers='keys', tablefmt='github'))
atexit.register(handle_exit)

# All potential network configurations.
opt_types = [
     'adam',
     'nadam',
     'rmsprop',
     'sgd',
    'adadelta',
     'adagrad',
    'adamax',
     'ftrl',
    ]
learning_rates = [
    1,
    0.1,
     0.01,
     0.001,
    0.0001]
conv_nums = [
    1,
    2,
    3]
conv_filters = [
    32,
    64,
    128]
conv_kernel_sizes = [
    3,
    5,
    10]
conv_strides = [
    # 1,
    3,
    5]

# Loads data from pre-precessed GTZAN dataset as melspectrograms.
x_tr, y_tr, x_te, y_te, x_cv, y_cv = _load_data(MELSPECTS_SOURCE_PATH)

# Makes sure the data passes basic sanity checks before moving forward.
if not checkData(x_tr, x_te, x_cv):
    sys.exit()

# Iterates through all the combinations of the different setup configs.
for combo in itertools.product(opt_types, learning_rates, conv_nums, conv_filters, conv_kernel_sizes, conv_strides):
    opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride = combo
    config_summary_str = 'opt=%s/lr=%f/conv#=%d/filter#=%d/kernel_size=%d/strides=%d' % (
        opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride
    )
    if _ignore_config_combo(combo):
        continue
    print('Training with model: %s.' % config_summary_str)
    try:
        history = train_with_config(x_tr, y_tr, x_te, y_te, opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride)
        # Prints and saves the best result from this config.
        max_acc_idx, max_acc = max(enumerate(history.history['val_categorical_accuracy']), key=itemgetter(1))
        print(
            'Done with the %s model. Best validation accuracy is %f, achieved at epoch #%d.' % (
            config_summary_str, max_acc, max_acc_idx + 1))
        results_table.append({
            'opt_type': opt_type,
            'learning_rate': learning_rate,
            'conv_num': conv_num,
            'conv_filter': conv_filter,
            'conv_kernel_size': conv_kernel_size,
            'conv_stride': conv_stride,
            'best_val_accuracy': max_acc,
            'best_val_accuracy_epoch': max_acc_idx + 1
        })
        # Save the learning curve figure from this config.
        batch_timestamp = str(datetime.datetime.now())
        fig_title = FIG_DIR_PATH + batch_timestamp + config_summary_str.replace('/', '-')
        plot_history(history, fig_title)
        print('Done saving the last learning curve. ')
    except ValueError as e:
        print('The config %s does not work. Error: %s.' % (config_summary_str, e))


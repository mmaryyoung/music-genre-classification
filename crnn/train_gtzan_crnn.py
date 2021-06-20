"""
Trains and fits a model to the GTZAN dataset.

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
from keras.optimizers import Adadelta, Adagrad, Adamax, Adam, Nadam, RMSprop, SGD, Ftrl
from operator import itemgetter
from process_history import plot_history
from tabulate import tabulate

import itertools
import datetime
import sys

# Where the learning curve figures go.
FIG_DIR_PATH = '/Users/maryyang/Learning/music-genre-classification/crnn/learning_curve_figs/'
MELSPECTS_SOURCE_PATH = '/Users/maryyang/Learning/music-genre-classification/crnn/gtzan/10secs/melspects.npz'

def load_data(filename=''):
    # Loads the data.
    tmp = np.load(filename)
    x_tr = tmp['x_tr']
    y_tr = tmp['y_tr']
    x_te = tmp['x_te']
    y_te = tmp['y_te']
    x_cv = tmp['x_cv']
    y_cv = tmp['y_cv']
    
    # Shuffles the data.
    tr_idx = np.random.permutation(len(x_tr))
    te_idx = np.random.permutation(len(x_te))
    cv_idx = np.random.permutation(len(x_cv))
    x_tr = x_tr[tr_idx]
    y_tr = y_tr[tr_idx]
    x_te = x_te[te_idx]
    y_te = y_te[te_idx]
    x_cv = x_cv[cv_idx]
    y_cv = y_cv[cv_idx]

    # Normalizes the data to be below 1.
    x_tr = x_tr / np.amax(x_tr)
    x_te = x_te / np.amax(x_te)
    x_cv = x_cv / np.amax(x_cv)

    # Supplements the data with a new axis.
    x_tr = x_tr[:,:,:,np.newaxis]
    x_te = x_te[:,:,:,np.newaxis]
    x_cv = x_cv[:,:,:,np.newaxis]

    # One-hot encodes the categorical labels.
    y_tr = np_utils.to_categorical(y_tr)
    y_te = np_utils.to_categorical(y_te)
    y_cv = np_utils.to_categorical(y_cv)
    return x_tr, y_tr, x_te, y_te, x_cv, y_cv

# Trains the global data with variable optimizer specs.
def train_with_config(opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=7)
    if opt_type == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif opt_type == 'nadam':
        opt = Nadam(learning_rate=learning_rate)
    elif opt_type == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif opt_type == 'adadelta':
        opt = Adadelta(learning_rate=learning_rate)
    elif opt_type == 'adagrad':
        opt = Adagrad(learning_rate=learning_rate)
    elif opt_type == 'adamax':
        opt = Adamax(learning_rate=learning_rate)
    elif opt_type == 'ftrl':
        opt = Ftrl(learning_rate=learning_rate)
    elif opt_type == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise NotImplementedError('Optimizer type %s is not implemented.' % opt_type)
    opt = Nadam(learning_rate=0.0005)
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

# Setup the failt-safe to always print out the results so far before exiting the program.
results_table = []
def handle_exit():
    # Prints the efficacy of different model configs in a table.
    print(tabulate(results_table, headers='keys', tablefmt='github'))
atexit.register(handle_exit)

# All potential network configurations.
opt_types = [
    'adam',
    # 'nadam',
    # 'rmsprop',
    'sgd',
    'adadelta',
    # 'adagrad',
    'adamax',
    # 'ftrl',
    ]
learning_rates = [
    1,
    0.1,
    # 0.01,
    # 0.001,
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
    1,
    3,
    5]

# Loads data from pre-precessed GTZAN dataset as melspectrograms.
x_tr, y_tr, x_te, y_te, x_cv, y_cv = load_data(MELSPECTS_SOURCE_PATH)

# Makes sure the data passes basic sanity checks before moving forward.
if not checkData(x_tr, x_te, x_cv):
    sys.exit()

# Iterates through all the combinations of the different setup configs.
for combo in itertools.product(opt_types, learning_rates, conv_nums, conv_filters, conv_kernel_sizes, conv_strides):
    opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride = combo
    config_summary_str = 'opt=%s/lr=%f/conv#=%d/filter#=%d/kernel_size=%d/strides=%d' % (
        opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride
    )
    print('Training with model: %s.' % config_summary_str)
    try:
        history = train_with_config(opt_type, learning_rate, conv_num, conv_filter, conv_kernel_size, conv_stride)
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
        fig_title = FIG_DIR_PATH + batch_timestamp + str(pair[0]) + '-' + str(pair[1])
        plot_history(history, fig_title)
        print('Done saving the last learning curve. ')
    except ValueError as e:
        print('The config %s does not work. Error: %s.' % (config_summary_str, e))


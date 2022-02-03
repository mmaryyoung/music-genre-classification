"""
Trains a CRNN with a limited amount of data and saves the temporary model.

The training data is read from the result of the .raw_wave_to_melspects.py script as the melspects.npz.
The model is created by the .keras_crnn.py script.
The training is done iteratively through multiple optimizer configs, and cut off early when the accuracy
no longer improves.
Each training's learning curve will be printed via the .process_history.py script.
"""

import argparse
import atexit
import csv
import training_utils
from data_sanity_check import checkData
import numpy as np
from operator import itemgetter
import os
from process_history import plot_history
from tabulate import tabulate
from tensorflow import keras

import datetime
import sys

# Where the learning curve figures go.
FIG_DIR_PATH = './learning_curve_figs/'
MELSPECTS_SOURCE_PATH = './gtzan/10secs/melspects.npz'

# Parsing command line arguments.
arg_parser = argparse.ArgumentParser(
    description='Trains a CRNN model with variable amount of data.')
arg_parser.add_argument(
    'train', help='The absolute path of a directory where the training data resides.')
arg_parser.add_argument(
    'test',
    help='The absolute path of a directory where the testing data resides.')
arg_parser.add_argument(
    'save_model',
    help='The absolute path where the saved model will be.')
arg_parser.add_argument(
    '-m', '--model', help='Start with a previously saved model and provide its path.')
arg_parser.add_argument(
    '-pf', '--processed_files', help='A file path that contains the song names that the current model already processed.')
arg_parser.add_argument(
    '-opt', '--opt_type', default='nadam', help='The optimizer type. Default is Nadam.')
arg_parser.add_argument(
    '-lr', '--learning_rate', default=0.001, help='The learning rate of the optimizer. Default is 0.001')
arg_parser.add_argument(
    '-cn', '--conv_num', default=2, help='The number of convolutional layers. Default is 2.')
arg_parser.add_argument(
    '-fn', '--conv_filter_num', default=32, help='The number of filters in a convolutional layer. Default is 32.')
arg_parser.add_argument(
    '-ks', '--conv_kernel_size', default=10, help='The size of a square shaped convolutional kernel. Default is 10.')
arg_parser.add_argument(
    '-cs', '--conv_stride', default=3, help='The distance between convolutional kernels. Default is 3.')
args = arg_parser.parse_args()

# Setup the fail-safe to always save the model before exiting the program.
current_model = keras.models.load_model(args.m) if args.m else None
processed_files = []
if args.ps:
    with open(args.ps, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        processed_files = [name for name in csvreader]
def handle_exit():
    # Saves the current model state.
    if current_model:
        current_model.save(args.save_model)
atexit.register(handle_exit)

# Set up all the training configs.
config = {
    'opt_type': args.opt,
    'learning_rate': args.lr,
    'conv_num': args.cn,
    'conv_filter_num': args.fn,
    'conv_kernel_size': args.ks,
    'conv_stride': args.cs
}
config_summary_str = training_utils.dict_to_string(config)
print('Training with model: %s.' % config_summary_str)

# Load the testing files.
print('Processing test files from %s' % args.test)
x_test = []
for root, dirs, files in os.walk(args.test):
    print(root)
    for f in files:
        x_test = x_test + np.load(root + '/' + f)
print('Testing data shape: %s' % str(x_test.shape))

for root, dirs, files in os.walk(args.train):
    print(root)
    for name in files:
        if name not in processed_files:
            x_train = np.load(root + '/' + name)
            try:
                history, current_model = training_utils.train_with_config(x_train, y_tr, x_test, y_te, **config)
                # Prints and saves the best result from this config.
                max_acc_idx, max_acc = max(enumerate(history.history['val_categorical_accuracy']), key=itemgetter(1))
                print(
                    'Done with the %s model. Best validation accuracy is %f, achieved at epoch #%d.' % (
                    config_summary_str, max_acc, max_acc_idx + 1))
                # Save the learning curve figure from this config.
                batch_timestamp = str(datetime.datetime.now())
                fig_title = FIG_DIR_PATH + batch_timestamp + config_summary_str.replace('/', '-')
                plot_history(history, fig_title)
                print('Done saving the last learning curve. ')
            except ValueError as e:
                print('The config %s does not work. Error: %s.' % (config_summary_str, e))


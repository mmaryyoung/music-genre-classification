"""
Trains a CRNN with a limited amount of data and saves the temporary model.

The training data is read from the result of the .raw_wave_to_melspects.py script as the melspects.npz.
The model is created by the .keras_crnn.py script.
The training is done iteratively through multiple optimizer configs, and cut off early when the accuracy
no longer improves.
Each training's learning curve will be printed via the .process_history.py script.

Example command:
python3 batch_train.py gtzan/fma/train gtzan/fma/test \
    trained_models/20200223.m trained_models/20200223.csv \
    /Users/maryyang/Downloads/fma_metadata/tracks.csv
"""

import argparse
import atexit
import csv
from pkgutil import get_data
import file_loader_utils
import training_utils
from parse_track_genres import get_track_genre_map, get_all_genres
import pandas as pd
import numpy as np
import os
from process_history import plot_history
from tabulate import tabulate
from tensorflow import keras

# Where the learning curve figures go.
FIG_DIR_PATH = './learning_curve_figs/'

# Constants
INITIAL_TRAINING_SIZE = 1000
FOLLOWUP_TRAINING_SIZE = 100

# Parsing command line arguments.
arg_parser = argparse.ArgumentParser(
    description='Trains a CRNN model with variable amount of data.')
arg_parser.add_argument(
    'train', help='The absolute path of a directory where the training data resides.')
arg_parser.add_argument(
    'test', help='The absolute path of a directory where the testing data resides.')
arg_parser.add_argument(
    'save_model', help='The absolute path where the saved model will be.')
arg_parser.add_argument(
    'processed_files', help='A file path that contains the song names that the current model already processed.')
arg_parser.add_argument(
    'genre_map', help='The CSV file that contains the track_id-to-genre mapping.')
arg_parser.add_argument(
    '-m', '--model', help='Start with a previously saved model and provide its path.')
arg_parser.add_argument(
    '-opt', '--opt_type', default='nadam', help='The optimizer type. Default is Nadam.')
arg_parser.add_argument(
    '-lr', '--learning_rate', default=0.001, help='The learning rate of the optimizer. Default is 0.001')
arg_parser.add_argument(
    '-cn', '--conv_num', default=2, help='The number of convolutional layers. Default is 2.')
arg_parser.add_argument(
    '-fn', '--conv_filter', default=32, help='The number of filters in a convolutional layer. Default is 32.')
arg_parser.add_argument(
    '-ks', '--conv_kernel_size', default=10, help='The size of a square shaped convolutional kernel. Default is 10.')
arg_parser.add_argument(
    '-cs', '--conv_stride', default=3, help='The distance between convolutional kernels. Default is 3.')
args = arg_parser.parse_args()

# Processes previous model and records if any.
current_model = keras.models.load_model(args.model) if args.model else None
processed_files = []
try:
    with open(args.processed_files, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        processed_files = [name for name in csvreader]
except IOError:
    # If the file does not exist, do nothing.
    print("Processed file list does not exist. Starting afresh.")
    pass

# Setup the fail-safe to always save the model before exiting the program.
def handle_exit():
    # Saves the current model state.
    print("Exiting now. Trying to save the current model.")
    if current_model:
        current_model.save(args.save_model)
        print("Saved current model at %s. " % args.save_model)
    if processed_files:
        pd.Series(processed_files).to_csv(args.processed_files, index=False)
        print("Saved processed file list to %s. " % args.processed_files)

atexit.register(handle_exit)

# Set up all the training configs.
config = {
    'opt_type': args.opt_type,
    'learning_rate': args.learning_rate,
    'conv_num': args.conv_num,
    'conv_filter': args.conv_filter,
    'conv_kernel_size': args.conv_kernel_size,
    'conv_stride': args.conv_stride
}
config_summary_str = training_utils.dict_to_string(config)
print('Training with model: %s.' % config_summary_str)

# Load the track-to-genre map
genre_map = get_track_genre_map(args.genre_map)

all_train_paths = file_loader_utils.get_all_track_paths(args.train, processed_files)
all_test_paths = file_loader_utils.get_all_track_paths(args.test, processed_files)

all_genres = sorted(
    get_all_genres(genre_map, all_train_paths).union(get_all_genres(genre_map, all_test_paths)))

# Load the testing files.
shuffled_testing_idx = np.random.permutation(len(all_test_paths))
all_test_paths = np.array(all_test_paths)[shuffled_testing_idx]
x_test, y_test = file_loader_utils.get_data_labels_from_paths(all_test_paths, genre_map)
x_test = file_loader_utils.prep_x_data(x_test)
y_test = file_loader_utils.prep_y_data(y_test, all_genres)

print('Testing data shape: %s' % str(x_test.shape))
print('Testing labels shape: %s' % str(y_test.shape))

# Decide if this is the first ever run, if so, load more data at once.
train_batch_size = INITIAL_TRAINING_SIZE if current_model is None else FOLLOWUP_TRAINING_SIZE

# Shuffle the training files for reading.
shuffled_training_idx = np.random.permutation(len(all_train_paths))
all_train_paths = np.array(all_train_paths)[shuffled_training_idx]

print("All candidate training files: %d files. " % len(all_train_paths))

while len(all_train_paths) >= train_batch_size:
    # Load the training files
    x_train, y_train = file_loader_utils.get_data_labels_from_paths(all_train_paths[:train_batch_size], genre_map)
    x_train = file_loader_utils.prep_x_data(x_train)
    y_train = file_loader_utils.prep_y_data(y_train, all_genres)
    try:
        if current_model:
            current_model = training_utils.train_with_model(
                x_train, y_train, x_test, y_test, current_model, config['opt_type'], config['learning_rate']/10)
        else:
            current_model = training_utils.train_with_all_configs(
                training_utils.get_all_configs(), x_train, y_train, x_test, y_test)
        # Marks the files as "processed".
        processed_files += [os.path.basename(file) for file in all_train_paths[train_batch_size:]]
        all_train_paths = all_train_paths[train_batch_size:]
        print("Size of processed files %d" % len(processed_files))
        print("Size of all_train_paths %d" % len(all_train_paths))
        
    except ValueError as e:
        print('The config %s does not work. Error: %s.' % (config_summary_str, e))
        break
    train_batch_size = 10

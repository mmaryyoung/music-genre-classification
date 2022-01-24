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
import training_utils
from data_sanity_check import checkData
from load_training_data import load_data
from operator import itemgetter
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

args = arg_parser.parse_args()

# Setup the fail-safe to always print out the results and save the model before exiting the program.
results_table = []
current_model = keras.models.load_model(args.model)
def handle_exit():
    # Prints the efficacy of different model configs in a table.
    print(tabulate(results_table, headers='keys', tablefmt='github'))
atexit.register(handle_exit)

# Loads data from pre-precessed GTZAN dataset as melspectrograms.
x_tr, y_tr, x_te, y_te, x_cv, y_cv = load_data(MELSPECTS_SOURCE_PATH)

# Makes sure the data passes basic sanity checks before moving forward.
if not checkData(x_tr, x_te, x_cv):
    sys.exit()

# Iterates through all the combinations of the different setup configs.
for config in training_utils.get_all_configs():
    config_summary_str = training_utils.dict_to_string(config)
    if training_utils.is_ignore_config(config):
        continue
    print('Training with model: %s.' % config_summary_str)
    try:
        history, model = training_utils.train_with_config(x_tr, y_tr, x_te, y_te, **config)
        # Prints and saves the best result from this config.
        max_acc_idx, max_acc = max(enumerate(history.history['val_categorical_accuracy']), key=itemgetter(1))
        print(
            'Done with the %s model. Best validation accuracy is %f, achieved at epoch #%d.' % (
            config_summary_str, max_acc, max_acc_idx + 1))
        results_table.append({
            **config,
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


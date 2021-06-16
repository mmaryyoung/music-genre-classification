"""
Trains and fits a model to the GTZAN dataset.

The training data is read from the result of the .raw_wave_to_melspects.py script as the melspects.npz.
The model is created by the .keras_crnn.py script.
The training is done iteratively through multiple optimizer configs, and cut off early when the accuracy
no longer improves.
Each training's learning curve will be printed via the .process_history.py script.
"""

import numpy as np
import tensorflow as tf
from data_sanity_check import checkData
from keras_crnn import createCRNNModel
from keras.utils import np_utils
from keras.optimizers import Adadelta, Adagrad, Adamax, Adam, Nadam, RMSprop, SGD, Ftrl
from operator import itemgetter
from process_history import plot_history
import tabulate

import itertools
import datetime
import sys

# Where the learning curve figures go.
FIG_DIR_PATH = '/Users/maryyang/Learning/music-genre-classification/crnn/learning_curve_figs/'
MELSPECTS_SOURCE_PATH = '/Users/maryyang/Learning/music-genre-classification/crnn/gtzan/10secs/melspects.npz'

def loadall(filename=''):
    tmp = np.load(filename)
    x_tr = tmp['x_tr']
    y_tr = tmp['y_tr']
    x_te = tmp['x_te']
    y_te = tmp['y_te']
    x_cv = tmp['x_cv']
    y_cv = tmp['y_cv']
    return {'x_tr' : x_tr, 'y_tr' : y_tr,
            'x_te' : x_te, 'y_te' : y_te,
            'x_cv' : x_cv, 'y_cv' : y_cv, }

# Loads data from pre-precessed GTZAN dataset as melspectrograms.
data = loadall(MELSPECTS_SOURCE_PATH)

x_tr = data['x_tr']
y_tr = data['y_tr']
x_te = data['x_te']
y_te = data['y_te']
x_cv = data['x_cv']
y_cv = data['y_cv']

tr_idx = np.random.permutation(len(x_tr))
te_idx = np.random.permutation(len(x_te))
cv_idx = np.random.permutation(len(x_cv))

x_tr = x_tr[tr_idx]
y_tr = y_tr[tr_idx]
x_te = x_te[te_idx]
y_te = y_te[te_idx]
x_cv = x_cv[cv_idx]
y_cv = y_cv[cv_idx]

x_tr = x_tr / np.amax(x_tr)
x_te = x_te / np.amax(x_te)
x_cv = x_cv / np.amax(x_cv)

x_tr = x_tr[:,:,:,np.newaxis]
x_te = x_te[:,:,:,np.newaxis]
x_cv = x_cv[:,:,:,np.newaxis]

y_tr = np_utils.to_categorical(y_tr)
y_te = np_utils.to_categorical(y_te)
y_cv = np_utils.to_categorical(y_cv)

# Makes sure the data passes basic sanity checks before moving forward.
if not checkData(x_tr, x_te, x_cv):
    sys.exit()

# Trains the global data with variable optimizer specs.
def variable_training(opt_type, learning_rate):
    print('Running training with optimizer %s and learning rate of %f.' % (opt_type, learning_rate))
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
    model = createCRNNModel(x_tr.shape[1:])
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


opt_types = ['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad', 'adamax', 'ftrl']
learning_rates = [1, 0.1, 0.01, 0.001, 0.0005, 0.0001]

results_table = []
# Iterates through all the combinations of the different setup configs.
for pair in itertools.product(opt_types, learning_rates):
    opt_type = pair[0]
    learning_rate = pair[1]
    history = variable_training(opt_type, learning_rate)
    # Prints and saves the best result from this config.
    max_acc_idx, max_acc = max(enumerate(history.history['val_categorical_accuracy']), key=itemgetter(1))
    print('Done with the %s/%s model. Best validation accuracy is %f, achieved at epoch #%d' % (
        opt_type, learning_rate, max_acc, max_acc_idx + 1))
    results_table.append({
        'opt_type': opt_type,
        'learning_rate': learning_rate,
        'best_val_accuracy': max_acc,
        'best_val_accuracy_epoch': max_acc_idx + 1
    })
    # Save the learning curve figure from this config.
    batch_timestamp = str(datetime.datetime.now())
    fig_title = FIG_DIR_PATH + batch_timestamp + str(pair[0]) + '-' + str(pair[1])
    plot_history(history, fig_title)
    print('Done saving the last learning curve. ')

print(tabulate(results_table, headers='keys', tablefmt='fancy_grid'))

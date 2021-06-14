import numpy as np
import tensorflow as tf
from data_sanity_check import checkData
from keras_crnn import createCRNNModel
from keras.utils import np_utils
from keras.optimizers import Adadelta, Adagrad, Adamax, Adam, Nadam, RMSprop, SGD, Ftrl
from process_history import plot_history

import itertools
import datetime
import sys

# Where the learning curve figures go.
FIG_DIR_PATH = '/Users/maryyang/Learning/music-genre-classification/crnn/figs/'

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

data = loadall('melspects.npz')

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

if not checkData(x_tr, x_te, x_cv):
    sys.exit()


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
    print('Done with this model spec. Best validation accuracy is %f.' % max(history.history['val_categorical_accuracy']))
    batch_timestamp = str(datetime.datetime.now())
    fig_title = batch_timestamp + str(opt_type) + '-' + str(learning_rate)
    plt = plot_history(history, fig_title)
    print('Done saving the last learning curve. ')


opt_types = ['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad', 'adamax', 'ftrl']
learning_rates = [1, 0.1, 0.01, 0.001, 0.0005, 0.0001]

for pair in itertools.product(opt_types, learning_rates):
    variable_training(pair[0], pair[1])

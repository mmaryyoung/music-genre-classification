import numpy as np
from keras.utils import np_utils

# Takes a file name (of npz), which should contain a training set,
# a testing set, and a validation set. It shuffles each set within
# their own, normalizes the data, supplements the data with a new
# inner axis, and returns the sets in the order of:
# training samples, training labels, testing samples, testing labels,
# validation samples, and validation labels.
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
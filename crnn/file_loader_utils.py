import numpy as np
import os
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

# Takes a np array of shape (num_samples, num_mels, chunks_per_sample)
# and turns it into normalized and shape of
# (num_samples, num_mels, chunks_per_sample, 1)
def prep_x_data(x_data):
  # Normalizes the data to be below 1.
  x_data = x_data / np.amax(x_data)
  # Supplements the data with a new axis.
  x_data = x_data[:,:,:,np.newaxis]
  return np.array(x_data)

# Takes a 1-D np array containing num_samples of string illustrated genre labels
# and a genre map.
# Turns the np array into a one-hot encoded array of shape (num_samples, num_genres).
# Note that the num_genres here includes all possible genres in the map, not just the
# ones appearing in this y_data.
def prep_y_data(y_data, all_genres):
    y_data = [all_genres.index(label) for label in y_data]
    return np_utils.to_categorical(y_data, num_classes=len(all_genres))

# Reads all the data samples (ends in .npy) in a directory and retrieves
# the corresponding label for the data, based on the file name. The file
# name should exist in the genre map as the track id.
def get_data_labels_from_paths(paths, genre_map):
    data = []
    labels = []
    for file in paths:
        if 'npy' not in file:
            continue
        new_samples = np.load(file)
        data = np.concatenate((data, new_samples)) if len(data) > 0 else new_samples
        labels = labels + [genre_map.at[str(int(os.path.basename(file).split('.')[0])), 'genre_top']] * len(new_samples)
    return data, np.array(labels)

# Given a root directory, load all the files excepting the ones in a blocklist.
def get_all_track_paths(track_root, excluded_files=None):
    print("Processing tracks in %s" % track_root)
    paths = []
    for root, _, files in os.walk(track_root):
        for name in files:
            if not excluded_files or name not in excluded_files:
                paths.append(root + '/' + name)
    print("Processed %d files. " % len(paths))
    return paths
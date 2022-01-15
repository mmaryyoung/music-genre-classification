import pickle
import sPickle
import numpy as np
import random
import sys
import os

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

for root, dirs, files in os.walk('/data/hibbslab/jyang/outputs/biResults/'):
    for file in files:
        if '.hdf5' in file:
            g = file.split("Weights")[0]
            model = Sequential()

            model.add(Conv2D(32, (3, 3), padding='same',
                             input_shape=(128, 126, 1), name='conv1'+g))
            model.add(Activation('relu', name='relu1'+g))
            model.add(Conv2D(32, (3, 3), name='conv2'+g))
            model.add(Activation('relu', name='relu2'+g))
            model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'+g))
            model.add(Dropout(0.25, name='dropout1'+g))

            model.add(Conv2D(64, (3, 3), padding='same', name='conv3'+g))
            model.add(Activation('relu', name='relu3'+g))
            model.add(Conv2D(64, (3, 3), name='conv4'+g))
            model.add(Activation('relu', name='relu4'+g))
            model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'+g))
            model.add(Dropout(0.25, name='dropout2'+g))

            model.add(Flatten(name='flatten'+g))
            model.add(Dense(128, name='dense1'+g))
            model.add(Activation('relu', name='relu5'+g))
            model.add(Dropout(0.5, name='dropout3'+g))
            model.add(Dense(2, name='dense2'+g))
            model.add(Activation('softmax', name='softmax'+g))
            model.load_weights(root + file)
            model.save_weights('/data/hibbslab/jyang/outputs/bModels/' + file)


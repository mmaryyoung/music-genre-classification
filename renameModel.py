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



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]), name='conv1')
model.add(Activation('relu'), name='relu1')
model.add(Conv2D(32, (3, 3)), name='conv2')
model.add(Activation('relu'), name='relu2')
model.add(MaxPooling2D(pool_size=(2, 2)), name='pool1')
model.add(Dropout(0.25), name='dropout1')

model.add(Conv2D(64, (3, 3), padding='same'), name='conv3')
model.add(Activation('relu'), name='relu3')
model.add(Conv2D(64, (3, 3)), name='conv4')
model.add(Activation('relu'), name='relu4')
model.add(MaxPooling2D(pool_size=(2, 2)), name='pool2')
model.add(Dropout(0.25), name='dropout2')

model.add(Flatten(), name='flatten')
model.add(Dense(128), name='dense1')
model.add(Activation('relu'), name='relu5')
model.add(Dropout(0.5), name='dropout3')
model.add(Dense(num_classes), name='dense2')
model.add(Activation('softmax'), name='softmax')

for root, dirs, files in os.walk('/data/hibbslab/jyang/outputs/biResults/'):
    for file in files:
        if '.hdf5' in file:
            model.load_weights(root + file)
            model.save_weights('/data/hibbslab/jyang/outputs/bModels/' + file)


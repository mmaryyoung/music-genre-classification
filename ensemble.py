import pickle
import sPickle
import numpy as np
import random
import sys
import os

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, merge, concatenate, Concatenate, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True

dataPath = '/data/hibbslab/jyang/tzanetakis/ver5.0/'
x_train = pickle.load(open(dataPath + 'x_train_mel.p', 'rb'))
y_train = pickle.load(open(dataPath + 'y_train_mel.p', 'rb'))
x_test = pickle.load(open(dataPath + 'x_test_mel.p', 'rb'))
y_test = pickle.load(open(dataPath + 'y_test_mel.p', 'rb'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=1e-6, decay=1e-10)
# was 2e-3 for a long time (first batch of binary data)
nadam = keras.optimizers.Nadam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), schedule_decay=0.004)

firstLayer = []
firstInput = Input(shape=(128, 126, 1))
for root, dirs, files in os.walk('/data/hibbslab/jyang/outputs/bModels/'):
    for file in files:
        if '.hdf5' in file:
            g = file.split('Weights')[0]
            conv = Conv2D(32, (3, 3), input_shape=(128,126,1), padding='same',name='conv1'+g, trainable=False)(firstInput)
            merged = Activation('relu', name='relu1' + g)(conv)
            merged = Conv2D(32, (3, 3), name='conv2'+ g)(merged)
            merged = Activation('relu', name='relu2'+ g)(merged)
            merged = MaxPooling2D(pool_size=(2, 2), name='pool1'+ g)(merged)
            merged = Dropout(0.25, name='dropout1'+ g)(merged)

            merged = Conv2D(64, (3, 3), padding='same', name='conv3'+ g)(merged)
            merged = Activation('relu', name='relu3'+ g)(merged)
            merged = Conv2D(64, (3, 3), name='conv4'+ g)(merged)
            merged = Activation('relu', name='relu4'+ g)(merged)
            merged = MaxPooling2D(pool_size=(2, 2), name='pool2'+ g)(merged)
            merged = Dropout(0.25, name='dropout2'+ g)(merged)

            merged = Flatten(name='flatten'+ g)(merged)
            merged = Dense(128, name='dense1'+ g)(merged)
            merged = Activation('relu', name='relu5'+ g)(merged)
            merged = Dropout(0.5, name='dropout3'+ g)(merged)
            merged = Dense(2, name='dense2'+ g)(merged)
            merged = Activation('softmax', name='softmax'+ g)(merged)
            model = Model(firstInput, merged)
            model.load_weights(root + file, by_name=True)
            single_result = model.predict(x_test)
            print("shape of single result: " + str(single_result.shape))







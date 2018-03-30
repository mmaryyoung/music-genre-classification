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

firstLayer = []
firstInput = Input(shape=(128, 126, 1))
for root, dirs, files in os.walk('/data/hibbslab/jyang/outputs/bModels/'):
    for file in files:
        if '.hdf5' in file:
            #oneG = Sequential()
            conv = Conv2D(32, (3, 3), padding='same',name='conv1', trainable=False)(firstInput)
            #oneG.add(Conv2D(32, (3, 3), padding='same',
            #         input_shape=(128,126,1), name='conv1', trainable=False))
            m = Model(input=firstInput, output=conv)
            m.load_weights(root + file, by_name=True)
            firstLayer.append(m.get_layer('conv1'))
assert(len(firstLayer) == num_classes)

merged = concatenate(firstLayer, axis=1)
#part1 = Model(input=[firstInput]*num_classes, )

# not sure about axis here and input shape
#merged = merge(firstLayer,mode='concat', concat_axis=1)

merged = Activation('relu', name='relu1')(merged)
merged = Conv2D(32, (3, 3), name='conv2')(merged)
merged = Activation('relu', name='relu2')(merged)
merged = MaxPooling2D(pool_size=(2, 2), name='pool1')(merged)
merged = Dropout(0.25, name='dropout1')(merged)

merged = Conv2D(64, (3, 3), padding='same', name='conv3')(merged)
merged = Activation('relu', name='relu3')(merged)
merged = Conv2D(64, (3, 3), name='conv4')(merged)
merged = Activation('relu', name='relu4')(merged)
merged = MaxPooling2D(pool_size=(2, 2), name='pool2')(merged)
merged = Dropout(0.25, name='dropout2')(merged)

merged = Flatten(name='flatten')(merged)
merged = Dense(128, name='dense1')(merged)
merged = Activation('relu', name='relu5')(merged)
merged = Dropout(0.5, name='dropout3')(merged)
merged = Dense(num_classes, name='dense2')(merged)
merged = Activation('softmax', name='softmax')(merged)

model = Model(firstInput, merged)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=1e-6, decay=1e-10)
# was 2e-3 for a long time (first batch of binary data)
nadam = keras.optimizers.Nadam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), schedule_decay=0.004)
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=nadam,
              metrics=['accuracy'])

print(np.amax(x_test))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.amax(x_train)
x_test /= np.amax(x_test)

print('Not using data augmentation.')

model.fit([x_train]*num_classes, y_train, batch_size=batch_size, 
	epochs=epochs,
	validation_data=([x_test]*num_classes, y_test), shuffle=True)




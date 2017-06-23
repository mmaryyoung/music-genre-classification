import pickle
import random
import numpy as np

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#K.set_image_dim_ordering('th')

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True

x_train = []
y_train = []

for i in range(800):
	sample = np.random.random((100,110))
	sample = map(lambda row: map(lambda x: [x],row),sample)
	x_train.append(sample)
	ysample = [0]*10
	ysample[i%10] = 1
	y_train.append(ysample)

x_test = []
y_test = []

for i in range(200):
	sample = np.random.random((100,110))
	sample = map(lambda row: map(lambda x: [x],row),sample)
	x_test.append(sample)
	ysample = [0]*10
	ysample[i%10] = 1
	y_test.append(ysample)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

print('Not using data augmentation.')

print("x_train: "+ str(x_train.shape))
print("y_train: "+ str(y_train.shape))
model.fit(x_train, y_train, batch_size=batch_size, 
	epochs=epochs,
	validation_data=(x_test, y_test), shuffle=True)

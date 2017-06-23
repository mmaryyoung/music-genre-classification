import os
import pickle
import numpy as np


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# TODO:
# Read each folder in
# Slice every file into 1 second chunks
# Put every chunk into x_train
# Put corresponding label array into y_train


x_train = []
y_train = []
x_test = []
y_test = []


gid = 0
for root, dirs, files in os.walk('Homemade Dataset'):
	if '_pickle' in root:
		sid = 0
		for name in files:
			if '.p' in name:
				print root + '/' + name
				longgrid = pickle.load(open(root + '/' + name, 'rb'))
				chunks = [longgrid[x:x+100] for x in range(0, len(longgrid),100)]

				oneLabel = [0]*10
				oneLabel[gid] = 1
				# Assumes at least 50 seconds
				# Train:test = 5:1 if len(chunks)==60
				[x_train.append(x) for x in chunks[:50]]
				[y_train.append(x) for x in [oneLabel]*50]
				[x_test.append(x) for x in chunks[50:]]
				[y_test.append(x) for x in [oneLabel]*(len(chunks)-50)]
				sid +=1
		gid += 1


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print 'x_train: ', x_train.shape
print 'y_train: ',  y_train.shape
print 'x_test: ', x_test.shape
print 'y_test: ', y_test.shape


#K.set_image_dim_ordering('th')

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True


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
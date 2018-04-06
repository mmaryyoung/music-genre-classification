import pickle
import sPickle
import numpy as np

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def genToArr(gen):
	lst = []
	for i in gen:
		lst.append(i)
	return np.asarray(lst)

#K.set_image_dim_ordering('th')

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True

dataPath = '/data/hibbslab/jyang/tzanetakis/ver5.0/'
x_train = pickle.load(open(dataPath + 'x_train_mel.p', 'rb'))
y_train = pickle.load(open(dataPath + 'y_train_mel.p', 'rb'))
x_test = pickle.load(open(dataPath + 'x_test_mel.p', 'rb'))
y_test = pickle.load(open(dataPath + 'y_test_mel.p', 'rb'))

#x_train_g = sPickle.s_load(open(dataPath + 'x_train.p', 'rb'))
#y_train_g = sPickle.s_load(open(dataPath + 'y_train.p', 'rb'))
#x_test_g = sPickle.s_load(open(dataPath + 'x_test.p', 'rb'))
#y_test_g = sPickle.s_load(open(dataPath + 'y_test.p', 'rb'))

#x_train = genToArr(x_train_g)
#y_train = genToArr(y_train_g)
#x_test = genToArr(x_test_g)
#y_test = genToArr(y_test_g)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)


# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)


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
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=1e-6, decay=1e-10)
nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), schedule_decay=0.004)
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

print("x_train: "+ str(x_train.shape))
print("y_train: "+ str(y_train.shape))
model.fit(x_train, y_train, batch_size=batch_size, 
	epochs=epochs,
    validation_data=(x_test, y_test), shuffle=True)

model.save('cnn3sec.h5')

import pickle
import sPickle
import numpy as np
import random

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
num_classes = 2
epochs = 200
data_augmentation = True

dataPath = '/data/hibbslab/jyang/tzanetakis/ver5.0/'
x_train = pickle.load(open(dataPath + 'x_train_mel.p', 'rb'))
y_train = pickle.load(open(dataPath + 'y_train_mel.p', 'rb'))
x_test = pickle.load(open(dataPath + 'x_test_mel.p', 'rb'))
y_test = pickle.load(open(dataPath + 'y_test_mel.p', 'rb'))

# x_train_g = sPickle.s_load(open(dataPath + 'x_train.p', 'rb'))
# y_train_g = sPickle.s_load(open(dataPath + 'y_train.p', 'rb'))
# x_test_g = sPickle.s_load(open(dataPath + 'x_test.p', 'rb'))
# y_test_g = sPickle.s_load(open(dataPath + 'y_test.p', 'rb'))

# x_train = genToArr(x_train_g)
# y_train = genToArr(y_train_g)
# x_test = genToArr(x_test_g)
# y_test = genToArr(y_test_g)

gtzan_genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

###### TWO GENRE ADAPTATION ######

this_genre = "metal"
genre_idx = gtzan_genres.index(this_genre)

y_train = [[1,0] if y[genre_idx] else [0,1] for y in y_train]
y_test = [[1,0] if y[genre_idx] else [0,1] for y in y_test]
y_train = np.array(y_train)
y_test = np.array(y_test)

idx_pool = []
other_pool = []
for i in range(len(y_train)):
    if y_train[i][0] == 1:
        idx_pool.append(i)
    else:
        other_pool.append(i)
random.shuffle(other_pool) 
idx_pool += other_pool[:len(idx_pool)]
np.random.shuffle(idx_pool)

x_tmp = [x_train[i] for i in idx_pool]
y_tmp = [y_train[i] for i in idx_pool]
x_train = np.array(x_tmp)
y_train = np.array(y_tmp)

idx_pool = []
other_pool = []
for i in range(len(y_test)):
    if y_test[i][0] == 1:
        idx_pool.append(i)
    else:
        other_pool.append(i)
random.shuffle(other_pool) 
idx_pool += other_pool[:len(idx_pool)]
np.random.shuffle(idx_pool)

x_tmp2 = [x_test[i] for i in idx_pool]
y_tmp2 = [y_test[i] for i in idx_pool]
x_test = np.array(x_tmp2)
y_test = np.array(y_tmp2)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

filepath = '/data/hibbslab/jyang/outputs/' + this_genre + 'Weights.{epoch:3d}-{val_loss:.2f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=True, 
                                save_weights_only=True, 
                                mode='auto', 
                                period=1)

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
opt = keras.optimizers.rmsprop(lr=1e-7, decay=1e-10)
nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=K.epsilon(), schedule_decay=0.004)
# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=nadam,
              metrics=['accuracy'])

print(np.amax(x_test))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.amax(x_train)
x_test /= np.amax(x_test)

print('Not using data augmentation.')

model.fit(x_train, y_train, batch_size=batch_size, 
	epochs=epochs,
	validation_data=(x_test, y_test), shuffle=True)

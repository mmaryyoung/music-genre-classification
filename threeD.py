from __future__ import print_function
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

K.set_image_dim_ordering('th')

batch_size = 32
num_classes = 1
epochs = 200
data_augmentation = True

# 44100 samples per second
# 441 samples per section
# 100 section per second

def parseAudio(fName):
	rate, data = wav.read(fName)
	leftChannel = data[:,0]
	rightChannel = data[:,1]
	return (leftChannel, rightChannel)
	

# get one second
def parseOneSec(song, secIndex):
	startPoint = secIndex*44100
	endPoint = (secIndex+1)*44100
	leftChannel = song[0]
	rightChannel = song[1]
	grid1 = []
	grid2 = []
	grid3 = []
	for i in range(startPoint,endPoint,441):
		sectionOut1 = np.fft.fft(leftChannel[i:(i+441)])
		sectionAmp1 = map(lambda x: x.real * x.real + x.imag * x.imag, sectionOut1)
		grid1.append(sectionAmp1)
		sectionOut2 = np.fft.fft(rightChannel[i:(i+441)])
		sectionAmp2 = map(lambda x: x.real * x.real + x.imag * x.imag, sectionOut2)
		grid2.append(sectionAmp2)
		sectionAmp3 = [x+y for x, y in zip(sectionAmp1, sectionAmp2)]
		grid3.append(sectionAmp3)

	n = [grid1, grid2, grid3]
	return n

def mapOneSec(grid):
	plt.imshow(grid, cmap='hot')
	plt.show()


song1 = parseAudio('Redbone.wav')
song2 = parseAudio('Have Some Love.wav')



# get first minute
firstMin1 = []
for i in range(0,60):
	firstMin1.append(parseOneSec(song1, i))

firstMin2 = []
for i in range(0,60):
	firstMin2.append(parseOneSec(song2, i))

test1 = []
for i in range(60,70):
	test1.append(parseOneSec(song1, i))

test2 = []
for i in range(60,70):
	test2.append(parseOneSec(song2, i))

x_train = np.asarray(firstMin1 + firstMin2)
y_train = np.asarray([1]*60 + [0]*60)
x_test = np.asarray(test1 + test2)
y_test = np.asarray([1]*10 + [0]*10)

print(x_train.shape[1:])

# Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

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
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

print('Not using data augmentation.')

print("x_train: "+ str(x_train.shape))
print("y_train: "+ str(y_train.shape))
model.fit(x_train, y_train, batch_size=batch_size, 
	epochs=epochs,
	validation_data=(x_test, y_test), shuffle=True)


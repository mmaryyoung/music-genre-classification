# Based on the fact that GTZAN5 was ordered based on song. Every 10 samples are one song. 

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

from keras.models import load_model

model = load_model('cnn3sec.h5')


dataPath = '/data/hibbslab/jyang/tzanetakis/ver5.0/'
x_train = pickle.load(open(dataPath + 'x_train_mel.p', 'rb'))
y_train = pickle.load(open(dataPath + 'y_train_mel.p', 'rb'))
x_test = pickle.load(open(dataPath + 'x_test_mel.p', 'rb'))
y_test = pickle.load(open(dataPath + 'y_test_mel.p', 'rb'))

results = []
for i in range(0,len(x_train)-10, 10):
	correct_count = 0
	for j in range(10):
		prediction = model.predict(x_test[i+j])
		truth = y_test[i+j]
		prediction = prediction.index(max(prediction)) 
		truth = truth.index(max(truth))
		if prediction == truth:
			correct_count += 1
	if correct_count > 5:
		results.append(True)
	else:
		results.append(False)

print "voting accuracy: ", 1.0*results.count(True)/len(results)



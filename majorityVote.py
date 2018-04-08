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

model = load_model('3L-cnn-Nadam.h5')


dataPath = '/data/hibbslab/jyang/tzanetakis/ver5.0/'
x_test = pickle.load(open(dataPath + 'x_holdout_mel.p', 'rb'))
y_test = pickle.load(open(dataPath + 'y_holdout_mel.p', 'rb'))

results = []
predictions = []
truths = []
for i in range(0,len(x_test)-10, 10):
    correct_count = 0
    prediction = model.predict(x_test[i:i+10])
    prediction = [p.tolist().index(max(p)) for p in prediction]
    prediction = max(set(prediction), key=prediction.count)
    predictions.append(prediction)
    truth = y_test[i]
    #assert(y_test[i].all() == y_test[i+1].all() and y_test[i+1].all() == y_test[i+9].all())
    truth = truth.tolist().index(max(truth))
    truths.append(truth)
    #correct = [truth[i] == prediction[i] for i in range(10)].count(True)
    if truth == prediction:
        results.append(True)
    else:
        results.append(False)


print "voting accuracy: ", 1.0*results.count(True)/len(results)



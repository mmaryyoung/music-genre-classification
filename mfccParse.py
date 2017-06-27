from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

import librosa
import librosa.display

import os
import pickle

# TODO:
# Load wav file
# Get MFCC Output
# Slice into appropriate chunks
# Save for training


x_train = []
y_train = []
x_test = []
y_test = []
x_holdout = []
y_holdout = []


def parseAudio(genreIndex, songIndex, fName):
	y, sr = librosa.load(fName)
	# CHANGE HERE
	audioLength = 60*sr

	# Leave the center if longer than one minute
	if y.shape[0] > audioLength:
		extraLength = int((y.shape[0] - audioLength)/2)
		y = y[extraLength : audioLength + extraLength]
	else:
		audioLength = y.shape[0]
	
	logam = librosa.logamplitude
	melgram = librosa.feature.melspectrogram
	longgrid = logam(melgram(y=y, sr=22050,n_fft=1024, n_mels=128),ref_power=1.0)
	longgrid = np.expand_dims(longgrid, axis=3)
	# chunks = map(lambda col: [col[x:x+128] for x in range(0, len(col)-128, 128)], longgrid)
	chunks = [longgrid[:, x:x+128] for x in range(0, len(longgrid[0])-128,128)]
	chunks = np.asarray(chunks)
	# print(chunks.shape)

	oneLabel = [0]*10
	oneLabel[genreIndex] = 1

	if songIndex < 5:
		[x_train.append(x) for x in chunks]
		[y_train.append(x) for x in [oneLabel]*len(chunks)]
	elif songIndex > 7:
		[x_holdout.append(x) for x in chunks]
		[y_holdout.append(x) for x in [oneLabel]*len(chunks)]
	else:
		[x_test.append(x) for x in chunks]
		[y_test.append(x) for x in [oneLabel]*len(chunks)]
	
	
	print(fName)
	print('x_train: ', len(x_train), len(x_train[0]), len(x_train[0][0]))
	print('y_train: ', len(y_train))
	print('x_test: ', len(x_test), len(x_test[0]), len(x_test[0][0]))
	print('y_test: ', len(y_test))


#parseAudio(0,0,'stupid cupid.wav')
gid = 0
for root, dirs, files in os.walk('/data/hibbslab/jyang/genres'):
	if '_pickle' not in root and '_img' not in root:
		sid = 0
		print(root, gid)
		for name in files:
			if 'au' in name:
				parseAudio(gid, sid, root + '/' + name)
				sid +=1
		if sid != 0:
			gid +=1

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


pickle.dump(x_train, open('/data/hibbslab/jyang/tzanetakis/x_train_mel.p', 'wb'))
pickle.dump(y_train, open('/data/hibbslab/jyang/tzanetakis/y_train_mel.p', 'wb'))
pickle.dump(x_test, open('/data/hibbslab/jyang/tzanetakis/x_test_mel.p', 'wb'))
pickle.dump(y_test, open('/data/hibbslab/jyang/tzanetakis/y_test_mel.p', 'wb'))

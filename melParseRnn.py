from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

import librosa
import librosa.display

import os
import pickle


x_train = []
y_train = []
x_test = []
y_test = []
x_holdout = []
y_holdout = []

# TODO
# Target output shape: [batch_size, time_steps, features] 

batch_size = 10000 #how many samples
time_steps = 1292/12 # per 5 second
features = 128

def parseAudio(genreIndex, songIndex, fName):
	y, sr = librosa.load(fName)
	# CHANGE HERE
	audioLength = 30*sr

	# Leave the center if longer than one minute
	if y.shape[0] > audioLength:
		extraLength = int((y.shape[0] - audioLength)/2)
		y = y[extraLength : audioLength + extraLength]
	else:
		audioLength = y.shape[0]
	
	logam = librosa.logamplitude
	melgram = librosa.feature.melspectrogram
	longgrid = logam(melgram(y=y, sr=22050,n_fft=1024, n_mels=features),ref_power=1.0)
	chunks = np.swapaxes(longgrid, 0, 1)
	#print(chunks.shape)
	#Mel for RNN (1292, 128)
	if chunks.shape != (1292, 128):
		print(chunks.shape)
		if chunks.shape[0]>1292:
			chunks = chunks[:1292]
		elif chunks.shape[0] < 1292:
			diff = 1292 - chunks.shape[0]
			chunks = np.insert(chunks, 0, [[0.0]*128]*diff, axis=0)

	oneLabel = [0]*10
	oneLabel[genreIndex] = 1

	#CHANGE HERE FOR HM DS
	#CHANGE HERE FOR GTZAN
	if songIndex < 50:
		x_train.append(chunks)
		y_train.append(oneLabel)
		# print('x_train: ', np.asarray(x_train).shape)
		# print('y_train: ', np.asarray(y_train).shape)
	elif songIndex > 70:
		x_holdout.append(chunks)
		y_holdout.append(oneLabel)
		# print('x_holdout: ', len(x_holdout), len(x_holdout[0]), len(x_holdout[0][0]))
		# print('y_holdout: ', len(y_holdout))
	else:
		x_test.append(chunks)
		y_test.append(oneLabel)
		# print('x_test: ', len(x_test), len(x_test[0]), len(x_test[0][0]))
		# print('y_test: ', len(y_test))
	
	


#parseAudio(0,0,'stupid cupid.wav')
gid = 0
# CHANGE PATH
sourcePath = "/data/hibbslab/eherbert/millionSong/"
for root, dirs, files in os.walk(sourcePath):
	if '_pickle' not in root and '_img' not in root:
		sid = 0
		print(root, gid)
		for name in files:
			# CHANGE HERE FOR FILE TYPE
			if 'mp3' in name or 'wav' in name or 'au' in name:
				parseAudio(gid, sid, root + '/' + name)
				sid +=1
		if sid != 0:
			gid +=1


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_holdout = np.asarray(x_holdout)
y_holdout = np.asarray(y_holdout)
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)
destPath = "/data/hibbslab/jyang/MSD/"
pickle.dump(x_train, open(destPath + 'x_train_mel.p', 'wb'))
pickle.dump(y_train, open(destPath + 'y_train_mel.p', 'wb'))
pickle.dump(x_test, open(destPath + 'x_test_mel.p', 'wb'))
pickle.dump(y_test, open(destPath + 'y_test_mel.p', 'wb'))
pickle.dump(x_holdout, open(destPath + 'x_holdout_mel.p', 'wb'))
pickle.dump(y_holdout, open(destPath + 'y_holdout_mel.p', 'wb'))

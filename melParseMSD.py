from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

import librosa
import librosa.display

import os
import pickle


# TODO
# Target output shape: [batch_size, time_steps, features] 

batch_size = 10000 #how many samples
time_steps = 1292/12 # per 5 second
features = 128

def parseAudio(fName):
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
	longgrid = logam(melgram(y=y, sr=sr,n_fft=1024, n_mels=features),ref_power=1.0)
	chunks = np.swapaxes(longgrid, 0, 1)
	#print(chunks.shape)
	#Mel for RNN (1292, 128)
	if chunks.shape != (1292, 128):
		if chunks.shape[0]>1292:
			chunks = chunks[:1292]
		elif chunks.shape[0] < 1292:
			print(fName, "\t", chunks.shape)
			diff = 1292 - chunks.shape[0]
			chunks = np.insert(chunks, 0, [[0.0]*128]*diff, axis=0)




gid = 0
# CHANGE PATH
sourceRoot = "/data/hibbslab/eherbert/millionSong/"
destRoot = "/data/hibbslab/jyang/msd/ver2.0/"
for root, dirs, files in os.walk(sourceRoot):
	for name in files:
		# CHANGE HERE FOR FILE TYPE
		if '.mp3' in name:
			parsed = parseAudio(root + '/' + name)
			destPath = destRoot + name[3] + "/" + name[4] + "/" + name[5] + "/"
			#pickle.dump(parsed, open(destPath + os.path.splitext(name)[0] + ".p", "wb"))

#pickle.dump(x_train, open(destPath + 'x_train_mel.p', 'wb'))
#pickle.dump(y_train, open(destPath + 'y_train_mel.p', 'wb'))
#pickle.dump(x_test, open(destPath + 'x_test_mel.p', 'wb'))
#pickle.dump(y_test, open(destPath + 'y_test_mel.p', 'wb'))
#pickle.dump(x_holdout, open(destPath + 'x_holdout_mel.p', 'wb'))
#pickle.dump(y_holdout, open(destPath + 'y_holdout_mel.p', 'wb'))

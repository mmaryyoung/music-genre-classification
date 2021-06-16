"""
Turns GTZAN raw wave samples into melspectrogram samples.

The input of this file is the result of the .audio_to_grouped_raw_waves.py file. 
The raw waves were already grouped into training, testing, and validation. The job of this file is to read
those files and turn them into one .npz file containing all three.
"""

import numpy as np
import librosa as lb 
import matplotlib.pyplot as plt

RAW_WAVE_SOURCE_DIR = '/Users/maryyang/Learning/music-genre-classification/crnn/gtzan/10secs/'
MELSPECTS_DEST_PATH = '/Users/maryyang/Learning/music-genre-classification/crnn/gtzan/10secs/melspects.npz'

SR = 22050
N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64 

def log_melspectrogram(data, log=True, plot=False, num='', genre=""):

	melspec = lb.feature.melspectrogram(y=data, hop_length = HOP_LENGTH, n_fft = N_FFT, n_mels = N_MELS)

	if log:
		melspec = lb.power_to_db(melspec**2)

	if plot:
		melspec = melspec[np.newaxis, :]
		plt.imshow(melspec.reshape((melspec.shape[1],melspec.shape[2])))
		plt.savefig('melspec'+str(num)+'_'+str(genre)+'.png')

	return melspec

def batch_log_melspectrogram(data_list, log=True, plot=False):
	melspecs = np.asarray([log_melspectrogram(data_list[i],log=log,plot=plot) for i in range(len(data_list))])
	# This line may or may not be neccesary idk.
	# melspecs = melspecs.reshape(melspecs.shape[0], melspecs.shape[1], melspecs.shape[2], 1)
	return melspecs

training = np.load(RAW_WAVE_SOURCE_DIR + 'gtzan_tr.npy')
data_tr = np.delete(training, -1, 1)
label_tr = training[:,-1]
spects_tr = batch_log_melspectrogram(data_tr)

testing = np.load(RAW_WAVE_SOURCE_DIR + 'gtzan_te.npy')
data_te = np.delete(testing, -1, 1)
label_te = testing[:,-1]
spects_te = batch_log_melspectrogram(data_te)

crossval = np.load(RAW_WAVE_SOURCE_DIR + 'gtzan_cv.npy')
data_cv = np.delete(crossval, -1, 1)
label_cv = crossval[:,-1]
spects_cv = batch_log_melspectrogram(data_cv)

print("Saving to %s" % MELSPECTS_DEST_PATH)
np.savez(MELSPECTS_DEST_PATH, x_tr=spects_tr, y_tr=label_tr, x_te=spects_te, y_te=label_te, x_cv=spects_cv, y_cv=label_cv)
print("Done")


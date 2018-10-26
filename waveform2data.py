import numpy as np 
import pickle
import os
import librosa
import librosa.display



source_path = "/data/hibbslab/jyang/genres/"
dest_path = "/data/hibbslab/jyang/tzanetakis/ver8.0/"

source_rate = 22050
song_length = 30
seconds_per_sample = 5
samples_per_song = song_length/seconds_per_sample
num_genres = 10
songs_per_genre = 100

x_train = np.empty([num_genres*50*samples_per_song, source_rate*seconds_per_sample])
y_train = np.empty([num_genres*50*samples_per_song, num_genres])
x_test = np.empty([num_genres*20*samples_per_song, source_rate*seconds_per_sample])
y_test = np.empty([num_genres*20*samples_per_song, num_genres])
x_holdout = np.empty([num_genres*30*samples_per_song, source_rate*seconds_per_sample])
y_holdout = np.empty([num_genres*30*samples_per_song, num_genres])


def parseAudio(genreIndex, songIndex, fName):
	y, sr = librosa.load(fName)
	# print('loaded ', fName)
	# CHANGE HERE TO 60 FOR MSD
	# CHANGE HERE TO 30 FOR GTZAN
	audioLength = 30*sr

	# Leave the center if longer than one minute
	if y.shape[0] > audioLength:
		extraLength = int((y.shape[0] - audioLength)/2)
		y = y[extraLength : audioLength + extraLength]
	else:
 		y = np.concatenate((y, np.zeros([audioLength - y.shape[0]]) ))
	chunks = y.reshape([samples_per_song, -1])
	labels = np.zeros([samples_per_song, num_genres])
	labels[:, genreIndex] = 1

	#CHANGE HERE FOR HM DS
	#CHANGE HERE FOR GTZAN
	if songIndex < 50:
		x_train[songIndex*samples_per_song:(songIndex+1)*samples_per_song] = chunks
		y_train[songIndex*samples_per_song:(songIndex+1)*samples_per_song] = labels 
		# [x_train.append(x) for x in chunks]
		# [y_train.append(x) for x in [oneLabel]*len(chunks)]
		# print('x_train: ', len(x_train), len(x_train[0]), len(x_train[0][0]))
		# print('y_train: ', len(y_train))
	elif songIndex > 70:
		x_holdout[songIndex*samples_per_song:(songIndex+1)*samples_per_song] = chunks
		y_holdout[songIndex*samples_per_song:(songIndex+1)*samples_per_song] = labels 
		# print('x_holdout: ', len(x_holdout), len(x_holdout[0]), len(x_holdout[0][0]))
		# print('y_holdout: ', len(y_holdout))
	else:
		x_test[songIndex*samples_per_song:(songIndex+1)*samples_per_song] = chunks
		y_test[songIndex*samples_per_song:(songIndex+1)*samples_per_song] = labels 
		# print('x_test: ', len(x_test), len(x_test[0]), len(x_test[0][0]))
		# print('y_test: ', len(y_test))
	
	

#parseAudio(0,0,'stupid cupid.wav')
# CHANGE PATH
gid = 0
for root, dirs, files in os.walk(source_path):
	if '_pickle' not in root and '_img' not in root:
		sid = 0
		print(root, gid)
		for name in files:
			# CHANGE HERE FOR FILE TYPE
			if 'wav' in name or 'au' in name:
				parseAudio(gid, sid, root + '/' + name)
				sid +=1
		if sid != 0:
			gid +=1


# Normalize the data
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
x_holdout = np.asarray(x_holdout)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_holdout = np.asarray(y_holdout)

print("x_train: " + str(x_train.shape))
print("y_train: " + str(y_train.shape))
print("x_test: " + str(x_test.shape))
print("y_test: " + str(y_test.shape))
print("x_holdout: " + str(x_holdout.shape))
print("y_holdout: " + str(y_holdout.shape))


pickle.dump(x_train, open(dest_path+'x_train_mel.p', 'wb'))
pickle.dump(y_train, open(dest_path+'y_train_mel.p', 'wb'))
pickle.dump(x_test, open(dest_path+'x_test_mel.p', 'wb'))
pickle.dump(y_test, open(dest_path+'y_test_mel.p', 'wb'))
pickle.dump(x_holdout, open(dest_path+'x_holdout_mel.p', 'wb'))
pickle.dump(y_holdout, open(dest_path+'y_holdout_mel.p', 'wb'))

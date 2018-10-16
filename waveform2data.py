import numpy as np 
import pickle
import os
import librosa
import librosa.display



source_path = "/Users/mac/Desktop/genres/"
dest_path = "/Users/mac/Desktop/waveformData/"

x_train = []
y_train = []
x_test = []
y_test = []
x_holdout = []
y_holdout = []


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
		audioLength = y.shape[0]
	chunks = y
	oneLabel = [0]*10
	oneLabel[genreIndex] = 1

	#CHANGE HERE FOR HM DS
	#CHANGE HERE FOR GTZAN
	if songIndex < 50:
		[x_train.append(x) for x in chunks]
		[y_train.append(x) for x in [oneLabel]*len(chunks)]
		# print('x_train: ', len(x_train), len(x_train[0]), len(x_train[0][0]))
		# print('y_train: ', len(y_train))
	elif songIndex > 70:
		[x_holdout.append(x) for x in chunks]
		[y_holdout.append(x) for x in [oneLabel]*len(chunks)]
		# print('x_holdout: ', len(x_holdout), len(x_holdout[0]), len(x_holdout[0][0]))
		# print('y_holdout: ', len(y_holdout))
	else:
		[x_test.append(x) for x in chunks]
		[y_test.append(x) for x in [oneLabel]*len(chunks)]
		# print('x_test: ', len(x_test), len(x_test[0]), len(x_test[0][0]))
		# print('y_test: ', len(y_test))
	
	

#parseAudio(0,0,'stupid cupid.wav')
gid = 0
# CHANGE PATH
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
x_train = np.array(x_train).astype('float32')
x_test = np.array(x_test).astype('float32')
x_holdout = np.array(x_holdout).astype('float32')
x_train /= np.amax(x_train)
x_test /= np.amax(x_test)
x_holdout /= np.amax(x_holdout)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_holdout = np.array(y_holdout)

print("x_train: " + x_train.shape)
print("y_train: " + y_train.shape)
print("x_test: " + x_test.shape)
print("y_test: " + y_test.shape)
print("x_holdout: " + x_holdout.shape)
print("y_holdout: " + y_holdout.shape)


pickle.dump(x_train, open(dest_path+'x_train_mel.p', 'wb'))
pickle.dump(y_train, open(dest_path+'y_train_mel.p', 'wb'))
pickle.dump(x_test, open(dest_path+'x_test_mel.p', 'wb'))
pickle.dump(y_test, open(dest_path+'y_test_mel.p', 'wb'))
pickle.dump(x_holdout, open(dest_path+'x_holdout_mel.p', 'wb'))
pickle.dump(y_holdout, open(dest_path+'y_holdout_mel.p', 'wb'))

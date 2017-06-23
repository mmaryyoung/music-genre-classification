from __future__ import print_function
import sunau
import wave
import os
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from os.path import join, getsize, splitext


# Assumes mone
# Get Audio Params
sampleAudio = sunau.open('genres/rock/rock.00000.au', 'r')
audioRate = sampleAudio.getframerate()
audioFrames = sampleAudio.getnframes()
audioLength = audioFrames/22000
#audioLength = 1

x_train = []
y_train = []
x_test = []
y_test = []
x_holdout = []
y_holdout = []


def heatMap(grid, fName = 'defaultName'):
	grid = map(lambda row: map(lambda x: x[0], row), grid)
	plt.imshow(grid, cmap='hot')
	plt.axis('off')
	plt.savefig('./' + fName, dpi=1200, bbox_inches='tight')
	#plt.show()


def parseAudio(genreIndex, songIndex, fName):
	audio = sampleAudio
	if '.au' in fName:
		audio = sunau.open(fName, 'rb')
	elif '.wav' in fName:
		audio = wave.open(fName, 'rb')
	else:
		print('Incorrect file type in file: ', fName)
		return
	audioRate = audio.getframerate()
	audioFrames = audio.getnframes()
	audioLength = audioFrames/audioRate
	# CHANGE HERE
	audioLength = 60

	oneLabel = [0]*10
	oneLabel[genreIndex] = 1

	# Every 0.01 sec
	frequencyBins = audioRate/100
	longGrid = []
	for i in range(0, audioLength):
		grid = []
		for j in range(0, 100):
			data = audio.readframes(frequencyBins)
			if audio.getnchannels() > 1:
				data = data + audio.readframes(frequencyBins)
				data = np.fromstring(data,dtype=np.int16)
				# Left Channel Possibly
				data = data[::2]
			else:
				data = np.fromstring(data,dtype=np.int16)

			sectionOut = np.fft.fft(data)
			sectionAmp = map(lambda x: [x.real * x.real + x.imag * x.imag], sectionOut)
			# CHANGE HERE
			sectionAmp = sectionAmp[:110]
			grid.append(sectionAmp)
		if songIndex < 80:
			x_train.append(grid)
			y_train.append(oneLabel)
		else:
			x_test.append(grid)
			y_test.append(oneLabel)
		longGrid += grid
	
	#heatMap(longGrid)
	return longGrid
	audio.close()
	
	

# gid = 0;
# for root, dirs, files in os.walk('genres'):
# 	if(root != 'genres' and root != 'genres/edm'):
# 		sid = 0
# 		for name in files:
# 			if '.au' in name:
# 				parseAudio(gid, sid, root + '/' +name)
# 				sid += 1
# 		print('id for ', root, ' is ', gid)
# 		gid+=1

# x_train = np.asarray(x_train)
# y_train = np.asarray(y_train)
# x_test = np.asarray(x_test)
# y_test = np.asarray(y_test)

# pickle.dump(x_train, open('/data/hibbslab/jyang/x_train.p', 'wb'))
# pickle.dump(y_train, open('/data/hibbslab/jyang/y_train.p', 'wb'))
# pickle.dump(x_test, open('/data/hibbslab/jyang/x_test.p', 'wb'))
# pickle.dump(y_test, open('/data/hibbslab/jyang/y_test.p', 'wb'))


gid = 1
sid = 1
grid = parseAudio(gid, sid, 'The Kinks - You really Got me (live).wav')
heatMap(grid, 'You Really Got Me')

# for root, dirs, files in os.walk('edm'):
# 	sid = 0
# 	for name in files:
# 		grid = parseAudio(gid, sid, root + '/' + name)
# 		noExt = splitext(name)[0]
# 		noExt = re.sub('[.]+', '', noExt)
# 		print(noExt)
# 		heatMap(grid, noExt)
# 		grid = np.asarray(grid)
# 		pickle.dump(grid, open('edm_pickle/'+noExt+'.p', 'wb'))
# 		sid +=1

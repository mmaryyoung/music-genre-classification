from __future__ import print_function
import sunau
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from os.path import join, getsize


# Assumes mone
# Get Audio Params
sampleAudio = sunau.open('genres/blues/blues.00000.au', 'r')
audioRate = sampleAudio.getframerate()
audioFrames = sampleAudio.getnframes()
audioLength = audioFrames/22000
#audioLength = 1

x_train = []
y_train = []
x_test = []
y_test = []
x_holdback = []
y_holdback = []

def heatMap(grid):
	plt.imshow(grid, cmap='hot')
	plt.show()


def parseAudio(genreIndex, songIndex, fName):
	audio = sunau.open(fName, 'r')

	oneLabel = [0]*10
	oneLabel[genreIndex] = 1

	# Every 0.01 sec
	frequencyBins = audioRate/100
	for i in range(0, audioLength):
		grid = []
		for j in range(0, 100):
			data = audio.readframes(frequencyBins)
			data = np.fromstring(data,dtype=np.int16)
			sectionOut = np.fft.fft(data)
			sectionAmp = map(lambda x: [x.real * x.real + x.imag * x.imag,0,0], sectionOut)
			
			sectionAmp = sectionAmp[:110]
			grid.append(sectionAmp)
		# CHANGE HERE
		#heatMap(grid)
		if songIndex < 80:
			x_train.append(grid)
			y_train.append(oneLabel)
		else:
			x_test.append(grid)
			y_test.append(oneLabel)
	
	#labels = oneLabel*audioLength
	#y_train.append(labels)
	audio.close()
	
	

gid = 0;
for root, dirs, files in os.walk('genres'):
	if(root != 'genres'):
		sid = 0
		for name in files:
			if '.au' in name:
				parseAudio(gid, sid, root + '/' +name)
				sid += 1
			#print(gid, root + '/' + name)
		print('id for ', root, ' is ', gid)
		gid+=1

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

half_length = len(x_train)/2
x_train_1 = x_train[:half_length]
x_train_2 = x_train[half_length:]

pickle.dump(x_train_1, open('/data/hibbslab/jyang/3Channels/x_train_1.p', 'wb'))
pickle.dump(x_train_2, open('/data/hibbslab/jyang/3Channels/x_train_2.p', 'wb'))
pickle.dump(y_train, open('/data/hibbslab/jyang/3Channels/y_train.p', 'wb'))
pickle.dump(x_test, open('/data/hibbslab/jyang/3Channels/x_test.p', 'wb'))
pickle.dump(y_test, open('/data/hibbslab/jyang/3Channels/y_test.p', 'wb'))





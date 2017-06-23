from __future__ import print_function
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import wave

from matplotlib.patches import Rectangle

batch_size = 32
num_classes = 1
epochs = 200
data_augmentation = True

sampleAudio = wave.open('Redbone.wav', 'rb')
audioRate = sampleAudio.getframerate()

sampleChanneltmp = sampleAudio.readframes(audioRate)
sampleChannel = np.fromstring(sampleChanneltmp, dtype = np.int16)
sampleChannel = sampleChannel[:audioRate]

parseOneSec()




rate, data = wav.read('Redbone.wav')
leftChannel = data[:,0]

# 44100 samples per second
# 441 samples per section
# 100 section per second

def parseAudio(fName):
	rate, data = wav.read(fName)
	leftChannel = data[:,0]
	return leftChannel
	

# get one second
def parseOneSec(leftChannel, secIndex):
	startPoint = secIndex*44100
	endPoint = (secIndex+1)*44100
	grid = []
	for i in range(startPoint,endPoint,441):
		sectionOut = np.fft.fft(leftChannel[i:(i+441)])
		sectionAmp = map(lambda x: x.real * x.real + x.imag * x.imag, sectionOut)
		grid.append(sectionAmp)
		# frequencies = np.fft.fftfreq(len(sectionOut), 1.0/rate)
		# plt.plot(frequencies, sectionAmp)
		# plt.show()

	# n = np.zeros((1,100,441))
	# n[0][:][:] = np.asarray(grid)
	# return n
	return grid

def mapOneSec(grid):
	plt.figure()
	currentAxis = plt.gca()
	for x in range(0,99):
		for y in range(0, 99):
			theColor = '%.17f'%(1-grid[x][y])
			currentAxis.add_patch(Rectangle((0.01*x,0.01*y),0.01,0.01,alpha=1,facecolor = theColor))
	plt.show()

def heatMap(grid):
	plt.imshow(grid, cmap='hot')
	plt.show()


song1 = parseAudio('Redbone.wav')
song2 = parseAudio('Have Some Love.wav')

grid = parseOneSec(song1, 25)
maxi = max(max(grid))
mini = min(min(grid))
grid = map(lambda row: map(lambda x: (x-mini)/(maxi-mini), row), grid)

# print(max(max(grid)))
# print(min(min(grid)))
# print(max(grid[30][:]))

#heatMap(np.asarray(grid))
mapOneSec(np.asarray(grid))


# a = np.random.random((16,16))
# mapOneSec(a)

from __future__ import print_function
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import wave
import sunau

from matplotlib.patches import Rectangle

def heatMap(grid):
	plt.imshow(grid, cmap='hot')
	plt.show()

sampleAudio = wave.open('Redbone.wav', 'rb')
audioRate = sampleAudio.getframerate()

sampleChanneltmp = sampleAudio.readframes(audioRate)
sampleChannel = np.fromstring(sampleChanneltmp, dtype = np.int16)
sampleChannel = sampleChannel[:audioRate]

def parseAudio(fName):
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

	# Every 0.01 sec
	frequencyBins = audioRate/100
	for i in range(0, audioLength):
		grid = []
		for j in range(0, 100):
			data = audio.readframes(frequencyBins)
			data = np.fromstring(data,dtype=np.int16)
			sectionOut = np.fft.fft(data)
			sectionAmp = map(lambda x: x.real * x.real + x.imag * x.imag, sectionOut)
			
			sectionAmp = sectionAmp[:110]
			grid.append(sectionAmp)
		# CHANGE HERE
		heatMap(grid)
	
	audio.close()

parseAudio('Redbone.wav')
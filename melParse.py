from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
import IPython.display
import librosa
import librosa.display
import os

def parseAudio(fPath):
	audio_path = fPath

	y, sr = librosa.load(audio_path, duration=30)
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
	log_S = librosa.logamplitude(S, ref_power=np.max)
	plt.figure(figsize=(12,4))
	librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
	plt.title('mel power spectrogram')
	plt.colorbar(format='%+02.0f dB')
	plt.tight_layout()
	plt.show()
	# tmpList = fPath.split('/')
	# fName = tmpList[len(tmpList)-1]
	# fName = os.path.splitext(fName)[0]
	# gName = tmpList[len(tmpList)-2]
	# print(fName)
	# plt.savefig('/Users/mac/Desktop/' + gName + '/' + fName + '.png', dpi = 1200, )


parseAudio('/Users/mac/Desktop/Bob Marley - Is This Love.wav')
# for root, dirs, files in os.walk('/Users/mac/Desktop/Homemade Dataset/hiphop'):
# 	if '_pickle' not in root and '_img' not in root:
# 		print(root)
# 		for name in files:
# 			if 'wav' in name:
# 				parseAudio(root + '/' + name)
import tables
import os
import os.path
import numpy as np
import pickle
import librosa
import sys
#from melParseMSD import parseAudio

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
		print(chunks.shape)
		if chunks.shape[0]>1292:
			chunks = chunks[:1292]
		elif chunks.shape[0] < 1292:
			diff = 1292 - chunks.shape[0]
			chunks = np.insert(chunks, 0, [[0.0]*128]*diff, axis=0)



# I MANUALLY WENT THROUGH THE TAGGER FILE AND GOT THIS
tags = ['Reggae', 'Jazz', 'RnB', 'Metal', 'Pop', 'Punk', 'Country', 'Latin', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk']
# THERE ARE APPROXIMATELY 191,401 TRACKS
# TENTATIVELY 1 - 95,700 TRAIN, 95,701 - 153,120 TEST, 153121 - END HOLDOUT
tagsPath = '/home/jyang/summer2017/msd_tagtraum_cd2c.cls'
#tagsFile = open(tagsPath, 'r')
sourceRoot = "/data/hibbslab/eherbert/millionSong/mp3/"

with open(tagsPath, 'r') as tagsFile:
	for line in tagsFile:
		if line[:2] == 'TR':
			trackID, genre = line.split('\t')
			genreID = tags.index(genre[:-1])
			lev1 = trackID[2]
			lev2 = trackID[3]
			lev3 = trackID[4]
			genreRoot = '/data/hibbslab/jyang/msd/genres/' + genre[:-1] + '/'
			sourcePath = sourceRoot + lev1 + '/' + lev2 + '/' + lev3 + '/' + trackID + '.mp3'
			destPath = genreRoot + trackID + ".p"

			if not os.path.isdir(genreRoot):
				os.makedirs(genreRoot)
			
			# If the file was not parsed and we have a mp3 version
			if not os.path.isfile(destPath) and  os.path.isfile(sourcePath):
				parsed = parseAudio(sourcePath)
				pickle.dump(parsed, open(destPath, "wb"))
				print("parsed file " + trackID + " as " + genre[:-1]) 
                sys.stdout.flush()



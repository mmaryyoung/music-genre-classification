import tables
import os
import os.path
import numpy as np
import librosa
from melParseMSD import parseAudio


# I MANUALLY WENT THROUGH THE TAGGER FILE AND GOT THIS
tags = ['Reggae', 'Jazz', 'RnB', 'Metal', 'Pop', 'Punk', 'Country', 'Latin', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk']
# THERE ARE APPROXIMATELY 191,401 TRACKS
# TENTATIVELY 1 - 95,700 TRAIN, 95,701 - 153,120 TEST, 153121 - END HOLDOUT
tagsPath = '/home/jyang/summer2017/msd_tagtraum_cd2c.cls'
tagsFile = open(tagsPath, 'r')
sourceRoot = "/data/hibbslab/eherbert/millionSong/"


for line in tagsFile:
	if line[:2] == 'TR':
		trackID, genre = line.split('\t')
		genreID = tags.index(genre[:-1])
		lev1 = trackID[2]
		lev2 = trackID[3]
		lev3 = trackID[4]

		genreRoot = '/data/hibbslab/jyang/msd/genres/' + genre[:-1] + '/'
		sourcePath = sourceRoot + sourceRoot + lev1 + '/' + lev2 + '/' + lev3 + '/' + trackID + '.mp3'

		if not os.path.isdir(genreRoot):
			os.mkdirs(genreRoot)


		# If the file was not parse and we have a mp3 version
		if not os.path.isfile(genreRoot + trackID) and  os.path.isfile(sourcePath):
			parsed = parseAudio(sourcePath)
			destPath = genreRoot + trackID + ".p"
			pickle.dump(parsed, open(destPath, "wb"))
			print "parsed file " + trackID + " as " + genre[:-1]




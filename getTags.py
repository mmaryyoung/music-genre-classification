import tables
import os
import numpy as np
from klepto.archives import *
import getters

x_train = []
y_train = []
x_test = []
y_test = []
x_holdout = []
y_holdout = []


# I MANUALLY WENT THROUGH THE TAGGER FILE AND GOT THIS
tags = ['Reggae', 'Jazz', 'RnB', 'Metal', 'Pop', 'Punk', 'Country', 'Latin', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk']
# THERE ARE APPROXIMATELY 191,401 TRACKS
# TENTATIVELY 1 - 95,700 TRAIN, 95,701 - 153,120 TEST, 153121 - END HOLDOUT
tagsPath = '/home/jyang/summer2017/msd_tagtraum_cd2c.cls'
tagsFile = open(tagsPath, 'r')
dataRoot = '/data/hibbslab/data/millionSong/'

rover = 0
for line in tagsFile:
	if line[:2] == 'TR':
		trackID, genre = line.split('\t')
		genreID = tags.index(genre[:-1])
		lev1 = trackID[2]
		lev2 = trackID[3]
		lev3 = trackID[4]
		try:
			h5 = tables.open_file(dataRoot + lev1 +'/' + lev2 + '/' + lev3 + '/' + trackID + '.h5', mode='r')
			oneLabel = [0]*15
			oneLabel[genreID] = 1
			for i in range(getters.get_num_songs(h5)):
				timbres = getters.get_segments_timbre(h5, i)
				pitches = getters.get_segments_pitches(h5, i)
				combo = np.concatenate((timbres,pitches),axis=1)
				chunks = [combo[x:x+50] for x in range(0, len(combo)-50,50)]
				chunks = np.expand_dims(chunks, axis=3)
				#print chunks.shape
				if rover < 95700:
					[x_train.append(x) for x in chunks]
					[y_train.append(x) for x in [oneLabel]*len(chunks)]
				elif rover < 153120:
					[x_test.append(x) for x in chunks]
					[y_test.append(x) for x in [oneLabel]*len(chunks)]
				else:
					[x_holdout.append(x) for x in chunks]
					[y_holdout.append(x) for x in [oneLabel]*len(chunks)]
			#print getters.get_title(h5), getters.get_danceability(h5)
			#print 'duration: ', getters.get_duration(h5), 'timbre shape: ', getters.get_segments_timbre(h5).shape, 'pitches shape: ', getters.get_segments_pitches(h5).shape, 'segments start', getters.get_segments_start(h5).shape
			rover +=1
			h5.close()
		except tables.exceptions.HDF5ExtError:
			print "Can't find file", trackID
		except IOError:
			print "Can't find file ", trackID 

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_holdout = np.asarray(x_holdout)
y_holdout = np.asarray(y_holdout)

print 'x_train shape: ', x_train.shape
print 'y_train shape: ', y_train.shape
print 'x_test shape: ', x_test.shape
print 'y_test shape: ', y_test.shape
print 'x_holdout shape: ', x_holdout.shape
print 'y_holdout shape: ', y_holdout.shape

arch = file_archive('/data/hibbslab/jyang/msd/ver1.0/klep')
arch['x_train'] = x_train
arch['y_train'] = y_train
arch['x_test'] = x_test
arch['y_test'] = y_test
arch['x_holdout'] = x_holdout
arch['y_holdout'] = y_holdout

arch.dump()

# pickle.dump(x_holdout, open('/data/hibbslab/jyang/msd/ver1.0/x_holdout.p', 'wb'))
# pickle.dump(y_holdout, open('/data/hibbslab/jyang/msd/ver1.0/y_holdout.p', 'wb'))
# pickle.dump(x_test, open('/data/hibbslab/jyang/msd/ver1.0/x_test.p', 'wb'))
# pickle.dump(y_test, open('/data/hibbslab/jyang/msd/ver1.0/y_test.p', 'wb'))
# pickle.dump(x_train, open('/data/hibbslab/jyang/msd/ver1.0/x_train.p', 'wb'))
# pickle.dump(y_train, open('/data/hibbslab/jyang/msd/ver1.0/y_train.p', 'wb'))



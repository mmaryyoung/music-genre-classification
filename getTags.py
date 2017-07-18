import tables
import os
import pickle
import numpy as np


def get_title(h5,songidx=0):
    """
    Get title from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.title[songidx]

def get_danceability(h5,songidx=0):
#     """
#     Get danceability from a HDF5 song file, by default the first song in it
#     """
    return h5.root.analysis.songs.cols.danceability[songidx]

def get_duration(h5,songidx=0):
    """
    Get duration from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.duration[songidx]

def get_num_songs(h5):
    """
    Return the number of songs contained in this h5 file, i.e. the number of rows
    for all basic informations like name, artist, ...
    """
    return h5.root.metadata.songs.nrows

def get_segments_timbre(h5,songidx=0):
    """
    Get segments timbre array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[songidx]:,:]
    return h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[songidx]:
                                            h5.root.analysis.songs.cols.idx_segments_timbre[songidx+1],:]

def get_segments_pitches(h5,songidx=0):
    """
    Get segments pitches array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_pitches[h5.root.analysis.songs.cols.idx_segments_pitches[songidx]:,:]
    return h5.root.analysis.segments_pitches[h5.root.analysis.songs.cols.idx_segments_pitches[songidx]:
                                             h5.root.analysis.songs.cols.idx_segments_pitches[songidx+1],:]

def get_segments_start(h5,songidx=0):
    """
    Get segments start array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_start[h5.root.analysis.songs.cols.idx_segments_start[songidx]:]
    return h5.root.analysis.segments_start[h5.root.analysis.songs.cols.idx_segments_start[songidx]:
                                           h5.root.analysis.songs.cols.idx_segments_start[songidx+1]]

# max_danceable_song = ''
# max_dance = 0

# for root, dirs, files in os.walk('/Users/mac/Downloads/MillionSongSubset/data'):
# 	for name in files:
# 		if '.h5' in name:
# 			try:
# 				h5 = tables.open_file(root + '/' + name, mode='r')
# 				danceability = get_danceability(h5)
# 				print danceability
# 				
# 				if danceability > max_dance:
# 					max_dance = danceability
# 					max_danceable_song = get_title(h5)
#				h5.close()
# 			except tables.exceptions.HDF5ExtError:
# 				print "Can't find file"

# print 'the most danceable song is ' + max_danceable_song

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
			for i in range(get_num_songs(h5)):
				timbres = get_segments_timbre(h5, i)
				pitches = get_segments_pitches(h5, i)
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
			#print get_title(h5), get_danceability(h5)
			#print 'duration: ', get_duration(h5), 'timbre shape: ', get_segments_timbre(h5).shape, 'pitches shape: ', get_segments_pitches(h5).shape, 'segments start', get_segments_start(h5).shape
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

pickle.dump(x_holdout, open('/data/hibbslab/jyang/msd/ver1.0/x_holdout.p', 'wb'))
pickle.dump(y_holdout, open('/data/hibbslab/jyang/msd/ver1.0/y_holdout.p', 'wb'))
pickle.dump(x_test, open('/data/hibbslab/jyang/msd/ver1.0/x_test.p', 'wb'))
pickle.dump(y_test, open('/data/hibbslab/jyang/msd/ver1.0/y_test.p', 'wb'))
pickle.dump(x_train, open('/data/hibbslab/jyang/msd/ver1.0/x_train.p', 'wb'))
pickle.dump(y_train, open('/data/hibbslab/jyang/msd/ver1.0/y_train.p', 'wb'))



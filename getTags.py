import tables
import os

# def get_title(h5,songidx=0):
#     """
#     Get title from a HDF5 song file, by default the first song in it
#     """
#     return h5.root.metadata.songs.cols.title[songidx]

# def get_danceability(h5,songidx=0):
#     """
#     Get danceability from a HDF5 song file, by default the first song in it
#     """
#     return h5.root.analysis.songs.cols.danceability[songidx]

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


# max_danceable_song = ''
# max_dance = 0

# for root, dirs, files in os.walk('/Users/mac/Downloads/MillionSongSubset/data'):
# 	for name in files:
# 		if '.h5' in name:
# 			try:
# 				h5 = tables.open_file(root + '/' + name, mode='r')
# 				danceability = get_danceability(h5)
# 				print danceability
# 				h5.close()
# 				if danceability > max_dance:
# 					max_dance = danceability
# 					max_danceable_song = get_title(h5)
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
tagsPath = '~/summer2017/msd_tagtraum_cd2c.cls'
tagsFile = open(tagsPath, 'r')
dataRoot = '/data/hibbslab/data/millionsong/'

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
				if rover < 95700:
					x_train.append(get_segments_timbre(h5, i))
					y_train.append(oneLabel)
				elif rover < 153120:
					x_test.append(get_segments_timbre(h5, i))
					y_test.append(oneLabel)
				else:
					x_holdout.append(get_segments_timbre(h5, i))
					y_holdout.append(oneLabel)






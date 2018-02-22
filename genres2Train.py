import sPickle
import pickle
import numpy as np
import os

sourcePath = "/data/hibbslab/jyang/msd/genres/"
destPath = "/data/hibbslab/jyang/msd/ver4.0/"

x_train = []
y_train = []
x_test = []
y_test = []

# Size Display cmd: for i in `ls -1D`; do echo $i; ls -1 $i/* | wc; done

#Blues has 1179 files, we will use this number for all following genres
genres = ['Blues', 'Rock', 'Metal', 'Reggae', 'Electronic', 'Jazz', 'Rap']

for root, dirs, files in os.walk(sourcePath):
    if os.path.basename(root) in genres:
        genreIdx = genres.index(os.path.basename(root))
        sid = 0
        for name in files:
            if '.p' in name and sid < 1179:
                song = pickle.load(open(root +'/'+ name, "rb"))
                # Current shape should be (1290, 128)
                song = np.expand_dims(song, axis=2)
                chunks = [song[x:x+129] for x in range(0, 1290-129, 129)]
                oneLabel = [0]*7
                oneLabel[genreIdx] = 1
                if sid < 950:
                    [x_train.append(x) for x in chunks]
                    [y_train.append(x) for x in [oneLabel]*len(chunks)]
                    print "parsed", name, "as training data"
                else:
                    [x_test.append(x) for x in chunks]
                    [y_test.append(x) for x in [oneLabel]*len(chunks)]
                    print "parsed", name, "as testing data"
                sid+=1

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.amax(x_train)
x_test /= np.amax(x_test)

print "done parsing and normalizing, final shapes are"

print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape

sPickle.s_dump(x_train, open(destPath + 'x_train.p', 'wb'))
sPickle.s_dump(y_train, open(destPath + 'y_train.p', 'wb'))
sPickle.s_dump(x_test, open(destPath + 'x_test.p', 'wb'))
sPickle.s_dump(y_test, open(destPath + 'y_test.p', 'wb'))



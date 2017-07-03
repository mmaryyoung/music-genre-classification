import os
import pickle
import numpy as np

# TODO:
# Read each folder in
# Slice every file into 1 second chunks
# Put every chunk into x_train
# Put corresponding label array into y_train


x_train = []
y_train = []
x_test = []
y_test = []


gid = 0
for root, dirs, files in os.walk('/data/hibbslab/jyang/Homemade Dataset'):
	if '_pickle' in root:
		sid = 0
		for name in files:
			if '.p' in name:
				print root + '/' + name
				longgrid = pickle.load(open(root + '/' + name, 'rb'))
				chunks = [longgrid[x:x+100] for x in range(0, len(longgrid),100)]

				oneLabel = [0]*10
				oneLabel[gid] = 1
				# Assumes at least 50 seconds
				# Train:test = 5:1 if len(chunks)==60
				[x_train.append(x) for x in chunks[:50]]
				[y_train.append(x) for x in [oneLabel]*50]
				[x_test.append(x) for x in chunks[50:]]
				[y_test.append(x) for x in [oneLabel]*(len(chunks)-50)]
				sid +=1
		gid += 1


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print 'x_train: ', x_train.shape
print 'y_train: ',  y_train.shape
print 'x_test: ', x_test.shape
print 'y_test: ', y_test.shape

pickle.dump(x_train, open('/data/hibbslab/jyang/x_train_hm.p', 'wb'))
pickle.dump(y_train, open('/data/hibbslab/jyang/y_train_hm.p', 'wb'))
pickle.dump(x_test, open('/data/hibbslab/jyang/x_test_hm.p', 'wb'))
pickle.dump(y_test, open('/data/hibbslab/jyang/y_test_hm.p', 'wb'))


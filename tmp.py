import pickle
import numpy as np

dataPath = '/data/hibbslab/jyang/tzanetakis/ver6.0/'
x_train = pickle.load(open(dataPath + 'x_train_mel.p', 'rb'))
x_test = pickle.load(open(dataPath + 'x_test_mel.p', 'rb'))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= np.amax(x_train)
x_test /= np.amax(x_test)

destPath = '/data/hibbslab/jyang/tzanetakis/ver7.0/'
pickle.dump(x_train, open(destPath + 'x_train.p', 'wb'))
pickle.dump(x_test, open(destPath + 'x_test.p', 'wb'))

print "done storing normalized data"

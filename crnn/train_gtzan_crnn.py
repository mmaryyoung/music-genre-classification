import numpy as np
from data_sanity_check import checkData
from keras_crnn import createCRNNModel
from keras.utils import np_utils
from keras.optimizers import Adam, Nadam, RMSprop
from process_history import plot_history
import sys

def loadall(filename=''):
    tmp = np.load(filename)
    x_tr = tmp['x_tr']
    y_tr = tmp['y_tr']
    x_te = tmp['x_te']
    y_te = tmp['y_te']
    x_cv = tmp['x_cv']
    y_cv = tmp['y_cv']
    return {'x_tr' : x_tr, 'y_tr' : y_tr,
            'x_te' : x_te, 'y_te' : y_te,
            'x_cv' : x_cv, 'y_cv' : y_cv, }

data = loadall('melspects.npz')

x_tr = data['x_tr']
y_tr = data['y_tr']
x_te = data['x_te']
y_te = data['y_te']
x_cv = data['x_cv']
y_cv = data['y_cv']

tr_idx = np.random.permutation(len(x_tr))
te_idx = np.random.permutation(len(x_te))
cv_idx = np.random.permutation(len(x_cv))

x_tr = x_tr[tr_idx]
y_tr = y_tr[tr_idx]
x_te = x_te[te_idx]
y_te = y_te[te_idx]
x_cv = x_cv[cv_idx]
y_cv = y_cv[cv_idx]

x_tr = x_tr / np.amax(x_tr)
x_te = x_te / np.amax(x_te)
x_cv = x_cv / np.amax(x_cv)

x_tr = x_tr[:,:,:,np.newaxis]
x_te = x_te[:,:,:,np.newaxis]
x_cv = x_cv[:,:,:,np.newaxis]

y_tr = np_utils.to_categorical(y_tr)
y_te = np_utils.to_categorical(y_te)
y_cv = np_utils.to_categorical(y_cv)

if not checkData(x_tr, x_te, x_cv):
    sys.exit()
opt = Nadam(learning_rate=0.0005)

model = createCRNNModel(x_tr.shape[1:])
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
history = model.fit(x=x_tr, y=y_tr, batch_size=20, epochs=100, verbose=2, validation_data=(x_te, y_te), shuffle=True)

plt = plot_history(history)
plt.show()
import numpy as np

def allNan(data):
    return np.any(np.isnan(data))

def allZero(data):
    return np.all((data==0), axis=(1, 2, 3)).any()

def maxMoreThanOne(data):
    return np.amax(data) > 1

def checkData(x_tr, x_te, x_cv):
    print('training data shape is %s' % str(x_tr.shape))

    if allNan(x_tr) or allNan(x_te) or allNan(x_cv) or allZero(x_tr) or allZero(x_te) or allZero(x_cv) or maxMoreThanOne(x_tr) or maxMoreThanOne(x_te) or maxMoreThanOne(x_cv):
        print('Some data is bad! ')
        print('Any item from the training data is nan? %s' % allNan(x_tr))
        print('Any item from the testing data is nan? %s' % allNan(x_te))
        print('Any item from the validation data is nan? %s' % allNan(x_cv))

        print('Any item from the training data is all-zero? %s' % allZero(x_tr))
        print('Any item from the testing data is all-zero? %s' % allZero(x_te))
        print('Any item from the validation data is all-zero? %s' % allZero(x_cv))

        print('Max training value: %f' % np.amax(x_tr))
        print('Max testing value: %f' % np.amax(x_te))
        print('Max validation value: %f' % np.amax(x_cv))
        return False
    print('Data is all good! ')
    return True

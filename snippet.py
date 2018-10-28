import numpy as np
import sPickle

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

source_rate = 22050
song_length = 30
seconds_per_sample = 5
samples_per_song = song_length/seconds_per_sample
num_genres = 10
songs_per_genre = 100

# Loading dataset
source_path = "/root/data/tzanetakis/ver9.0/"

def sPickleToArr(arr, fname):
    counter = 0
    for x in sPickle.s_load(open(source_path + fname)):
        arr[counter] = x
        counter+= 1
x_train = np.empty([num_genres*25*samples_per_song, source_rate*seconds_per_sample])
y_train = np.empty([num_genres*25*samples_per_song, num_genres])
x_test = np.empty([num_genres*10*samples_per_song, source_rate*seconds_per_sample])
y_test = np.empty([num_genres*10*samples_per_song, num_genres])

sPickleToArr(x_train, "x_train_mel.p")
sPickleToArr(y_train, "y_train_mel.p")
sPickleToArr(x_test, "x_test_mel.p")
sPickleToArr(y_test, "y_test_mel.p")

#x_train = sPickle.s_load(open(source_path + "x_train_mel.p")) 
#y_train = sPickle.s_load(open(source_path + "y_train_mel.p")) 
#x_test = sPickle.s_load(open(source_path + "x_test_mel.p")) 
#y_test = sPickle.s_load(open(source_path + "y_test_mel.p")) 
# Splitting data

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(x_train, y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
y_pred = classifier.predict(x_test)
print('Done.\nAccuracy: %f' % accuracy_score(y_test, y_pred))

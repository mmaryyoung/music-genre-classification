import librosa
import math
import numpy as np
import os

source_path = '/Users/maryyang/Learning/genres/'
dest_path = '/Users/maryyang/Learning/music-genre-classification/crnn/gtzan/10secs/'

source_rate = 22050
seconds_per_sample = 10
samples_per_song = 2
song_length = seconds_per_sample * samples_per_song
num_genres = 10
songs_per_genre = 100

training = np.array([]).reshape(0, source_rate*seconds_per_sample+1)
testing = np.array([]).reshape(0, source_rate*seconds_per_sample+1)
validation = np.array([]).reshape(0, source_rate*seconds_per_sample+1)

def parse_audio(genre_index, song_index, file_name):
    y, sr = librosa.load(file_name)

    # Truncate the raw data to even multuples of source rate and then chop it up to seconds_per_sample chunks
    if(len(y) > sr * song_length):
        raw_wave = y[:sr * song_length]
    elif(len(y) < sr * song_length):
        # Padding raw wave with zeroes
        raw_wave = np.zeros(sr*song_length)
        raw_wave[:len(y)] = y
    else:
        raw_wave = y
    sample_length = sr*seconds_per_sample
    chunks = raw_wave.reshape([-1, sample_length])
    samples = np.random.permutation(chunks)[:samples_per_song]
    labels = [[genre_index]] * samples_per_song
    # Append the label to the end of each sample. It's weird, I know.
    combined = np.concatenate((samples, labels), axis=1)

    global training
    global testing
    global validation
    if song_index < 80:
        training = np.concatenate((training, combined), axis=0)
    elif song_index < 90:
        testing = np.concatenate((testing, combined), axis=0)
    else:
        validation = np.concatenate((validation, combined), axis=0)

gid = 0
for root, dirs, files in os.walk(source_path):
    sid = 0
    print(root, gid)
    for name in files:
        if 'wav' in name:
            parse_audio(gid, sid, root + '/' + name)
            sid += 1
    if sid != 0:
        gid += 1

np.save(dest_path+'gtzan_tr.npy', training)
np.save(dest_path+'gtzan_te.npy', testing)
np.save(dest_path+'gtzan_cv.npy', validation)


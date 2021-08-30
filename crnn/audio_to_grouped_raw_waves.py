import librosa
import numpy as np
import os
import random

SOURCE_PATH = '../../genres/'
DEST_PATH = './gtzan/10secs/'

SOURCE_RATE = 22050
SECONDS_PER_SAMPLE = 10
SAMPLES_PER_SONG = 2
SONG_LENGTH = SECONDS_PER_SAMPLE * SAMPLES_PER_SONG
SONGS_PER_GENRE = 100

training = np.array([]).reshape(0, SOURCE_RATE*SECONDS_PER_SAMPLE+1)
testing = np.array([]).reshape(0, SOURCE_RATE*SECONDS_PER_SAMPLE+1)
validation = np.array([]).reshape(0, SOURCE_RATE*SECONDS_PER_SAMPLE+1)

# Generate a list of shuffled song indices to help randomly distribute the songs in a genre to training,
# testing, or validation data set.
shuffled_song_indices = list(range(SONGS_PER_GENRE))
random.shuffle(shuffled_song_indices)
def parse_audio(genre_index, song_index, file_name):
    y, sr = librosa.load(file_name)

    # Truncate the raw data to even multuples of source rate and then chop it up to SECONDS_PER_SAMPLE chunks
    if(len(y) > sr * SONG_LENGTH):
        raw_wave = y[:sr * SONG_LENGTH]
    elif(len(y) < sr * SONG_LENGTH):
        # Padding raw wave with zeroes
        raw_wave = np.zeros(sr*SONG_LENGTH)
        raw_wave[:len(y)] = y
    else:
        raw_wave = y
    sample_length = sr*SECONDS_PER_SAMPLE
    chunks = raw_wave.reshape([-1, sample_length])
    samples = np.random.permutation(chunks)[:SAMPLES_PER_SONG]
    labels = [[genre_index]] * SAMPLES_PER_SONG
    # Append the label to the end of each sample. It's weird, I know.
    combined = np.concatenate((samples, labels), axis=1)

    global training
    global testing
    global validation
    if shuffled_song_indices[song_index] < 80:
        training = np.concatenate((training, combined), axis=0)
    elif shuffled_song_indices[song_index] < 90:
        testing = np.concatenate((testing, combined), axis=0)
    else:
        validation = np.concatenate((validation, combined), axis=0)

gid = 0
for root, dirs, files in os.walk(SOURCE_PATH):
    sid = 0
    print(root, gid)
    for name in files:
        if 'wav' in name:
            parse_audio(gid, sid, root + '/' + name)
            sid += 1
    if sid != 0:
        gid += 1

np.save(DEST_PATH+'gtzan_tr.npy', training)
np.save(DEST_PATH+'gtzan_te.npy', testing)
np.save(DEST_PATH+'gtzan_cv.npy', validation)


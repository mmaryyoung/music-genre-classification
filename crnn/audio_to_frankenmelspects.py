import librosa as lb
import numpy as np
import os
import matplotlib.pyplot as plt

SOURCE_PATH = '/Users/maryyang/Learning/genres/'
MELSPECTS_DEST_PATH = '/Users/maryyang/Learning/music-genre-classification/crnn/gtzan/franken5secs/melspects.npz'

SOURCE_RATE = 22050
SECONDS_PER_SAMPLE = 5
SAMPLES_PER_SONG = 6
SONG_LENGTH = SECONDS_PER_SAMPLE * SAMPLES_PER_SONG
NUM_GENRES = 10
SONGS_PER_GENRE = 100
N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64 

x_train = []
x_test = []
x_val = []
y_train = []
y_test = []
y_val = []

def log_melspectrogram(data, log=True, plot=False, num='', genre='', hop_length=HOP_LENGTH, n_mels=N_MELS):

	melspec = lb.feature.melspectrogram(y=data, hop_length = hop_length, n_fft = N_FFT, n_mels = n_mels)

	if log:
		melspec = lb.power_to_db(melspec**2)

	if plot:
		melspec = melspec[np.newaxis, :]
		plt.imshow(melspec.reshape((melspec.shape[1],melspec.shape[2])))
		plt.savefig('melspec'+str(num)+'_'+str(genre)+'.png')

	return melspec


def parse_audio(genre_index, song_index, file_name):
    y, sr = lb.load(file_name)

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
    # Processes the overall-song into a one-mel melspectrogram. The hop length has to be multiplied so that
    # the final product is of the same length as the sample_mels (t1 = t2). (n_mels, t)
    song_mel = log_melspectrogram(y, log=True, plot=False, hop_length=HOP_LENGTH*SONGS_PER_GENRE, n_mels=1)
    # Parses the chunks as spectrogram.
    mel_chunks = []
    for i in range(len(chunks)):
        sample_mel = log_melspectrogram(chunks[i], log=True, plot=False)
        # Stitch the song-level one-mel spectrogram to the sample spectrogram. There might be a one-off error
        # Pad or truncate the song-mel depending on the situation.
        if sample_mel.shape[1] > song_mel.shape[1]:
            song_mel = np.pad(song_mel, ((0, 0), (0, sample_mel.shape[1]-song_mel.shape[1])), mode='edge')
        elif sample_mel.shape[1] < song_mel.shape[1]:
            song_mel = song_mel[:,:sample_mel.shape[1]]
        mel_chunks.append(np.append(sample_mel, song_mel, axis=0))

    mel_chunks = np.asarray(mel_chunks)
    # Scramble the samples within the song
    samples = np.random.permutation(mel_chunks)[:SAMPLES_PER_SONG]
    labels = [genre_index] * SAMPLES_PER_SONG

    if song_index < 80:
        x_train.extend(samples)
        y_train.extend(labels)
    elif song_index < 90:
        x_test.extend(samples)
        y_test.extend(labels)
    else:
        x_val.extend(samples)
        y_val.extend(labels)


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

x_train = np.array(x_train)
x_test = np.array(x_test)
x_val = np.array(x_val)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)
print("Saving to %s" % MELSPECTS_DEST_PATH)
np.savez(MELSPECTS_DEST_PATH, x_tr=x_train, y_tr=y_train, x_te=x_test, y_te=y_test, x_cv=x_val, y_cv=y_val)
print("Done")


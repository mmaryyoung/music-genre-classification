"""
This script aims to convert one or many audio files into individual
melspectrum sub-samples (samples from one audio will be in the same file).

Output: an .npz file to a local directory. It can be retrieved by np.load(FILE_PATH) into a dictionary.
    Here are some sample sizes for the output data: 
	data.shape = (5, 64, 1292)
"""

import argparse
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

def parse_audio(file_name):
    y, sr = librosa.load(file_name)

    # Truncate the raw data to even multuples of source rate and then
    # chop it up to SAMPLES_PER_SONG chunks of size SECONDS_PER_SAMPLE.
    if(len(y) > sr * SONG_LENGTH):
        raw_wave = y[:sr * SONG_LENGTH]
    elif(len(y) < sr * SONG_LENGTH):
        # Padding raw wave with zeroes.
        raw_wave = np.zeros(sr*SONG_LENGTH)
        raw_wave[:len(y)] = y
    else:
        raw_wave = y
    sample_length = sr*SECONDS_PER_SAMPLE
    chunks = raw_wave.reshape([-1, sample_length])
    samples = np.random.permutation(chunks)[:SAMPLES_PER_SONG]
    return samples


def log_melspectrogram(data, log=True, plot=False, num='', genre=''):
	melspec = librosa.feature.melspectrogram(y=data, hop_length = HOP_LENGTH, n_fft = N_FFT, n_mels = N_MELS)
	if log:
		melspec = librosa.power_to_db(melspec**2)
	if plot:
		melspec = melspec[np.newaxis, :]
		plt.imshow(melspec.reshape((melspec.shape[1],melspec.shape[2])))
		plt.savefig('melspec'+str(num)+'_'+str(genre)+'.png')
	return melspec

def batch_log_melspectrogram(data_list, log=True, plot=False):
	melspecs = np.asarray([log_melspectrogram(data_list[i],log=log,plot=plot) for i in range(len(data_list))])
	# This line may or may not be neccesary idk.
	# melspecs = melspecs.reshape(melspecs.shape[0], melspecs.shape[1], melspecs.shape[2], 1)
	return melspecs

# Parsing command line arguments.
arg_parser = argparse.ArgumentParser(
    description='Transforms a raw audio file into multiple melspectrogram sub-samples.')
arg_parser.add_argument(
    'src', help='The absolute path of a directory or file where the audio need to be processed.')
arg_parser.add_argument(
    'dst',
    help='The absolute path of the destnation file or directory. A directory is the default unless a suffix of .npz is used.')
arg_parser.add_argument(
    '-sr', '--source_rate', default=22050, help='The source rate of the incoming audio. Default is 22050.')
arg_parser.add_argument(
    '-sps', '--seconds_per_sample', default=5, help='How many seconds are in one sub-sample.')
arg_parser.add_argument(
    '-spf', '--samples_per_file', default=5, help='How many samples should we extract out of this audio file.')
arg_parser.add_argument(
    '-fft', '--number_of_fft', default=512, help='How many FFT windows should there be.')
arg_parser.add_argument(
    '-mel', '--number_of_mels', default=64, help='How many mels should be used.')

args = arg_parser.parse_args()

SOURCE_PATH = args.src
DEST_PATH = args.dst

SOURCE_RATE = args.sr
SECONDS_PER_SAMPLE = args.sps
SAMPLES_PER_SONG = args.spf
SONG_LENGTH = SECONDS_PER_SAMPLE * SAMPLES_PER_SONG
N_FFT = args.fft
HOP_LENGTH = N_FFT // 2
N_MELS = args.mel

if os.path.isdir(SOURCE_PATH) != os.path.isdir(DEST_PATH):
    raise ValueError('The source variable and the destination variable must be both directories or both files.')

if os.path.isdir(SOURCE_PATH):
    for root, dirs, files in os.walk(SOURCE_PATH):
        print(root)
        for name in files:
            if 'mp3' in name:
                melspecs = batch_log_melspectrogram(parse_audio(root + '/' + name))
                np.savez(DEST_PATH + os.path.splitext(name)[0] + '.npz')
                # TODO: find a way to carry the label information through.
else:
    melspecs = batch_log_melspectrogram(parse_audio(SOURCE_PATH))
    np.savez(DEST_PATH + os.path.splitext(SOURCE_PATH)[-2] + '.npz')
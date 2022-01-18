"""
The purpose of this script is to slice 30-second raw audios into smaller
sample clips. These clips can be used for human training/testing purposes.
"""

import os
from pathlib import Path
from pydub import AudioSegment

SOURCE_PATH = '../../genres/'
DEST_PATH = './gtzan/raw_audio_samples/'
SONGS_PER_GENRE = 100
SAMPLES_PER_SONG = 6
SECONDS_PER_SAMPLE = 5
MILLISECONDS_PER_SAMPLE = SECONDS_PER_SAMPLE * 1000

def slice_audio(filename):
    song = AudioSegment.from_wav(filename)
    samples = [song[i * MILLISECONDS_PER_SAMPLE : (i+1) * MILLISECONDS_PER_SAMPLE] for i in range(SAMPLES_PER_SONG)]
    # Get genre name and song name.
    genre_name, song_name = filename.split('/')[-2:]
    Path(DEST_PATH + genre_name).mkdir(parents=True, exist_ok=True)
    # Save the samples to a genre subdirectory.
    for sample_idx, sample in enumerate(samples):
        output_name= "%s%s/%s%02d.wav" % (DEST_PATH, genre_name, song_name, sample_idx)
        sample.export(output_name, format='wav')

for root, dirs, files in os.walk(SOURCE_PATH):
    print(root)
    for name in files:
        if 'wav' in name:
            print("processing %s" % name)
            slice_audio(root + '/' + name)
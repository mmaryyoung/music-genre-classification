from __future__ import print_function

# We'll need numpy for some mathematical operations
import numpy as np


# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')



# and IPython.display for audio output
import IPython.display


# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

audio_path = librosa.util.example_audio_file()

# or uncomment the line below and point it at your favorite song:
#
audio_path = '/Users/mac/Desktop/Homemade Dataset/edm/SMLE,Helen Tess - Overflow.wav'

y, sr = librosa.load(audio_path, duration=60)

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(S, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

plt.show()

# y_harmonic, y_percussive = librosa.effects.hpss(y)

# # What do the spectrograms look like?
# # Let's make and display a mel-scaled power (energy-squared) spectrogram
# S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
# S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

# # Convert to log scale (dB). We'll use the peak power as reference.
# log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
# log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)


# # We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
# # We'll use the harmonic component to avoid pollution from transients
# C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
# mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)

# # Let's pad on the first and second deltas while we're at it
# delta_mfcc  = librosa.feature.delta(mfcc)
# delta2_mfcc = librosa.feature.delta(mfcc, order=2)

# print(mfcc.shape)

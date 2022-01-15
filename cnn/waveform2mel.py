import numpy as np 
import librasa
import sPickle

source_path = "/root/data/tzanetakis/ver9.0/"
dest_path = "/root/data/tzanetakis/ver9.1/"

def wave2mel(sample):
	logam = librosa.logamplitude
	melgram = librosa.feature.melspectrogram
	longgrid = logam(melgram(y=sample, sr=22050,n_fft=1024, n_mels=128),ref_power=1.0)
	return longgrid.flatten()

for root, dirs, files in os.walk(source_path):
	for name in files:
		if ".p" in name:
			arr = sPickle.s_load(open(root + '/' + name, 'rb'))
			dest = []
			for a in arr:
				b = wave2mel(a)
				dest.append(b)
			dest = np.asarray(dest)
			print name, dest.shape
			sPickle.s_dump(dest, open(dest_path + name))
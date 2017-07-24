import os

for root, dirs, files in os.walk('/data/hibbslab/data/millionSong'):
	for name in files:
		# CHANGE HERE FOR FILE TYPE
		if '.h5' in name:
			try:
				os.system('python get_preview_url.py 7dqyaqprpze7 ' + root + '/' + name)
				print('yay we got one! ' + root + '/' + name)
			except urllib2.HTTPError:
				pass



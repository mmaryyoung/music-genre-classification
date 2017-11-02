import os

ab =  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
root = "/data/hibbslab/jyang/msd/ver2.0"
for letter1 in ab:
	level1 = root + "/" + letter1
	os.mkdir(level1)
	for letter2 in ab:
		level2 = level1 + "/" + letter2
		os.mkdir(level2)
		for letter3 in ab:
			level3 = level2 + "/" + letter3
			os.mkdir(level3)

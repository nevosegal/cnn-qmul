import os, librosa
import numpy as np

input_path = '/Users/nevosegal/Development/datasets/junglevibe2net/'

# get all folders/classes
input_classes = [
	input_path + folder + "/" for folder in os.listdir(input_path) if not folder.startswith('.')
]

# get all files apart from hidden
input_files = [
	ic+file for ic in input_classes for file in os.listdir(ic) if not os.path.isdir(ic+file) and not file.startswith('.')
]


for file in input_files:
	# load file
	y, sr = librosa.core.load(file)
	three_sec = 3*sr
	# compute number of slices/chunks
	num_slices = len(y) // three_sec
	for ix, n in enumerate(xrange(0, len(y), three_sec)):
		if n+three_sec <= len(y):
			chunk = y[n:n+three_sec]
			# get path from file name using last occurence of "/" delimiter
			path = file.rsplit("/", 1)
			# remove extension from file name
			file_name = path[1].rsplit(".",1)[0]
			chunk_name = path[0] + "/sliced/" + file_name + "_%d.wav" % n
			print("writing %s... %d/%d.wav" % (chunk_name, ix+1, num_slices))
			# write new chunks
			librosa.output.write_wav(chunk_name, chunk, sr)

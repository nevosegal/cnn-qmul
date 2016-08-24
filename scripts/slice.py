# A script that slices audio files to 3-second snippet
# taking into account the RMS, to discard any empty snippets

import os, librosa
import numpy as np

# RMS threshold - everything below this value will be discarded
threshold = 0.01

# set the input of audio files
input_path = '/Users/nevosegal/Development/qmul/dataset/good_sounds/junglevibes/'

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
    # compute number of slices/chunks that fully-fit in the audio file
    num_slices = len(y) // three_sec
    # iterate over the length of the file, jumping three seconds at a time
    for ix, n in enumerate(xrange(0, len(y), three_sec)):

        # if the end of the three second snippet is still in the file
        if n+three_sec <= len(y):
            # grab chunk
            chunk = y[n:n+three_sec]
            # get path from file name using last occurence of "/" delimiter
            path = file.rsplit("/", 1)
            # remove extension from file name
            file_name = path[1].rsplit(".",1)[0]
            # generate name for snippet: filename_index
            chunk_name = path[0] + "/sliced/" + file_name + "_%d.wav" % n

            # compute RMS
            chunk_rms = np.mean(librosa.feature.rmse(chunk))
            # if it is above threshold, write to file. Otherwise, discard
            if chunk_rms > threshold:
                # write new chunks
                print("writing %s... %d/%d.wav" % (chunk_name, ix+1, num_slices))
                librosa.output.write_wav(chunk_name, chunk, sr)
            else:
                print("not enough instrumental data in this snippet of audio %s" % chunk_name)

# read and slice MedleyDB using their metadata files

import yaml, os, librosa
import numpy as np
import time

# set audio path
audio_dir = './MedleyDB/Audio/'
# grab song folder list
folder_list = os.listdir(audio_dir)

# set the snippet length (3 seconds)
snippet_length = 3*22050

# set RMS threshold
threshold = 0.01

# set output path
output_path = '/Users/nevosegal/Development/qmul/dataset/MedleyDB/'

def parse_metadata(meta, path):

    # grab the stems from the metadata file
    stems = meta['stems']
    # generate their path
    stems_path = path + '/' + meta['stem_dir']

    # for each stem
    for stem in stems:
        # read it
        y, sr = librosa.core.load(stems_path + '/' + stems[stem]['filename'])
        # remove all spaces and slashes from instrument name
        instrument = stems[stem]['instrument'].replace(" ", "")
        instrument = instrument.replace("/", "")

        # compute number of snippets that fit in the stem
        num_snippets = np.floor(len(y)/snippet_length)

        for ix, i in enumerate(range(num_snippets.astype(int))):
            # compute start and end samples
            start_sample = (i*snippet_length)
            end_sample = (i*snippet_length) + snippet_length

            # grab snippet from stem
            snippet = y[start_sample: end_sample]

            # compute its RMS
            snippet_rms = np.mean(librosa.feature.rmse(snippet))

            check if above threshold. If not, discard it.
            if snippet_rms > threshold:
                if not os.path.exists(os.path.join(output_path, instrument)):
                    try:
                        os.makedirs(os.path.join(output_path, instrument))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                curr_time = int(time.time())
                write audio snippet with the current UNIX timestamp as a UID.
                librosa.output.write_wav('%s%s%d_%d.wav' % (os.path.join(output_path, instrument + '/'), instrument, ix, curr_time), snippet, sr)
            else:
                print('not enough instrumental data in snippet %s%d, skipping...' % (instrument, ix))

# create array of paths to each song directory
paths = [
    audio_dir + folder for folder in os.listdir(audio_dir)
]

# iterate over it
for path in paths:
    for file in os.listdir(path):
        if file.endswith('.yaml'):
            # grab the yaml file (metadata)
            yaml_path = path + '/' + file
            # open it
            with open(yaml_path, 'rb') as yaml_file:
                # grab the metadata and parse it
                metadata = yaml.load(yaml_file)
                parse_metadata(metadata, path)

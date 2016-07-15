import yaml, os, librosa
import numpy as np
import time


audio_dir = './MedleyDB/Audio/'
folder_list = os.listdir(audio_dir)

snippet_length = 3*22050
threshold = 0.005
output_path = '/Users/nevosegal/Development/qmul/dataset/MedleyDB/'

def parse_metadata(meta, path):
    stems = meta['stems']
    stems_path = path + '/' + meta['stem_dir']
    for stem in stems:
        y, sr = librosa.core.load(stems_path + '/' + stems[stem]['filename'])
        instrument = stems[stem]['instrument'].replace(" ", "")
        instrument = instrument.replace("/", "")
        num_snippets = np.floor(len(y)/snippet_length)
        for ix, i in enumerate(range(num_snippets.astype(int))):
            start_sample = (i*snippet_length)
            end_sample = (i*snippet_length) + snippet_length 
            snippet = y[start_sample: end_sample]
            snippet_rms = np.mean(librosa.feature.rmse(snippet))
            if snippet_rms > threshold:
                if not os.path.exists(os.path.join(output_path, instrument)):
                    try:
                        os.makedirs(os.path.join(output_path, instrument))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                curr_time = int(time.time())

                librosa.output.write_wav('%s%s%d_%d.wav' % (os.path.join(output_path, instrument + '/'), instrument, ix, curr_time), snippet, sr)
            else:
                print('not enough instrumental data in snippet %s%d, skipping...' % (instrument, ix))


paths = [ 
    audio_dir + folder for folder in os.listdir(audio_dir)
]

for path in paths:
    for file in os.listdir(path):
        if file.endswith('.yaml'):
            yaml_path = path + '/' + file
            with open(yaml_path, 'rb') as yaml_file:
                metadata = yaml.load(yaml_file)
                parse_metadata(metadata, path)

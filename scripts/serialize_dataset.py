# Serialising and importing datasets
# based on the code from the Tensorflow tutorial

import librosa as lr

import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

import h5py

training_root = 'dataset/IRMAS-TrainingData/'
testing_root = 'dataset/IRMAS-TestingData/'
sr = 22050
file_length = sr * 3


def make_arrays(num_rows, audio_length):
    if num_rows:
        data = np.ndarray(shape=(num_rows, audio_length), dtype=np.float32)
        labels = np.ndarray(shape=num_rows, dtype=np.int32)
    else:
        data, labels = None, None

    return data, labels


def merge_datasets(pickle_files, num_training, num_validation = 0):

    num_classes = len(pickle_files)

    validation_dataset, validation_labels = make_arrays(num_validation, file_length)
    training_dataset, training_labels = make_arrays(num_training, file_length)
    vsize_per_class = num_validation // num_classes
    tsize_per_class = num_training // num_classes

    vstart, tstart = 0, 0
    vend, tend = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, mode='rb') as f:
                instrument_set = pickle.load(f)

                # shuffling the data
                np.random.shuffle(instrument_set)
                if validation_dataset is not None:
                    # store vsize_per_class number of validation data from a specific instrument class.
                    validation_instrument = instrument_set[:vsize_per_class, :]
                    # store this in the merged dataset, in the appropriate location.
                    validation_dataset[vstart:vend, :] = validation_instrument
                    validation_labels[vstart:vend] = label
                    # increment the start and end positions for the next class.
                    vstart += vsize_per_class
                    vend += vsize_per_class

                # store the rest in the training examples dataset
                train_instrument = instrument_set[vsize_per_class:end_l, :]
                training_dataset[tstart:tend, :] = train_instrument
                training_labels[tstart:tend] = label
                # increment the start and end positions for the next class
                tstart += tsize_per_class
                tend += tsize_per_class
        except Exception as e:
            print("Cannot open file %s: %s" % (pickle_file, e))

    return validation_dataset, validation_labels, training_dataset, training_labels


# function to compute the mel spectrogram
def compute_spectrogram(signal):
    global sr
    spec = lr.feature.melspectrogram(signal, sr=sr, n_mels=128)
    # log_spec = lr.logamplitude(spec, ref_power=np.max)

    return spec


def generate_spec_mat(dataset, num_spec):
    spec_mat = np.zeros((num_spec, 128, 130), dtype=np.float32)
    for i, s in enumerate(dataset):
        S = compute_spectrogram(s)
        spec_mat[i, :, :] = S
    return spec_mat


def read_instrument(instrument_folder):

    # list of files in folder, ignoring hidden ones
    file_list = [f for f in os.listdir(instrument_folder) if not f.startswith('.')]

    # create dataset list in advance
    dataset = np.ndarray(shape=(len(file_list), file_length), dtype=np.float32)

    # counter to check how many audio files were read
    curr_audio = 0
    for file_name in file_list:
        full_path = os.path.join(instrument_folder, file_name)
        try:
            # read audio data
            audio_file, sr = lr.core.load(full_path)

            # if it is not 3 seconds long as expected, raise an error.
            if len(audio_file) != file_length:
                raise Exception("Unexpected file shape: %s" % str(len(audio_file)))

            # store in dataset list
            dataset[curr_audio, :] = audio_file
            curr_audio += 1
        except IOError as e:
            print("Couldn't read:", file_name, ":", e, "- Skipping...")

    print("%s out of %s audio files were successfully read from %s" %
          (str(curr_audio), str(len(file_list)), instrument_folder))

    # return only the amount of images that were correctly opened
    return dataset[0:curr_audio, :]


# serialize (pickle) the audio dataset
def pickle_dataset(folders):
    pickled_folders = []

    # read all files from each folder in the training set, and pickle
    for f in folders:
        pickle_name = f+'.pickle'
        pickled_folders.append(pickle_name)

        # don't pickle of already pickled
        if os.path.exists(pickle_name):
            print("Pickle %s already exists" % pickle_name)
        else:
            print("Pickling %s" % f)
            # read all files in a folder
            dataset = read_instrument(f)
            try:
                with open(pickle_name, 'wb') as output_file:
                    pickle.dump(dataset, output_file, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', pickle_name, ':', e)

    return pickled_folders


def initialise_dataset():
    hdf5_file = 'dataset.h5'
    contents = {}

    if not os.path.exists(hdf5_file):
        # populate array with full path of each instrument folder
        train_data_folders = [
            training_root + folder for folder in os.listdir(training_root)
            if os.path.isdir(os.path.join(training_root, folder))
            ]

        # populate array with full path of each instrument folder
        test_data_folders = [
            testing_root + folder for folder in os.listdir(testing_root)
            if os.path.isdir(os.path.join(testing_root, folder))
            ]

        # pickle datasets by instrument
        print("Pickling train datasets")
        train_datasets = pickle_dataset(train_data_folders)
        print("Pickling train datasets")
        test_datasets = pickle_dataset(test_data_folders)

        num_train, num_test, num_valid = 13559, 3225, 448
        # merge them to one big dataset
        valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, num_train, num_valid)
        _, _, test_dataset, test_labels = merge_datasets(test_datasets, num_test)

        # randomize them
        train_dataset, train_labels = randomize(train_dataset, train_labels)
        test_dataset, test_labels = randomize(test_dataset, test_labels)
        valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

        print('Training:', train_dataset.shape, train_labels.shape)
        print('Validation:', valid_dataset.shape, valid_labels.shape)
        print('Testing:', test_dataset.shape, test_labels.shape)

        train_spec_mat = generate_spec_mat(train_dataset, num_train)
        test_spec_mat = generate_spec_mat(test_dataset, num_test)
        valid_spec_mat = generate_spec_mat(valid_dataset, num_valid)

        # create a pickle of the merged datasets
        try:
            with h5py.File(hdf5_file, 'w') as f:
                contents = {
                    'train_spectros': train_spec_mat,
                    'train_dataset': train_dataset,
                    'train_labels': train_labels,
                    'test_spectros': test_spec_mat,
                    'test_dataset': test_dataset,
                    'test_labels': test_labels,
                    'valid_spectros': valid_spec_mat,
                    'valid_dataset': valid_dataset,
                    'valid_labels': valid_labels,
                    'num_classes': [len(train_data_folders)]
                }

                for key, val in contents.iteritems():
                    # pickle.dump(contents, f, pickle.HIGHEST_PROTOCOL)
                    f.create_dataset(key, data=val)

        except Exception as e:
            print("Unable to write to file", hdf5_file, ":", e)
            raise

    try:
        with h5py.File(hdf5_file, 'r') as f:
            for key in f.keys():
                contents[key] = np.array(f.get(key))

    except Exception as e:
        print("Unable to read file", hdf5_file, ":", e)
        raise

    return contents

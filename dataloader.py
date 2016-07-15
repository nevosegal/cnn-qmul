import numpy as np
import h5py

# DataLoader class - allows me to access data more cleanly
class DataLoader():
    def __init__(self, dataset_path, batch_size, train=True):

        contents = {}
        self.batch_size = batch_size
        self.read_pointer = 0

        try:
            with h5py.File(dataset_path, 'r') as f:
                for key in f.keys():
                    contents[key] = np.array(f.get(key))

        except Exception as e:
            print("Unable to read file", dataset_path, ":", e)
            raise

        self.data = {}
        if train:
            self.data = {
                'spectros': contents['train_spectros'],
                'labels': contents['train_labels']
            }
        else:
            self.data = {
                'spectros': contents['test_spectros'],
                'labels': contents['test_labels']
            }

        self.data['num_classes'] = int(contents['num_classes'])


    # return the next batch using a read pointer depending on the batch size
    def load_next_batch(self):
        num_batches = len(self.data['labels']) // self.batch_size
        ix = self.read_pointer*self.batch_size
        data_splitted = self.data['spectros'][ix:ix+self.batch_size]
        labels_splitted = self.data['labels'][ix:ix+self.batch_size]
        self.read_pointer+=1
        return data_splitted, labels_splitted

    def reset_read_pointer(self):
        self.read_pointer = 0

    # randomize the data
    def randomize(self):
        permutation = np.random.permutation(self.data['labels'].shape[0])
        shuffled_dataset = self.data['spectros'][permutation, :]
        shuffled_labels = self.data['labels'][permutation]
        self.data['spectros'] = shuffled_dataset
        self.data['labels'] = shuffled_labels

    def get_num_classes(self):
        return self.data['num_classes']

    def get_data_size(self):
        return len(self.data['labels'])

    def get_batch_size(self):
        return self.batch_size
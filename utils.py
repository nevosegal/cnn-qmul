import numpy as np

# accepts labels as class indices + the number of classes and create a one-hot representation
def generate_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i, oh in enumerate(one_hot):
        label_ix = labels[i]
        oh[label_ix] = 1
    return one_hot
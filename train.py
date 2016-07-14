# Constructing the CNN

# from utils import initialise_dataset, randomize
import numpy as np
from model import Model
import tensorflow as tf
from dataloader import DataLoader

train_batch_size = 64
dataloader = DataLoader('/Users/nevosegal/Development/qmul/dataset.h5', train_batch_size)

model = Model(train_batch_size)

def generate_one_hot(labels):
    num_classes = dataloader.get_num_classes()
    one_hot = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i, oh in enumerate(one_hot):
        label_ix = labels[i]
        oh[label_ix] = 1
    return one_hot

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

epochs = 30
for j in range(epochs):
    dataloader.randomize()
    dataset_size = dataloader.get_data_size()
    print dataset_size
    print dataset_size // dataloader.get_batch_size()
    print("Epoch number: %d" % j)
    for i in range(dataset_size // dataloader.get_batch_size()):
        train_sepctro_batch, train_labels_batch = dataloader.load_next_batch()
        train_one_hot_batch = generate_one_hot(train_labels_batch)
        train_accuracy = model.accuracy.eval(feed_dict={model.x: train_sepctro_batch, model.y_: train_one_hot_batch, model.keep_prob: 1.0})

        print('mini batch %d, training accuracy %g' % (i, train_accuracy))
        model.train_step.run(feed_dict={model.x: train_sepctro_batch, model.y_: train_one_hot_batch, model.keep_prob: 0.5})
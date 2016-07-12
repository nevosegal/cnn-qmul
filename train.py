# Constructing the CNN

from utils import initialise_dataset, randomize
import numpy as np
from model import Model

import tensorflow as tf


data = initialise_dataset()
train_spectros = data['train_spectros']
train_labels = data['train_labels']
test_spectros = data['test_spectros']
test_labels = data['test_labels']
valid_spectros = data['valid_spectros']
valid_labels = data['valid_labels']
num_classes = data['num_classes']
train_batch_size = 64
args = {
    'batch_size': train_batch_size
}

model = Model(args)

def generate_one_hot(labels):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i, oh in enumerate(one_hot):
        label_ix = labels[i]
        oh[label_ix] = 1
    return one_hot

test_one_hot = generate_one_hot(test_labels)

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())


epochs = 30
for j in range(epochs):
    train_spectros, train_labels = randomize(train_spectros, train_labels)
    train_one_hot = generate_one_hot(train_labels)
    print("Epoch number: %d" % j)
    for i in range(len(train_labels)/train_batch_size):
        train_sepctro_batch = train_spectros[i*train_batch_size:(i*train_batch_size)+train_batch_size]
        train_one_hot_batch = train_one_hot[i*train_batch_size:(i*train_batch_size)+train_batch_size]

        train_accuracy = model.accuracy.eval(feed_dict={model.x: train_sepctro_batch, model.y_: train_one_hot_batch, model.keep_prob: 1.0})

        print('mini batch %d, training accuracy %g' % (i, train_accuracy))
        model.train_step.run(feed_dict={model.x: train_sepctro_batch, model.y_: train_one_hot_batch, model.keep_prob: 0.5})

test_batch_size = 52
for i in range(len(test_labels)/test_batch_size):
    test_spectro_batch = test_spectros[i*test_batch_size:(i*test_batch_size)+test_batch_size]
    test_one_hot_batch = test_one_hot[i*test_batch_size:(i*test_batch_size)+test_batch_size]    
    test_accuracy = model.accuracy.eval(feed_dict={model.x: test_spectro_batch, model.y_: test_one_hot_batch, model.keep_prob: 1.0})
    print("Test accuracy %g" % test_accuracy)

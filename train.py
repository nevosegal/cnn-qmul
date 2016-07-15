# Constructing the CNN

# from utils import initialise_dataset, randomize
import numpy as np
from model import Model
import tensorflow as tf
from dataloader import DataLoader
import utils, os

train_batch_size = 64
dataloader = DataLoader('dataset.h5', train_batch_size)

model = Model(train_batch_size)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

epochs = 30
for j in range(epochs):
    dataloader.reset_read_pointer()
    dataloader.randomize()
    dataset_size = dataloader.get_data_size()

    print("Epoch number: %d" % j)
    for i in range(dataset_size // dataloader.get_batch_size()):
        train_sepctro_batch, train_labels_batch = dataloader.load_next_batch()
        train_one_hot_batch = utils.generate_one_hot(train_labels_batch, dataloader.get_num_classes())
        train_accuracy = model.accuracy.eval(feed_dict={model.x: train_sepctro_batch, model.y_: train_one_hot_batch, model.keep_prob: 1.0})

        print('mini batch %d, training accuracy %g' % (i, train_accuracy))
        model.train_step.run(feed_dict={model.x: train_sepctro_batch, model.y_: train_one_hot_batch, model.keep_prob: 0.5})

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
saver.save(sess, 'checkpoints/model.ckpt')
print("Model saved to checkpoints/model.ckpt")

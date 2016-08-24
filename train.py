import numpy as np
from model import Model
import tensorflow as tf
from dataloader import DataLoader
import utils, os
import argparse

# create an command line argument parset, for ease of use
parser = argparse.ArgumentParser()

# whether we want to overwrite the existing trained model (if exists) or continue training
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='overwrites existing model')
parser.add_argument('--no-overwrite', dest='overwrite', action='store_false', help='continues training based on existing model')
parser.set_defaults(overwrite=False)

# parse arguments
args = parser.parse_args()

# set training batch size
train_batch_size = 64

# checkpoint file path
checkpoint_file = 'checkpoints/model.ckpt'

# create dataloader
dataloader = DataLoader('dataset.h5', train_batch_size)

# create model
model = Model(train_batch_size)

# create tensor flow session
sess = tf.InteractiveSession()

# create tensorflow saver to restore the model (if exists)
saver = tf.train.Saver()

# restore model or initialise parameters randomly
if not args.overwrite and os.path.isfile(checkpoint_file):
    print("Restoring model...")
    saver.restore(sess, checkpoint_file)
    print("Successfully restored model!")
else:
    sess.run(tf.initialize_all_variables())
    print("Training on random initial values...")

# begin training

print("Starting training:")
dataset_size = dataloader.get_data_size()

# set number of epochs
epochs = 30
for j in range(epochs):
    # on every epoch, reset read pointer and randomize data
    dataloader.reset_read_pointer()
    dataloader.randomize()
    print("Epoch number: %d" % j)

    for i in range(dataset_size // dataloader.get_batch_size()):
        # get next batch
        train_sepctro_batch, train_labels_batch = dataloader.load_next_batch()
        # generate one-hot encoding from the ground-truth labels of the current batch
        train_one_hot_batch = utils.generate_one_hot(train_labels_batch, dataloader.get_num_classes())
        # get accuracy
        train_accuracy = model.accuracy.eval(feed_dict={model.x: train_sepctro_batch, model.y_: train_one_hot_batch, model.keep_prob: 1.0})
        # and print it
        print('mini batch %d, training accuracy %g' % (i, train_accuracy))
        model.train_step.run(feed_dict={model.x: train_sepctro_batch, model.y_: train_one_hot_batch, model.keep_prob: 0.5})

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
# save to checkpoint file
saver.save(sess, checkpoint_file)
print("Model saved to %s" % checkpoint_file)

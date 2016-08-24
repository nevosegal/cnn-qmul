import utils
from dataloader import DataLoader
from model import Model
import tensorflow as tf
import os
import numpy as np

# set batch size
test_batch_size = 52
# create dataloader object. this will be used to get the testing examples
dataloader = DataLoader('dataset.h5', test_batch_size, train=False)

# create the model
model = Model(test_batch_size)

# create tensorflow session
sess = tf.InteractiveSession()

# create new saver to restore the model
saver = tf.train.Saver()

# if there is a checkpoint file, load it and populate the weights and biases.
if os.path.isfile('checkpoints/model.ckpt'):
    saver.restore(sess, 'checkpoints/model.ckpt')
    print("Model restored!")
else:
    # otherwise, set all parameters to a small random value
    sess.run(tf.initialize_all_variables())

# making sure read pointer is at zero
dataloader.reset_read_pointer()
# randomize data
dataloader.randomize()
dataset_size = dataloader.get_data_size()

# create confusion matrix
confusion_matrix = np.zeros((12,12))

# start feeding in the batches
for i in range(dataset_size // dataloader.get_batch_size()):
    # loading the next available batch
    test_spectro_batch, test_labels_batch = dataloader.load_next_batch()
    # generate one-hot encoding from the ground-truth labels of the current batch
    test_one_hot_batch = utils.generate_one_hot(test_labels_batch, dataloader.get_num_classes())
    # get the predicted labels
    predicted_labels = model.prediction.eval(feed_dict={model.x: test_spectro_batch, model.y_: test_one_hot_batch, model.keep_prob: 1.0})
    # get the accuaacy and print it
    test_accuracy = model.accuracy.eval(feed_dict={model.x: test_spectro_batch, model.y_: test_one_hot_batch, model.keep_prob: 1.0})
    print("Test accuracy %g" % test_accuracy)
    # populate confusion matrix
    for j in range(len(predicted_labels)):
        confusion_matrix[test_labels_batch[j]][predicted_labels[j]] += 1
# print the confusion matrix
print(confusion_matrix)

import utils
from dataloader import DataLoader
from model import Model
import tensorflow as tf
import inspect_element
import os

test_batch_size = 64
dataloader = DataLoader('dataset.h5', test_batch_size, train=False)

model = Model(test_batch_size)

sess = tf.InteractiveSession()

# create new server to restore the model
saver = tf.train.Saver()

if os.path.isfile('checkpoints/model.ckpt'):
    saver.restore(sess, 'checkpoints/model.ckpt')
    print("Model restored!")

dataloader.reset_read_pointer()
dataloader.randomize()
dataset_size = dataloader.get_data_size()
print dataset_size

for i in range(dataset_size // dataloader.get_batch_size()):
    test_spectro_batch, test_labels_batch = dataloader.load_next_batch()
    test_one_hot_batch = utils.generate_one_hot(test_labels_batch, dataloader.get_num_classes())
    test_accuracy = model.accuracy.eval(feed_dict={model.x: test_spectro_batch, model.y_: test_one_hot_batch, model.keep_prob: 1.0})
    print("Test accuracy %g" % test_accuracy)

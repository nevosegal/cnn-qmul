# Constructing the CNN

from utils import initialise_dataset, randomize
import tensorflow as tf
import numpy as np
from BatchNormalizer import BatchNormalizer
import librosa as lr

import matplotlib.pyplot as plt


data = initialise_dataset()
train_spectros = data['train_spectros']
train_labels = data['train_labels']
test_spectros = data['test_spectros']
test_labels = data['test_labels']
valid_spectros = data['valid_spectros']
valid_labels = data['valid_labels']
num_classes = data['num_classes']


def generate_one_hot(labels):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i, oh in enumerate(one_hot):
        label_ix = labels[i]
        oh[label_ix] = 1
    return one_hot

test_one_hot = generate_one_hot(test_labels)

# example plotting spectrograms
# lr.display.specshow(train_spectros[222], x_axis='time', y_axis='mel')
# plt.title(str(train_labels[222]))
# plt.show()

# Convolutional Neural Network

# building the graph

# input
x = tf.placeholder(tf.float32)
x_tensor = tf.reshape(x, [-1, 128, 130, 1])
bn = BatchNormalizer(1, 0.001, True)
x_tensor_normalized = bn.normalize(x_tensor)

# ground-truth output
y_ = tf.placeholder(tf.float32)

# weights and biases from input to hidden layer 1
w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# apply the convolution
conv1 = tf.nn.conv2d(x_tensor_normalized, w_conv1, strides=[1, 1, 1, 1], padding="SAME")

# input to ReLU is w*x + b
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

# hidden layer 2
w_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME")

h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

# # hidden layer 3
# w_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
# b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
#
# conv3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding="SAME")
#
# h_conv3 = tf.nn.relu(conv3 + b_conv3)
# h_pool3 = tf.nn.max_pool(h_conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")


# fully connected layer
w_fc1 = tf.Variable(tf.truncated_normal([33 * 32 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool3_flat = tf.reshape(h_pool2, [-1, 33 * 32 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = tf.Variable(tf.truncated_normal([1024, 13], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[13]))

# the result
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())


train_batch_size = 64
epochs = 60
for j in range(epochs):
    train_spectros, train_labels = randomize(train_spectros, train_labels)
    train_one_hot = generate_one_hot(train_labels)
    print("Epoch number: %d" % j)
    for i in range(len(train_labels)/train_batch_size):
        train_sepctro_batch = train_spectros[i*train_batch_size:(i*train_batch_size)+train_batch_size]
        train_one_hot_batch = train_one_hot[i*train_batch_size:(i*train_batch_size)+train_batch_size]

        train_accuracy = accuracy.eval(feed_dict={x: train_sepctro_batch, y_: train_one_hot_batch, keep_prob: 1.0})

        print('mini batch %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: train_sepctro_batch, y_: train_one_hot_batch, keep_prob: 0.5})

test_batch_size = 128
for i in range(len(test_labels)):
    test_spectro_batch = test_spectros[i*test_batch_size:(i*test_batch_size)+test_batch_size]
    test_one_hot_batch = test_one_hot[i*test_batch_size:(i*test_batch_size)+test_batch_size]    
    test_accuracy = accuracy.eval(feed_dict={x: test_spectro_batch, y_: test_one_hot_batch, keep_prob: 1.0})
    print("Test accuracy %g" % test_accuracy)

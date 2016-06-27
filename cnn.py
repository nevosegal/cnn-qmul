# Constructing the CNN

from project import initialise_dataset
import tensorflow as tf
import numpy as np
from BatchNormalizer import BatchNormalizer


data = initialise_dataset()
train_spectros = data['train_spec_mat']
train_labels = data['train_labels']
test_spectros = data['test_spec_mat']
test_labels = data['test_labels']
valid_spectros = data['valid_spec_mat']
valid_labels = data['valid_labels']
num_classes = data['num_classes']

train_one_hot = np.zeros((len(train_labels), num_classes), dtype=np.int32)

# create a one hot encoding for the labels
for i, one_hot in enumerate(train_one_hot):
    label_ix = train_labels[i]
    one_hot[label_ix] = 1

# example plotting spectrograms
# lr.display.specshow(data['train_spec_mat'][110], x_axis='time', y_axis='mel')
# plt.title(str(data['train_labels'][110]))
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
w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# apply the convolution
conv1 = tf.nn.conv2d(x_tensor_normalized, w_conv1, strides=[1, 1, 1, 1], padding="SAME")

# input to ReLU is w*x + b
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

# hidden layer 2
w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME")

h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

# hidden layer 3
w_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))

conv3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding="SAME")

h_conv3 = tf.nn.relu(conv3 + b_conv3)
h_pool3 = tf.nn.max_pool(h_conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")


# fully connected layer
w_fc1 = tf.Variable(tf.truncated_normal([17 * 16 * 128, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool3_flat = tf.reshape(h_pool3, [-1, 17 * 16 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = tf.Variable(tf.truncated_normal([1024, 11], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[11]))

# the result
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

batch_size = 55

for i in range(len(train_labels)/batch_size):
    spectro_batch = train_spectros[i*batch_size:(i*batch_size)+batch_size]
    one_hot_batch = train_one_hot[i*batch_size:(i*batch_size)+batch_size]

    train_accuracy = accuracy.eval(feed_dict={x: spectro_batch, y_: one_hot_batch, keep_prob: 1.0})

    print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: spectro_batch, y_: one_hot_batch, keep_prob: 0.5})


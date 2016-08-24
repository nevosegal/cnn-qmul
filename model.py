# Convolutional Neural Network model
# Adapted from https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network

import tensorflow as tf
from BatchNormalizer import BatchNormalizer
import numpy as np

# constructing the model
class Model():

    def __init__(self, batch_size):

        # set global parameter of batch size
        self.input_size = batch_size

        # instantiate a new batch normalizer
        self.bn = BatchNormalizer(1, 0.001, True)

        # placeholder to store amount of dropout, fed during training.
        self.keep_prob = tf.placeholder(tf.float32)

        # building the graph

        # input layer
        self.x = tf.placeholder(tf.float32)

        # reshape the input to a tensor
        self.x_tensor = tf.reshape(self.x, [self.input_size, 128, 130, 1])
        # and normalize it to have zero mean and unit variance
        self.x_tensor_normalized = self.bn.normalize(self.x_tensor)

        # ground-truth output - is fed during training/testing
        self.y_ = tf.placeholder(tf.float32)


        # weights and biases from input to hidden layer 1
        self.weights_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.1), name="weights_conv1")
        self.bias_conv1 = tf.Variable(tf.constant(0.1, shape=[64]), name="bias_conv1")

        # apply the first convolutional layer, moving in steps of 1 in each direction
        self.conv1 = tf.nn.conv2d(self.x_tensor_normalized, self.weights_conv1, strides=[1, 1, 1, 1], padding="SAME")

        # apply non-linear function - ReLU. input to it is w*x + b. in this case * stands for convolution
        # unlike fully connected layers that use multiplication.
        self.relu_conv1 = tf.nn.relu(self.conv1 + self.bias_conv1)

        # output from hidden layer goes through max-pooling.
        self.output_conv1 = tf.nn.max_pool(self.relu_conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")


        # hidden layer 2 - weights and biases
        self.weights_conv2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name="weights_conv2")
        self.bias_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name="bias_conv2")

        # second convolutional layer
        self.conv2 = tf.nn.conv2d(self.output_conv1, self.weights_conv2, strides=[1, 1, 1, 1], padding="SAME")

        # apply activation function
        self.relu_conv2 = tf.nn.relu(self.conv2 + self.bias_conv2)
        # apply drouput - amount of dropout is fed during training
        self.dropout_conv2 = tf.nn.dropout(self.relu_conv2, self.keep_prob)
        # apply max-pooling
        self.output_conv2 = tf.nn.max_pool(self.relu_conv2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")


        # hidden layer 3 - weights and biases
        self.weights_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name="weights_conv3")
        self.bias_conv3 = tf.Variable(tf.constant(0.1, shape=[64]), name="bias_conv3")

        # third convolutional layer
        self.conv3 = tf.nn.conv2d(self.output_conv2, self.weights_conv3, strides=[1, 1, 1, 1], padding="SAME")

        # apply activation function
        self.relu_conv3 = tf.nn.relu(self.conv3 + self.bias_conv3)
        # apply drouput - amount of dropout is fed during training
        self.dropout_conv3 = tf.nn.dropout(self.relu_conv3, self.keep_prob)
        # apply max-pooling
        self.output_conv3 = tf.nn.max_pool(self.dropout_conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")


        # hidden layer 4 - weights and biases
        self.weights_conv4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name="weights_conv4")
        self.bias_conv4 = tf.Variable(tf.constant(0.1, shape=[64]), name="bias_conv4")

        # fourth convolutional layer
        self.conv4 = tf.nn.conv2d(self.output_conv3, self.weights_conv4, strides=[1, 1, 1, 1], padding="SAME")

        # apply activation function
        self.relu_conv4 = tf.nn.relu(self.conv4 + self.bias_conv4)
        # apply dropout - amount of dropout is fed during training
        self.dropout_conv4 = tf.nn.dropout(self.relu_conv4, self.keep_prob)
        # apply max-pooling
        self.output_conv4 = tf.nn.max_pool(self.dropout_conv4, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")


        # first fully connected layer with 1024 outputs
        self.weights_fc1 = tf.Variable(tf.truncated_normal([int(8 * 9 * 64), 1024], stddev=0.1), name="weights_fc1")
        self.bias_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name="bias_fc1")

        self.fc1 = tf.reshape(self.output_conv4, [-1, int(8 * 9 * 64)])

        # output from fully connected layer after multiplication, bias and ReLU
        self.relu_fc1 = tf.nn.relu(tf.matmul(self.fc1, self.weights_fc1) + self.bias_fc1)

        # apply dropout
        self.output_fc1 = tf.nn.dropout(self.relu_fc1, self.keep_prob)


        # softmax layer: input 1024, output 12 which are the softmax probabilities
        self.weights_softmax = tf.Variable(tf.truncated_normal([1024, 12], stddev=0.1), name="weights_softmax")
        self.bias_softmax = tf.Variable(tf.constant(0.1, shape=[12]), name="bias_softmax")

        # the probability vector
        self.y_conv = tf.nn.softmax(tf.matmul(self.output_fc1, self.weights_softmax) + self.bias_softmax)


        # apply cross entropy as the cost function
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(self.y_conv, 1e-10, 1.0)))

        # try to minimize the cross entropy using an ADAM Optimizer
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # check how many correct predictions there are.
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        # save the index of the prediction, will be used to construct the confusion matrix
        self.prediction = tf.argmax(self.y_conv,1)

        # derive overall accuracy from it.
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

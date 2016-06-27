# based on http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

import tensorflow as tf


class BatchNormalizer(object):

    def __init__(self, depth, epsilon, post_scale):
        self.mean = tf.Variable(tf.constant(0.0, shape=[depth]))
        self.variance = tf.Variable(tf.constant(1.0, shape=[depth]))
        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
        self.epsilon = epsilon
        self.post_scale = post_scale

    def normalize(self, x):
        mean, variance = tf.nn.moments(x, [0, 1, 2])
        x_normalized = tf.nn.batch_norm_with_global_normalization(x, mean, variance, self.beta, self.gamma, self.epsilon, self.post_scale)
        return x_normalized

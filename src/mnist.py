import argparse
import tensorflow as tf
import numpy as np

class MNIST:

    def __init__(self):
        self.g = tf.Graph()
        with self.g.as_default():
            with tf.variable_scope("input"):
                self.x = tf.placeholder(tf.float32, shape=[None, 784])
                self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
                x_image = tf.reshape(self.x, [-1, 28,28,1])

            with tf.variable_scope("conv1"):
                W_conv1 = self.weight_variable([5, 5, 1, 32])
                b_conv = self.bias_variable([32])

                h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
                h_pool1 = sef.max_pool_2x2(h_conv1)

            with tf.variable_scope("conv2"):
                W_conv2 = self.weight_variable([5, 5, 32, 64])
                b_conv2 = self.bias_variable([64])

                h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
                h_pool2 = sef.max_pool_2x2(h_conv2)
                
import tensorflow as tf 
import numpy as np 
import png
import random
import os
import code

class GAN:

    def __init__(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')\

        self.g_x, self.g_y, self.g_y_logits = self.build_generator()

        with tf.variable_scope('discriminator') as scope:
            self.d_x = tf.placeholder(tf.float32, shape=[None, 784])
            self.d_y_ = tf.placeholder(tf.float32, shape=[None, 1])
            self.d_keep_prob = tf.placeholder(tf.float32, name='d_keep_prob')
            scope.reuse_variables()
            self.g_d_y, self.g_d_y_logit = self.build_descriminitor(self.g_y, self.d_keep_prob)
        
        vars = tf.trainable_variables()

        #building loss function for discriminator
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(self.d_y_logit, tf.ones_like(self.d_y_logit))
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(self.g_d_y_logit, tf.zeros_like(self.g_d_y_logit))
        self.d_loss = d_loss_real + d_loss_fake

        d_training_vars = [v for v in vars if v.name.startswith('discriminator/')]
        self.d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=d_training_vars)

        self.d_accuracy = tf.reduce_sum(f.cast(f.round(self.d_y_logit), tf.round(self.d_y_)), tf.float32)

        #building loss function for generator
        self.g_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.g_d_y_logit, tf.ones_like(self.tf.g_d_y_logit))

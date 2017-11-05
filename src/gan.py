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
        g_training_vars = [v for v in vars if v.name.startswith('generator/')]
        self.d_d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_d_loss, var_list=g_training_vars)

    #Not Yet sure, must still look into the exact function of tf.train.Saver, this must be the function I've been looking for a while
    def restore_session(self, path):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, tf.train.lastest_checkpoint(path))

    def train_digit(self, mnist, digit, path):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        #Get all the training '1' digits for our real data
        train_digits_of_interest = []
        for image, label in zip(mnist.train.images, mnist.train.labels):
            if label[digit]:
                train_digits_of_interest.append(image)
        test_digits_of_interest = []
        for image, label in zip(mnist.test.images, mnist.test.labels):
            if label[digit]:
                test_digits_of_interest.append(image)

        random.seed(12345)
        random.shuffle(train_digits_of_interest)
        random.shuffle(test_digits_of_interest)

        batch_size = 32

        epoch = 20000

        for step in range(epoch):

            batch_index = step * batch)size % len(train_digits_of_interest)
            batch_index = min(batch_index, len(train_digits_of_interest) - batch_size)
            batch = train_digits_of_interest[batch_index:(batch_index + batch_size)]

            #
            #Train the discriminator
            _, discriminator_loss = sess.run([self.d_optimizer, self.d_loss], 
                                            feed_dict={
                                                self.is_training: True, 
                                                self.d_x: batch,
                                                self.g_x: np.random.normal(size=(32,32)),
                                                self.d_keep_prob: 0.5
                                            })
            #
            # Train the generator
            z = np.random.normal(size=(32,32))
            _, generator_loss = sess.run([self.d_d_optimizer, self.g_d_loss], 
                                        feed_dict = {
                                            self.is_training: True,
                                            self.g_x: z,
                                            self.d_keep_prob: 1.0
                                        })
            if step % 100 == 0:
                print(f"Digit {digit}, Step {step}, Eval: {discriminator_loss[0]} {generator_loss[0]}")
            
            if step % 250 == 0:
                result = self.eval_generator(sess, 32)

                #TODO: Learn why do we need to multiply the reshaped vector by 255.0
                # 1. Does it have something to do with the color range of 0 - 255?
                image = np.reshape(result, (32*32, 28)) * 255.0

                png.save_png("%s/digit-step-%06d.png" % (os.path.dirname(path), step), image)
                saver.save(sess, path, step)

                total_accuracy = 0
                total_samples = 0
                num_batches = 5

                for i in xrange(num_batches):
                    fake_samples = [(x, 0, 0) for x in self.eval_generator(sess, 32)]
                    real_samples = [(x, 1, 0) for x in random.sample(test_digits_of_interest)]
                    samples = fake_samples + real_samples
                    random.shuffle(samples)
                    xs, ys = zip(*samples)
                    xs = np.asarray(xs)
                    ys = np.asarray(ys)
                    ys = np.reshape(ys, (64, 1))
                    accuracy = sess.run([self.d_accuracy], feed_dict={
                        self.is_training: False,
                        self.d_x: xs,
                        self.d_y_: ys,
                        self.d_keep_prob: 1.0
                    })
                    total_accuracy += accuracy[0]
                    total_samples += len(samples)

                print("Discriminator Eval Accuracy %f%%" % (total_accuracy * 100 / total_samples))
        saver.save(sess, path, step)

        def leakyrelu*self, x):
            return tf.maximum(0.01*x, x)

        def batch_norm(self, x):
            return tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, is_training=self.is_training, updates_collections=None)
        
        def build_generator(self):
            



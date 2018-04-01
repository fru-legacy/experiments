import tensorflow as tf
import numpy as np
from helper.BaseNetwork import BaseNetwork
from helper.decorators import scope
from helper.helper import repeat_constant_as_shape
from helper.flip_gradient import flip_gradient
from tested.gather_batch import gather_batch
from tested.mmd_kernel import compute_mmd


class Noise0Autoencoder(BaseNetwork):
    def __init__(self, data):
        super(Noise0Autoencoder, self).__init__(data)

        # Parameter
        self.latent_stddev = 1.0 / (255 * 8)

        # Data
        self.patches = self.data.patches([28, 28], auxiliary_max_count=0)
        self.base_size = self.data.info.image_values_count
        self.generator_size = self.data.info.label_count

        # Network
        self.optimizers.append(self.optimize)
        tf.summary.scalar('error', self.error)

    @scope(cached_property=True)
    def input(self):
        x = tf.cast(self.data.image_flat, tf.float32) / 255.0
        return x

    @scope(cached_property=True)
    def forward(self):
        x = self.input
        x = tf.layers.dense(x, self.base_size * 2, activation=tf.nn.elu)
        x = tf.layers.dense(x, self.base_size * 2, activation=tf.nn.elu)
        # x = x + tf.random_uniform(tf.shape(x), 0.0, 0.05)
        x = tf.layers.dense(x, self.base_size * 2, activation=tf.nn.elu)
        x = tf.layers.dense(x, self.generator_size, activation=tf.nn.elu)
        return x

    @scope(cached_property=True)
    def optimize(self):
        x = tf.losses.softmax_cross_entropy(self.data.label_one_hot, self.forward)
        tf.summary.scalar('cross entropy', x)
        return tf.train.AdamOptimizer().minimize(x)

    @scope(cached_property=True)
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.data.label_one_hot, 1), tf.argmax(self.forward, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

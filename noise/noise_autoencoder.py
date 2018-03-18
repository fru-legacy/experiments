import tensorflow as tf
from helper.decorators import scope
from helper.alpha_dropout import alpha_dropout_enabled
from tensorflow.contrib.keras.api.keras.initializers import lecun_normal
import tensorflow.contrib.layers as layers
from helper.helper import repeat_constant_as_shape
import numpy as np
from helper.flip_gradient import flip_gradient


class NoiseAutoencoder:
    def __init__(self, data):

        # Data
        self.data = data
        self.image = data.image
        self.image_size = [data.info.height, data.info.width]
        self.label = data.label_one_hot
        self.channels = data.info.color_channels
        self.patches = data.patches([4, 4], auxiliary_max_count=2)

        self.dropout_enabled = tf.placeholder_with_default(True, shape=())

        # Settings
        self.shape = [12, 64, 128]
        self.defaults = {'activation_fn': tf.nn.selu, 'weights_initializer': lecun_normal()}

        # Network
        self.generator_clipped = tf.clip_by_value(self.generator, 0, 1)
        self.patches.restored_image_summary('restored', self.generator_clipped)
        self.latent_random = tf.random_uniform(tf.shape(self.latent_image), 0.0, 1.0)
        self.discriminator_random = self._discriminator(self.latent_random, 0.0, reuse=False)
        self.discriminator_latent = self._discriminator(self.latent_image, 1.0, reuse=True)
        self.prediction_optimize = self._prediction_optimize()
        self.prediction_error = self._error()

    @scope(cached_property=True)
    def latent(self):
        # latent_input = tf.random_uniform(tf.shape(self.patches.data), 0.0, 1.0)
        # x = tf.concat([self.patches.data, latent_input], 1)
        x = self.patches.data
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = layers.fully_connected(x, self.shape[2], **self.defaults)
        x = layers.fully_connected(x, self.shape[0], **self.defaults)
        return x

    @scope(cached_property=True)
    def latent_image(self):
        return tf.reshape(self.latent, [-1, self.shape[0], self.patches.count, 1])

    @scope(cached_property=True)
    def generator(self):
        x = self.latent
        x = x + tf.random_uniform(tf.shape(x), 0.0, 0.05)
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = layers.fully_connected(x, self.shape[2], **self.defaults)
        x = alpha_dropout_enabled(x, 0.8, self.dropout_enabled)
        x = layers.fully_connected(x, self.patches.size, **self.defaults)
        return x

    @scope(cached_property=True)
    def generator_optimize(self):
        cross_entropy = tf.losses.mean_squared_error(self.patches.data, self.generator)
        tf.summary.scalar('error', cross_entropy)
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    @scope('discriminator', reuse=tf.AUTO_REUSE)
    def _discriminator(self, latent, target, reuse):
        x = flip_gradient(latent)
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = layers.fully_connected(x, self.shape[2], **self.defaults)
        x = layers.fully_connected(x, 1, **self.defaults)
        target = repeat_constant_as_shape(target, x)
        cross_entropy = tf.losses.mean_squared_error(x, target)
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    @scope
    def _prediction_optimize(self):
        x = tf.reshape(self.latent, [-1, self.patches.count * self.shape[0]])
        self.prediction = layers.fully_connected(x, 10, activation_fn=None)
        self.prediction_variables = tf.get_variable_scope().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        cross_entropy = tf.losses.softmax_cross_entropy(self.label, self.prediction)
        return tf.train.AdamOptimizer().minimize(cross_entropy, var_list=self.prediction_variables)

    @scope
    def _error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
import tensorflow as tf
from helper.decorators import variable_scope
from helper.alpha_dropout import alpha_dropout_enabled
from tensorflow.contrib.keras.api.keras.initializers import lecun_normal
from tested.image_patches import split_patches_into_batch, op_join_patches_from_batch
import tensorflow.contrib.layers as layers
import numpy as np
from helper.flip_gradient import flip_gradient

class NoiseAutoencoder:
    def __init__(self, data):
        self.image = data.image
        self.label = data.label
        self.original_image_size = [data.height, data.width]
        self.channels = data.channels
        self.patches = split_patches_into_batch(self.image, patch_size=[4, 4], channels=data.channels, reversible=True)
        self.patches_size = int(np.prod(self.patches.get_shape()[1:]))
        self.dropout_enabled = tf.placeholder_with_default(True, shape=())

        # Settings
        self.shape = [64, 128]
        self.defaults = {'activation_fn': tf.nn.selu, 'weights_initializer': lecun_normal()}

        # Network
        self.latent_input = tf.random_uniform(tf.shape(self.patches), 0.0, 1.0)
        self.latent = self._latent()
        self.generator = self._generator()
        self.generator_clipped = tf.clip_by_value(self.generator, 0, 1)
        self.generator_optimize = self._generator_optimize()
        self.restored = op_join_patches_from_batch(self.generator_clipped, self.original_image_size, patch_size=[4, 4], channels=data.channels)
        self.latent_random = tf.random_uniform(tf.shape(self.latent), 0.0, 1.0)
        self.discriminator_random = self._discriminator(self.latent_random, tf.constant([[0.0]]), reuse=False)
        self.discriminator_latent = self._discriminator(self.latent, tf.constant([[1.0]]), reuse=True)
        #self.prediction_optimize = self._prediction_optimize()
        #self.prediction_error = self._error()
        #self.optimize_discriminator = self._optimize_discriminator()

    @variable_scope('latent')
    def _latent(self):
        x = tf.concat([self.patches, self.latent_input], 1)
        x = layers.fully_connected(x, self.shape[0], **self.defaults)
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = layers.fully_connected(x, self.shape[0], **self.defaults)
        return x

    @variable_scope('generator')
    def _generator(self):
        x = self.latent
        x = x + tf.random_uniform(tf.shape(x), 0.0, 0.05)
        x = layers.fully_connected(x, self.shape[0], **self.defaults)
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = alpha_dropout_enabled(x, 0.8, self.dropout_enabled)
        x = layers.fully_connected(x, self.patches_size, **self.defaults)
        return x

    @variable_scope()
    def _generator_optimize(self):
        self.cross_entropy = tf.losses.mean_squared_error(self.patches, self.generator)
        return tf.train.AdamOptimizer().minimize(self.cross_entropy)

    @variable_scope('discriminator', reuse=tf.AUTO_REUSE)
    def _discriminator(self, latent, target, reuse):
        x = flip_gradient(latent)
        x = layers.fully_connected(x, self.shape[0], scope="t1", **self.defaults, reuse=tf.AUTO_REUSE)
        x = layers.fully_connected(x, self.shape[1], scope="t2", **self.defaults)
        x = layers.fully_connected(x, 1, scope="t3", **self.defaults)
        cross_entropy = tf.losses.mean_squared_error(x, tf.tile(target, tf.shape(x)))
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    @variable_scope
    def _prediction_optimize(self):
        x = tf.reshape(self.latent, [128, 8^4])
        self.prediction = layers.fully_connected(x, 10, activation_fn=None)
        cross_entropy = tf.losses.softmax_cross_entropy(self.label, self.prediction)
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    @variable_scope
    def _error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
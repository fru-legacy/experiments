import tensorflow as tf
from helper.decorators import scope
from helper.alpha_dropout import alpha_dropout_enabled
from tensorflow.contrib.keras.api.keras.initializers import lecun_normal
from tested.image_patches import split_patches_into_batch, op_join_patches_from_batch
import tensorflow.contrib.layers as layers
import numpy as np
from helper.flip_gradient import flip_gradient
from cached_property import cached_property

class NoiseAutoencoder:
    def __init__(self, data):

        # Data
        self.data = data
        self.image = data.image
        self.image_size = [data.height, data.width]
        self.label = data.label
        self.channels = data.channels

        patches = split_patches_into_batch(self.image, patch_size=[4, 4], channels=data.channels, reversible=True)
        self.patches, self.patches_size, self.patches_count = patches

        self.dropout_enabled = tf.placeholder_with_default(True, shape=())

        # Settings
        self.shape = [12, 64, 128]
        self.defaults = {'activation_fn': tf.nn.selu, 'weights_initializer': lecun_normal()}

        # Network
        self.generator_clipped = tf.clip_by_value(self.generator, 0, 1)
        #self.generator_optimize = self._generator_optimize()
        self.restored = op_join_patches_from_batch(self.generator_clipped, self.image_size, patch_size=[4, 4], channels=data.channels)
        self.latent_random = tf.random_uniform(tf.shape(self.latent_image), 0.0, 1.0)
        self.discriminator_random = self._discriminator(self.latent_random, 0.0, reuse=False)
        self.discriminator_latent = self._discriminator(self.latent_image, 1.0, reuse=True)
        self.prediction_optimize = self._prediction_optimize()
        self.prediction_error = self._error()

    @scope(cached_property=True)
    def latent(self):
        print('Test2')
        latent_input = tf.random_uniform(tf.shape(self.patches), 0.0, 1.0)
        x = tf.concat([self.patches, latent_input], 1)
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = layers.fully_connected(x, self.shape[2], **self.defaults)
        x = layers.fully_connected(x, self.shape[0], **self.defaults)
        return x

    @scope(cached_property=True)
    def latent_image(self):
        return tf.reshape(self.latent, [-1, self.shape[0], self.patches_count, 1])

    @scope(cached_property=True)
    def generator(self):
        x = self.latent
        x = x + tf.random_uniform(tf.shape(x), 0.0, 0.05)
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = layers.fully_connected(x, self.shape[2], **self.defaults)
        x = alpha_dropout_enabled(x, 0.8, self.dropout_enabled)
        x = layers.fully_connected(x, self.patches_size, **self.defaults)
        return x

    @scope(cached_property=True)
    def generator_optimize(self):
        cross_entropy = tf.losses.mean_squared_error(self.patches, self.generator)
        tf.summary.scalar('error', cross_entropy)
        generator_clipped = tf.clip_by_value(self.generator, 0, 1)
        self.restored = op_join_patches_from_batch(generator_clipped, self.data.image_size, patch_size=[4, 4],
                                              channels=self.data.channels)
        tf.summary.image('generated', self.restored, 1)
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    def repeat_constant_as_shape(self, constant, shape):
        dims = np.repeat(1, len(shape.get_shape()))
        return tf.tile(tf.reshape(constant, dims), tf.shape(shape))

    @scope('discriminator', reuse=tf.AUTO_REUSE)
    def _discriminator(self, latent, target, reuse):
        x = flip_gradient(latent)
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = layers.fully_connected(x, self.shape[2], **self.defaults)
        x = layers.fully_connected(x, 1, **self.defaults)
        target = self.repeat_constant_as_shape(target, x)
        cross_entropy = tf.losses.mean_squared_error(x, target)
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    @scope
    def _prediction_optimize(self):
        x = tf.reshape(self.latent, [-1, self.patches_count * self.shape[0]])
        self.prediction = layers.fully_connected(x, 10, activation_fn=None)
        self.prediction_variables = tf.get_variable_scope().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        cross_entropy = tf.losses.softmax_cross_entropy(self.label, self.prediction)
        return tf.train.AdamOptimizer().minimize(cross_entropy, var_list=self.prediction_variables)

    @scope
    def _error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
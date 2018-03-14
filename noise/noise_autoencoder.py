import tensorflow as tf
from helper.decorators import variable_scope
from helper.alpha_dropout import alpha_dropout_enabled
from tensorflow.contrib.keras.api.keras.initializers import lecun_normal
from tested.image_patches import split_patches_into_batch, op_join_patches_from_batch
import tensorflow.contrib.layers as layers
import numpy as np

class NoiseAutoencoder:
    def __init__(self, data):
        self.image = data.image
        self.original_image_size = [data.height, data.width]
        self.channels = data.channels
        self.patches = split_patches_into_batch(self.image, patch_size=[4, 4], channels=data.channels, reversible=True)
        #self.patches_reversible = split_patches_into_batch(self.image, patch_size=[4, 4], channels=data.channels, reversible=True)
        #self.patches_size = int(np.prod(self.patches.get_shape()[1:]))
        #self.dropout_enabled = tf.placeholder_with_default(True, shape=())

        # Settings
        self.shape = [64, 128]
        self.defaults = {'activation_fn': tf.nn.selu, 'weights_initializer': lecun_normal()}

        # Network
        #self.latent = self._latent()
        #self.generator = self._generator()
        #self.generator_clipped = tf.clip_by_value(self.generator, 0, 1)
        #self.generator_optimize = self._generator_optimize()
        self.restored = op_join_patches_from_batch(self.patches, self.original_image_size, patch_size=[4, 4], channels=data.channels)
        #self.discriminator = self._discriminator()
        #self.optimize_discriminator = self._optimize_discriminator()

    @variable_scope('latent')
    def _latent(self):
        x = self.patches
        x = layers.fully_connected(x, self.shape[0], **self.defaults)
        x = layers.fully_connected(x, self.shape[1], **self.defaults)
        x = layers.fully_connected(x, self.shape[0], **self.defaults)
        return x

    @variable_scope('generator')
    def _generator(self):
        x = self.latent
        x = layers.fully_connected(x, self.patches_size, **self.defaults)
        return x
        # x = alpha_dropout_enabled(x, 0.8, self.dropout_enabled)
        # x = tf.random_uniform(tf.shape(x), 0.0, 1.0 / (2 * 255))
        #
        # x = layers.fully_connected(self.latent, self.shape[0], **self.defaults)
        # x = layers.fully_connected(x, self.shape[1], **self.defaults)
        # x = layers.fully_connected(x, self.shape[0], **self.defaults)
        # x = tf.reshape(x, [-1] + self.size_before_reshape)  # remove convolutions to the batch
        # x = layers.convolution2d_transpose(x, self.channels, **self.kernel, activation_fn=None)
        # return x

    @variable_scope
    def _generator_optimize(self):
        self.cross_entropy = tf.losses.mean_squared_error(self.patches, self.generator)
        return tf.train.AdamOptimizer().minimize(self.cross_entropy)

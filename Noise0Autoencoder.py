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
        self.learning_rate = 1e-4
        self.default_initializer = tf.contrib.layers.xavier_initializer()

        # Data
        self.patches = self.data.patches([7, 7], auxiliary_max_count=0)
        self.base_size = self.generator_size = self.patches.size

        # Network
        self.optimizers.append(self.optimize_generator)
        self.optimizers.append(self.optimize_discriminator)
        self.create_generator_summary()

    # Autoencoder: image -> latent + epsilon noise -> image

    @scope(cached_property=True)
    def input_normalized(self):
        return tf.cast(self.patches.data, tf.float32) / 255.0

    def dense(self, x, size, activation=tf.nn.elu):
        return tf.layers.dense(x, size, activation=activation, kernel_initializer=self.default_initializer)

    @scope(cached_property=True)
    def latent(self):
        x = self.input_normalized
        tf.summary.histogram('input', x)
        x = self.dense(x, self.base_size * 2)
        x = self.dense(x, self.base_size * 4)
        x = self.dense(x, self.base_size * 8)
        tf.summary.histogram('latent', x)
        return x, BaseNetwork.get_current_trainable_vars()

    @scope(cached_property=True)
    def generator(self):
        x = self.latent_minimum_noise
        x = self.dense(x, self.base_size * 8)
        x = self.dense(x, self.base_size * 4)
        x = self.dense(x, self.base_size * 2)
        x = self.dense(x, self.generator_size, activation=None)
        tf.summary.histogram('generator', x)
        return x

    @scope()
    def discriminator(self, x):
        x, _ = self.latent
        x = self.dense(x, self.base_size * 8)
        x = self.dense(x, self.base_size * 4)
        x = self.dense(x, self.base_size * 2)
        x = self.dense(x, 2, activation=None)
        return x, BaseNetwork.get_current_trainable_vars()

    @scope(cached_property=True)
    def latent_minimum_noise(self):
        latent, _ = self.latent
        return latent + tf.random_normal(tf.shape(latent), stddev=self.latent_stddev)

    def create_generator_summary(self):
        self.patches.restored_image_summary('restored', self.generator, 1)
        tf.summary.image('original', self.data.image, 1)

    @scope(cached_property=True)
    def optimize_generator(self):
        x = tf.losses.mean_squared_error(self.input_normalized, self.generator)
        tf.summary.scalar('cross entropy', x)
        return tf.train.AdamOptimizer(self.learning_rate).minimize(x)

    @scope(cached_property=True)
    def optimize_mmd(self):
        normalize = gather_batch(self.latent[0], 30)
        noise = tf.random_normal(tf.shape(normalize))
        tf.summary.histogram('target', noise)
        x = compute_mmd(normalize, noise)
        tf.summary.scalar('mmd', x)
        tf.summary.scalar('mmd baseline', compute_mmd(tf.random_normal(tf.shape(normalize)), noise))
        return tf.train.AdamOptimizer(self.learning_rate).minimize(x)

    @scope(cached_property=True)
    def optimize_discriminator(self):
        x1 = self.discriminator(self.latent[0])
        x2 = self.discriminator(tf.random_normal(tf.shape(self.latent[0])))
        x1_entropy = tf.losses.softmax_cross_entropy([1, 0], x1)
        x2_entropy = tf.losses.softmax_cross_entropy([0, 1], x2)
        tf.summary.scalar('discriminate', x1_entropy)
        return tf.train.AdamOptimizer(self.learning_rate).minimize(x1_entropy + x2_entropy)


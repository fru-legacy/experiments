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

        # Data
        self.patches = self.data.patches([28, 28], auxiliary_max_count=0)
        self.generator_size = self.patches.size
        self.base_pixel_count = int(np.prod(self.data.info.dim_image))

        # Network
        self.optimizers.append(self.optimize_generator)
        self.create_generator_summary()

    # Autoencoder: image -> latent + epsilon noise -> image

    @scope(cached_property=True)
    def input_normalized(self):
        return tf.cast(self.patches.data, tf.float32) / 255.0

    @scope(cached_property=True)
    def latent(self):
        x = self.input_normalized
        #x = self.fully_connected(x, 40)
        #x = self.fully_connected(x, 40)
        #x = self.fully_connected(x, 40)
        tf.summary.histogram('latent', x)
        return x

    @scope(cached_property=True)
    def generator(self):
        x = self.latent  # _minimum_noise
        #x = self.fully_connected(x, 40)
        #x = self.fully_connected(x, 40)
        #x = self.fully_connected(x, 40)
        x = self.fully_connected(x, self.generator_size, plain=True)
        tf.summary.histogram('generator', x)
        return x

    @scope(cached_property=True)
    def latent_minimum_noise(self):
        return self.latent + tf.random_normal(tf.shape(self.latent), stddev=self.latent_stddev)

    def create_generator_summary(self):
        self.patches.restored_image_summary('restored', self.generator, 1)
        tf.summary.image('original', self.data.image, 1)

    @scope(cached_property=True)
    def optimize_generator(self):
        x = tf.losses.mean_squared_error(self.input_normalized, self.generator)
        #y = compute_mmd(gather_batch(self.input_normalized, 30), gather_batch(self.generator, 30))
        tf.summary.scalar('cross entropy restore', x)
        #tf.summary.scalar('mmd', y)
        return tf.train.AdamOptimizer(self.learning_rate).minimize(x)

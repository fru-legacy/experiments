import tensorflow as tf
import numpy as np
from helper.BaseNetwork import BaseNetwork
from helper.decorators import scope
from helper.helper import repeat_constant_as_shape
from helper.flip_gradient import flip_gradient
from tested.mmd_kernel import compute_mmd


class Noise0Autoencoder(BaseNetwork):
    def __init__(self, data):
        super(Noise0Autoencoder, self).__init__()
        self.data = data
        self.base = int(np.prod(self.data.info.dim_image))
        #self.add_control_linear_regression('latent', self.latent)
        self.optimizers.append(self.generator_optimize)
        self.optimizers.append(self.generator_summary)

    # TODO: how will epsilon affect reconstruction
    epsilon = 1.0 / (255 * 8)

    # Autoencoder: image -> latent + epsilon noise -> image

    def add_control_linear_regression(self, name, latent):
        with tf.variable_scope('linear_regression_' + name):
            prediction = self.fully_connected(latent, self.data.info.label_count, plain=True)
            cross_entropy = tf.losses.softmax_cross_entropy(self.data.label_one_hot, prediction)
            tf.summary.scalar('linear regress ' + name, cross_entropy)
            in_scope = self.get_current_trainable_vars(expected_count=2)
            optimize = tf.train.AdamOptimizer().minimize(cross_entropy, var_list=in_scope)
            self.optimizers.append(optimize)

    @scope(cached_property=True)
    def latent(self):
        x = self.data.image_flat
        #x = self.fully_connected(x, self.base)
        #x = self.fully_connected(x, self.base * 2)
        #x = self.fully_connected(x, self.base * 2)
        #x = self.fully_connected(x, self.base * 2)
        #x = self.fully_connected(x, self.base * 2)
        return x

    @scope(cached_property=True)
    def latent_minimum_noise(self):
        x = self.latent
        #x = x + tf.random_normal(tf.shape(x), stddev=Noise0Autoencoder.epsilon)
        return x

    @scope(cached_property=True)
    def generator(self):
        x = self.latent_minimum_noise
        #x = self.fully_connected(x, self.base * 2)
        #x = self.fully_connected(x, self.base * 2)
        x = self.fully_connected(x, self.base)
        return x

    @scope(cached_property=True)
    def generator_summary(self):
        x = tf.reshape(self.generator, [-1] + self.data.info.dim_image)
        tf.summary.image('restored', tf.clip_by_value(x, 0, 1), 1)
        tf.summary.image('original', self.data.image, 1)
        return x

    @scope(cached_property=True)
    def generator_optimize(self):
        x = tf.reduce_mean(tf.abs(self.data.image_flat - self.generator))
        tf.summary.scalar('cross_entropy', x)
        return tf.train.AdamOptimizer(learning_rate=1e-8).minimize(x)

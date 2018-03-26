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
        self.image_normal_dims = [int(np.prod(self.data.info.dim_image)), 8, 2]
        self.image_normal_base = int(np.prod(self.image_normal_dims))
        #self.add_control_linear_regression('latent', self.latent)
        self.create_generator_summary()
        #tf.summary.scalar('linear regress', 0)
        self.optimizers.append(self.generator_optimize)

    # TODO: how will epsilon affect reconstruction
    epsilon = 1.0 / (255 * 8)

    @scope(cached_property=True)
    def image_normal_encoding(self, repeat=4):
        direction = tf.tile(self.data.image_byte, [1, 1, repeat]) * 2 - 1
        normal = tf.abs(tf.random_normal(tf.shape(direction)))
        return tf.layers.flatten(tf.multiply(normal, direction))

    def image_normal_cross_entropy(self, result):
        assert(result.get_shape().as_list() == [None, self.image_normal_base])
        one_hot_bytes = tf.one_hot(tf.cast(self.data.image_byte, tf.int32), 2)
        one_hot_result = tf.reshape(result, [-1] + self.image_normal_dims)
        return tf.losses.softmax_cross_entropy(one_hot_bytes, one_hot_result)

    def image_normal_restored(self, result):
        max = tf.cast(tf.argmax(tf.reshape(result, [-1] + self.image_normal_dims), -1), tf.uint8)
        return tf.reshape(Noise0Autoencoder.pack_bits_to_uint8(max), [-1] + self.data.info.dim_image)

    @staticmethod
    def pack_bits_to_uint8(values):
        assert(values.get_shape()[-1] == 8)
        bases = tf.cast(2 ** tf.range(8), tf.uint8)
        converted = tf.reduce_sum(tf.reshape(values, [-1, 8]) * bases, 1)
        print(values.get_shape(), values.get_shape().as_list()[1:-1])
        return tf.reshape(converted, [-1] + values.get_shape().as_list()[1:-1])

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
        x = self.image_normal_encoding
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
        x = self.fully_connected(x, self.image_normal_base)
        return x

    def create_generator_summary(self):
        tf.summary.image('restored', self.image_normal_restored(self.generator), 1)
        tf.summary.image('original', self.data.image, 1)

    @scope(cached_property=True)
    def generator_optimize(self):
        x = self.image_normal_cross_entropy(self.generator)
        tf.summary.scalar('cross_entropy', x)
        return tf.train.AdamOptimizer(learning_rate=1e-4).minimize(x)

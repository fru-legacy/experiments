import tensorflow as tf
import numpy as np
from helper.BaseNetwork import BaseNetwork
from helper.decorators import scope
from helper.helper import repeat_constant_as_shape
from helper.flip_gradient import flip_gradient
from tested.mmd_kernel import compute_mmd


class Noise0Autoencoder(BaseNetwork):
    def __init__(self, data):

        self.data = data
        super(Noise0Autoencoder, self).__init__()

        # Parameter

        self.latent_stddev = 1.0 / (255 * 8)
        self.learning_rate = 1e-5
        self.restore_compare_k = 1
        self.restore_stddev = 0.02
        self.restore_classes = 8
        self.restore_steps = 2

        # Helper

        self.base_pixel_count = int(np.prod(self.data.info.dim_image))
        self.generator_size = self.base_pixel_count * self.restore_classes

        # Network

        self.latent_input = tf.layers.flatten(self.data.image_byte * 2 - 1)
        self.create_generator_summary()
        self.optimizers.append(self.optimize_generator)

    # Selu needs stddev = 1 and mean = 0

    def image_normal_encoding_single(self, repeat=8):
        direction = tf.tile(self.data.image_byte, [1, 1, repeat]) * 2 - 1
        normal = tf.abs(tf.random_normal(tf.shape(direction)))
        return tf.layers.flatten(tf.multiply(normal, direction))

    @scope(cached_property=True)
    def latent_input_normalized_version1(self):
        e1 = self.image_normal_encoding_single()
        e2 = self.image_normal_encoding_single()
        return tf.concat([e1, -e2], axis=1)

    @scope(cached_property=True)
    def latent_input_normalized_version2(self):
        e1 = tf.random_normal(tf.shape(self.latent_input), mean=self.latent_input / 1.5, stddev=0.8)
        e2 = tf.random_normal(tf.shape(self.latent_input), mean=self.latent_input / 1.5, stddev=0.8)
        return tf.concat([e1, -e2], axis=1)

    @scope(cached_property=True)
    def latent_input_normalized_version3(self):
        return tf.concat([self.latent_input], axis=1)

    @scope(cached_property=True)
    def latent_input_normalized_base(self):
        return tf.random_normal(tf.shape(self.latent_input))

    # Log mean, variance and the histogram of a tensor

    @staticmethod
    def log_distribution(name, data):
        mean, variance = tf.nn.moments(data, axes=list(range(len(data.get_shape()))))
        tf.summary.scalar('mean ' + name, mean)
        tf.summary.scalar('variance ' + name, variance)
        tf.summary.histogram('normal' + name, data)

    # Restore

    def restore_image_tanh(self, x):
        baseline = tf.random_normal(tf.shape(x), stddev=self.restore_stddev)
        return tf.tanh((x - baseline) * self.restore_compare_k)

    def restore_image_raw(self, x):
        bases = tf.reverse(tf.cast(self.restore_steps ** tf.range(self.restore_classes), tf.float32), [0])
        return tf.reduce_sum(tf.reshape((x + 1) / 2, [-1, self.restore_classes]) * bases, 1)

    # Use this to control feature usability

    def add_control_classification(self, name, latent):
        with tf.variable_scope('control_classification_' + name):
            prediction = self.fully_connected(latent, self.data.info.label_count, plain=True)
            cross_entropy = tf.losses.softmax_cross_entropy(self.data.label_one_hot, prediction)
            tf.summary.scalar('control classification ' + name, cross_entropy)
            in_scope = self.get_current_trainable_vars(expected_count=2)
            optimize = tf.train.AdamOptimizer().minimize(cross_entropy, var_list=in_scope)
            self.optimizers.append(optimize)

    # Autoencoder: image -> latent + epsilon noise -> image

    @scope(cached_property=True)
    def latent(self):
        x = self.latent_input_normalized_base
        Noise0Autoencoder.log_distribution('input', x)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        x = self.fully_connected(x, self.generator_size // 40)
        Noise0Autoencoder.log_distribution('latent', x)
        x = self.fully_connected(x, self.generator_size // 4)
        x = self.fully_connected(x, self.generator_size // 4)
        return x

    @scope(cached_property=True)
    def generator(self):
        x = self.latent_minimum_noise
        x = self.fully_connected(x, self.generator_size // 4)
        x = self.fully_connected(x, self.generator_size // 4)
        return x

    @scope(cached_property=True)
    def latent_minimum_noise(self):
        return self.latent + tf.random_normal(tf.shape(self.latent), stddev=self.latent_stddev)

    @scope(cached_property=True)
    def restored(self):
        connected = self.fully_connected(self.generator, self.generator_size, plain=True)
        raw = self.restore_image_raw(self.restore_image_tanh(connected))
        return tf.reshape(raw, [-1] + self.data.info.dim_image)

    def create_generator_summary(self):
        tf.summary.image('restored', self.restored, 1)
        tf.summary.image('original', self.data.image, 1)

    @scope(cached_property=True)
    def optimize_generator(self):
        x = tf.losses.mean_squared_error(self.data.image, self.restored)
        tf.summary.scalar('cross_entropy', x)
        return tf.train.RMSPropOptimizer(self.learning_rate).minimize(x)

    @scope(cached_property=True)
    def optimize_labels(self):
        generated_labels = self.fully_connected(self.generator, 10, plain=True)
        x = tf.losses.softmax_cross_entropy(self.data.label_one_hot, generated_labels)
        tf.summary.scalar('cross_entropy', x)
        return tf.train.RMSPropOptimizer(self.learning_rate).minimize(x)

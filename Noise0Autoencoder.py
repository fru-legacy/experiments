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
        self.generator_size = self.base_pixel_count * self.restore_classes * 2

        # Network

        self.latent_input = tf.layers.flatten(self.data.image_byte * 2 - 1)
        self.create_generator_summary()
        self.optimizers.append(self.optimize_generator)

    # Selu needs stddev = 1 and mean = 0

    def image_normal_encoding_single(self, repeat):
        direction = tf.tile(self.data.image_byte, [1, 1, repeat]) * 2 - 1
        normal = tf.abs(tf.random_normal(tf.shape(direction)))
        return tf.multiply(normal, direction)

    @scope(cached_property=True)
    def latent_input_normalized(self):
        repeat = 1  # TODO Why does increasing repeat make latent variance != 1?
        e1 = tf.expand_dims(self.image_normal_encoding_single(repeat), -1)
        e2 = tf.expand_dims(self.image_normal_encoding_single(repeat), -1)
        return tf.reshape(tf.concat([e1, -e2], axis=-1), [-1, self.base_pixel_count, 8*2*repeat])

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

    def restore_bases(self):
        return [self.restore_steps ** float(x) for x in range(self.restore_classes)]

    def restore_image_tanh_v2(self, x):  # needs x2 outputs
        x = tf.reshape(x, [-1, self.base_pixel_count, self.restore_classes, 2])
        x = tf.reduce_sum(x * [1, -1], 3)
        x = (tf.tanh(x * self.restore_compare_k) + 1) / 2
        x = tf.reduce_sum(x * list(reversed(self.restore_bases())), 2)
        return x


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
        x = self.latent_input_normalized
        Noise0Autoencoder.log_distribution('input', x)
        x = self.fully_connected(x, 40)
        x = self.fully_connected(x, 40)
        x = self.fully_connected(x, 40)
        Noise0Autoencoder.log_distribution('latent', x)
        return x

    @scope(cached_property=True)
    def generator(self):
        x = self.latent_minimum_noise
        x = self.fully_connected(x, 40)
        x = self.fully_connected(x, 40)
        x = self.fully_connected(x, self.restore_classes * 2, plain=True)
        return tf.layers.flatten(x)

    @scope(cached_property=True)
    def latent_minimum_noise(self):
        return self.latent + tf.random_normal(tf.shape(self.latent), stddev=self.latent_stddev)

    @scope(cached_property=True)
    def restored(self):
        # If assert fails, add: self.fully_connected(generator, self.generator_size, plain=True)
        assert(self.generator.get_shape().as_list() == [None, self.generator_size])
        raw = self.restore_image_tanh_v2(self.generator)
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

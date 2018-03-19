import tensorflow as tf
from helper.BaseNetwork import BaseNetwork
from helper.decorators import scope
from helper.helper import repeat_constant_as_shape
from helper.flip_gradient import flip_gradient


class NoiseAutoencoder(BaseNetwork):
    def __init__(self, data, version):
        self.data = data
        self.image = data.image
        self.label = data.label_one_hot
        self.patches = data.patches([4, 4], auxiliary_max_count=2)
        super(NoiseAutoencoder, self).__init__(version)

    def get_optimizers(self):
        return tf.group(self.generator_optimize, self.discriminate_against_random, self.prediction_optimize)

    @scope(cached_property=True)
    def latent(self):
        x = self.patches.data
        x = self.fully_connected(x, self.shape[1])
        x = self.fully_connected(x, self.shape[2])
        x = self.fully_connected(x, self.shape[2])
        x = self.alpha_dropout(x, 0.8)
        x = self.fully_connected(x, self.shape[2])
        x = self.fully_connected(x, self.shape[0])
        return x

    @scope(cached_property=True)
    def latent_image(self):
        return tf.reshape(self.latent, [-1, self.shape[0], self.patches.count, 1])

    @scope(cached_property=True)
    def generator(self):
        x = self.latent
        x = x + tf.random_uniform(tf.shape(x), 0.0, 0.05)
        x = self.fully_connected(x, self.shape[2])
        x = self.fully_connected(x, self.patches.size)
        self.patches.restored_image_summary('restored', tf.clip_by_value(x, 0, 1))
        return x

    @scope(cached_property=True)
    def generator_optimize(self):
        cross_entropy = tf.losses.mean_squared_error(self.patches.data, self.generator)
        tf.summary.scalar('error', cross_entropy)
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    @scope(reuse=tf.AUTO_REUSE)
    def discriminate_single_target(self, latent, target):
        x = flip_gradient(latent)
        x = self.fully_connected(x, self.shape[1])
        x = self.fully_connected(x, self.shape[2])
        x = self.fully_connected(x, 1)
        target = repeat_constant_as_shape(target, x)
        cross_entropy = tf.losses.mean_squared_error(x, target)
        return tf.train.AdamOptimizer().minimize(cross_entropy)

    @scope(cached_property=True)
    def discriminate_against_random(self):
        latent = self.latent_image
        random = tf.random_uniform(tf.shape(latent), 0.0, 1.0)
        discriminator_random = self.discriminate_single_target(random, 0.0)
        discriminator_latent = self.discriminate_single_target(latent, 1.0)
        return tf.group(discriminator_random, discriminator_latent)

    @scope(cached_property=True)
    def prediction_optimize(self):
        x = tf.reshape(self.latent, [-1, self.patches.count * self.shape[0]])
        self.prediction = self.fully_connected(x, 10, plain=True)
        prediction_variables = tf.get_variable_scope().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        cross_entropy = tf.losses.softmax_cross_entropy(self.label, self.prediction)
        return tf.train.AdamOptimizer().minimize(cross_entropy, var_list=prediction_variables)

    @scope
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        tf.summary.scalar('error', tf.reduce_mean(tf.cast(mistakes, tf.float32)))

import tensorflow as tf
from helper.BaseNetwork import BaseNetwork
from helper.decorators import scope
from helper.helper import repeat_constant_as_shape
from helper.flip_gradient import flip_gradient


class NoiseAutoencoder(BaseNetwork):
    def __init__(self, data, version):
        self.data = data
        self.patches = data.patches([4, 4], auxiliary_max_count=2)
        super(NoiseAutoencoder, self).__init__(version)

    def get_optimizers(self):
        return tf.group(self.generator_optimize, self.discriminate_against_random, self.control_optimize)

    # Autoencoder: patches -> latent -> generator -> patches'

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

    # Discriminator: patches -> latent -> discriminate | random -> discriminate

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
        latent = self.patches.extract_patches_from_batch(self.latent)
        random = tf.random_uniform(tf.shape(latent), 0.0, 1.0)
        discriminator_random = self.discriminate_single_target(random, 0.0)
        discriminator_latent = self.discriminate_single_target(latent, 1.0)
        return tf.group(discriminator_random, discriminator_latent)

    # Control: patches -> latent -> regress | input -> regress

    def linear_regress_labels(self, name, latent):
        with tf.variable_scope(name):
            prediction = self.fully_connected(latent, 10, plain=True)
            cross_entropy = tf.losses.softmax_cross_entropy(self.data.label_one_hot, prediction)
            tf.summary.scalar('linear regress ' + name, cross_entropy)
            return tf.train.AdamOptimizer().minimize(cross_entropy, var_list=self.get_current_trainable_vars())

    @scope(cached_property=True)
    def control_optimize(self):
        patches_flat = tf.layers.flatten(self.patches.extract_patches_from_batch(self.patches.data))
        latent_flat = tf.layers.flatten(self.patches.extract_patches_from_batch(self.latent))

        control_patches = self.linear_regress_labels('patches', patches_flat)
        control_image = self.linear_regress_labels('input', tf.layers.flatten(self.data.image))
        control_latent = self.linear_regress_labels('latent', latent_flat)

        return tf.group(control_patches, control_image, control_latent)

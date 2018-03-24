import tensorflow as tf
from Noise0Autoencoder import Noise0Autoencoder
from tested.mmd_kernel import compute_mmd


class Noise1Autoencoder(Noise0Autoencoder):
    def __init__(self, data):
        super(Noise1Autoencoder, self).__init__(data)
        self.optimizers.append(self.enforce_normal_level_1(self.latent))
        self.optimizers.append(self.generator_optimize)

    # Autoencoder: image -> latent + epsilon noise -> image

    # TODO: how will epsilon affect reconstruction
    # TODO: what is the best layer to double_with_noise

    def double_with_noise(self, x):
        return tf.concat([x, tf.random_normal(tf.shape(x))], axis=1)

    def enforce_normal_level_1(self, latent):
        mmd = compute_mmd(latent, tf.random_normal(tf.shape(latent)))
        tf.summary.scalar('level 1', mmd)
        return tf.train.AdamOptimizer().minimize(mmd)

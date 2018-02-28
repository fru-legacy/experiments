import tensorflow as tf
from helper.decorators import variable_scope


class Encoder:
    def __init__(self, data):
        self.image = data.image_flat
        self.label = data.label

        self.prediction = self._prediction()
        self.optimize = self._optimize()
        self.error = self._error()

    @variable_scope(initializer=tf.contrib.layers.xavier_initializer())
    def _prediction(self):
        x = self.image
        x = tf.contrib.layers.fully_connected(x, 200)
        x = tf.contrib.layers.fully_connected(x, 200)
        x = tf.contrib.layers.fully_connected(x, 10, tf.nn.softmax)
        return x

    @variable_scope
    def _optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        return tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    @variable_scope
    def _error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

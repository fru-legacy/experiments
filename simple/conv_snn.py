import tensorflow as tf
from helper.decorators import variable_scope
from helper.alpha_dropout import alpha_dropout_enabled
import L4

class ConvSnn:
    def __init__(self, data):
        self.image = data.image
        self.label = data.label

        self.dropout_enabled = tf.placeholder_with_default(True, shape=())
        self.prediction = self._prediction()
        self.optimize = self._optimize()
        self.error = self._error()

    @variable_scope(initializer=tf.contrib.layers.xavier_initializer())
    def _prediction(self):
        lecun_norm = tf.keras.initializers.lecun_normal()
        x = self.image
        x = tf.contrib.layers.conv2d(x, 32, (3, 3), activation_fn=tf.nn.selu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.conv2d(x, 64, (3, 3), activation_fn=tf.nn.selu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.max_pool2d(x, (2, 2), stride=2)
        x = alpha_dropout_enabled(x, 0.25, self.dropout_enabled)
        x = tf.contrib.layers.flatten(x)
        x = tf.contrib.layers.fully_connected(x, 512, tf.nn.selu, weights_initializer=lecun_norm)
        x = alpha_dropout_enabled(x, 0.5, self.dropout_enabled)
        x = tf.contrib.layers.fully_connected(x, 10, activation_fn=None, weights_initializer=lecun_norm)
        return x

    @variable_scope
    def _optimize(self):
        self.cross_entropy = tf.losses.softmax_cross_entropy(self.label, self.prediction)
        return tf.train.AdamOptimizer().minimize(self.cross_entropy)
        #return L4.L4Adam(fraction=0.20).minimize(cross_entropy)

    @variable_scope
    def _error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

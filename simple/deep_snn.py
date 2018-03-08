import tensorflow as tf
from helper.decorators import variable_scope
from helper.alpha_dropout import alpha_dropout_enabled
from simple.conv_snn import ConvSnn
import L4
from tensorflow.python.ops.control_flow_ops import with_dependencies

class DeepSnn(ConvSnn):
    def __init__(self, data):
        super(DeepSnn, self).__init__(data)

    @variable_scope(initializer=tf.contrib.layers.xavier_initializer())
    def _prediction(self):
        lecun_norm = tf.keras.initializers.lecun_normal()
        x = self.image
        x = tf.contrib.layers.conv2d(x, 32, (3, 3), activation_fn=tf.nn.selu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.flatten(x)

        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        x = alpha_dropout_enabled(x, 0.75, self.dropout_enabled)

        skip = x
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        #x = skip + x
        x = alpha_dropout_enabled(x, 0.75, self.dropout_enabled)

        skip = x
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        #x = skip + x
        x = alpha_dropout_enabled(x, 0.75, self.dropout_enabled)

        skip = x
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        x = tf.contrib.layers.fully_connected(x, 1024, tf.nn.relu, weights_initializer=lecun_norm)
        #x = skip + x
        x = alpha_dropout_enabled(x, 0.75, self.dropout_enabled)

        x = tf.contrib.layers.fully_connected(x, 10, activation_fn=None, weights_initializer=lecun_norm)
        return x

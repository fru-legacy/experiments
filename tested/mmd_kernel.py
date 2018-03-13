import unittest
import tensorflow as tf


# Old code

def _old_compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def _old_compute_mmd(x, y):
    x_kernel = _old_compute_kernel(x, x)
    y_kernel = _old_compute_kernel(y, y)
    xy_kernel = _old_compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


# Refactored

def _mean_squared_difference(a, b):
    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)
    return tf.reduce_mean(tf.squared_difference(a, b), 2)


def _mean_kernel(a, b):
    dim = float(a.get_shape().as_list()[1])
    return tf.reduce_mean(tf.exp(-_mean_squared_difference(a, b) / dim))


def compute_mmd(x, y):
    return _mean_kernel(x, x) + _mean_kernel(y, y) - 2 * _mean_kernel(x, y)


# Test

class _Tests(tf.test.TestCase):

    def testEqualToOld(self):
        a = tf.random_normal([100, 100], mean=1.0, stddev=1.0)
        b = tf.random_normal([100, 100], mean=0.0, stddev=1.0)
        new = compute_mmd(a, b)
        old = _old_compute_mmd(a, b)
        with tf.Session() as session:
            new_val, old_val = session.run([new, old])
            self.assertEqual(new_val, old_val)


if __name__ == '__main__':
    unittest.main()

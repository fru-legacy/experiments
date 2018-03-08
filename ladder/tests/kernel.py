import unittest
import tensorflow as tf


class OriginalMmd:
    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, [x_size, 1, dim]), [1, y_size, 1]) # => x_size, y_size. dim
        tiled_y = tf.tile(tf.reshape(y, [1, y_size, dim]), [x_size, 1, 1]) # => x_size, y_size. dim
        return tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2)

    def compute_kernel_normalized(self, x, y):
        dim = tf.shape(x)[1]
        return tf.reduce_mean(tf.exp(-self.compute_kernel(x, y) / tf.cast(dim, tf.float32)))

    def compute_mmd(self, x, y, sigma_sqr=1.0):
        x_kernel = self.compute_kernel_normalized(x, x)
        y_kernel = self.compute_kernel_normalized(y, y)
        xy_kernel = self.compute_kernel_normalized(x, y)
        return x_kernel + y_kernel - 2 * xy_kernel


class TfSyntaxCleanup(OriginalMmd):
    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        x = tf.expand_dims(x, 1)
        y = tf.expand_dims(y, 0)
        return tf.reduce_mean(tf.squared_difference(x, y), 2)






class TestStringMethods(tf.test.TestCase):

    def testPerformance(self):
        with tf.Session() as session:
            a = tf.random_normal([100, 100], mean=1.0, stddev=1.0)
            b = tf.random_normal([100, 100], mean=0.0, stddev=1.0)
            test = TfSyntaxCleanup().compute_mmd(a, b)
            for _ in range(100):
                print(session.run(test))



if __name__ == '__main__':
    unittest.main()



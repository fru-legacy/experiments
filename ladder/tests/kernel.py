import unittest
import tensorflow as tf

class OriginalMmd:
    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        red = tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2)
        return tf.exp(-red / tf.cast(dim, tf.float32))

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


class Original2Mmd:
    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        x = tf.expand_dims(x, 1)
        y = tf.expand_dims(y, 0)
        #x = tf.tile(x, [1, y_size, 1]) # => x_size, y_size. dim
        #y = tf.tile(y, [x_size, 1, 1]) # => x_size, y_size. dim
        return tf.reduce_mean(tf.squared_difference(x, y), axis=2)

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

    def testEqual(self):
        a = tf.random_normal([100, 100], mean=1.0, stddev=1.0)
        b = tf.random_normal([100, 100], mean=0.0, stddev=1.0)
        test = OriginalMmd().compute_mmd(a, b)
        test2 = Original2Mmd().compute_mmd(a, b)
        with tf.Session() as session:
            self.assertEqual(0.0, session.run(test - test2))



if __name__ == '__main__':
    unittest.main()



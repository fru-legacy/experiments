import unittest
import tensorflow as tf


def gather_batch(x, count):
    shuffled_range = tf.random_shuffle(tf.range(tf.shape(x)[0]))
    indices = tf.slice(shuffled_range, [0], [count])
    return tf.gather(x, indices, axis=0)


class _Tests(tf.test.TestCase):

    def testEqualToOld(self):
        a = [[0], [1], [2], [3]]
        gathered = gather_batch(a, 2)

        with tf.Session() as session:
            gathered_val = session.run(gathered)
            self.assertEqual(len(gathered_val), 2)


if __name__ == '__main__':
    unittest.main()

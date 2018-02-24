import tensorflow as tf
from helper.decorators import variable_scope
from helper.datasets2d import load_fashion_mnist


class Model:
    def __init__(self, data):
        self.image = data.image_flat
        self.label = data.label

        self.prediction = self._prediction()
        self.optimize = self._optimize()
        self.error = self._error()

    @variable_scope(initializer=tf.contrib.slim.xavier_initializer())
    def _prediction(self):
        x = self.image
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
        return x

    @variable_scope
    def _optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @variable_scope
    def _error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def main():
    tf.set_random_seed(123)
    (data, feed) = load_fashion_mnist()
    model = Model(data)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for _ in range(10):
        error = sess.run(model.error, {**feed.testing()})
        print('Test error {:6.2f}%'.format(100 * error))
        for _ in range(60):
            sess.run(model.optimize, {**feed.next_batch(100)})


if __name__ == '__main__':
    main()

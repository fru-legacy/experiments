import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from helper.decorators import variable_scope


class Model:
    def __init__(self, image, label):
        self.image = image
        self.label = label

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
    mnist = input_data.read_data_sets('/data/MNIST-data', one_hot=True)
    # mnist.image
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    # mnist.next_batch
    def next_batch(size):
        images, labels = mnist.train.next_batch(size)
        return {image: images, label: labels}
    def testing():
        return {image: mnist.test.images, label: mnist.test.labels}

    model = Model(image, label)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for _ in range(10):
        error = sess.run(model.error, {**testing()})
        print('Test error {:6.2f}%'.format(100 * error))
        for _ in range(60):
            sess.run(model.optimize, {**next_batch(100)})


if __name__ == '__main__':
    main()

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from helper.decorators import property_scoped, property_initializer


class Model:
    @property_initializer
    def __init__(self, image, label):
        self.image = image
        self.label = label

    @property_scoped(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = self.image
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
        return x

    @property_scoped
    def optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        optimizer = tf.train.RMSPropOptimizer(0.03)
        return optimizer.minimize(cross_entropy)

    @property_scoped
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def main():
    mnist = input_data.read_data_sets('/data/MNIST-data', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = Model(image, label)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for _ in range(10):
        images, labels = mnist.test.images, mnist.test.labels
        error = sess.run(model.error, {image: images, label: labels})
        print('Test error {:6.2f}%'.format(100 * error))
        for _ in range(60):
            images, labels = mnist.train.next_batch(100)
            sess.run(model.optimize, {image: images, label: labels})


if __name__ == '__main__':
  main()
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# from tensorflow.python.keras.datasets import cifar10, cifar100
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape)


class _MnistLoader():
    def __init__(self, file):
        self._data = input_data.read_data_sets(file, one_hot=True)
        # np.savez_compressed('/data/fashion-mnist.npz',
        #                     x_train=self._data.train.images, y_train=self._data.train.labels,
        #                     x_test=self._data.test.images,   y_test=self._data.test.labels)
    def next_batch(self, size):
        return self._data.train.next_batch(size)

    def testing(self):
        return (self._data.test.images, self._data.test.labels)

    def loader(self, constant_tf_seed):
        return _Dataset2D(28, 28, 1, 10, self, constant_tf_seed)


class _Dataset2D():
    def __init__(self, width, height, channels, target, loader, constant_tf_seed):
        if constant_tf_seed:
            tf.set_random_seed(0)
            np.random.seed(0)

        self.width = width
        self.height = height
        self.target = target
        self.channels = channels
        self.dim_image = [width, height, channels]
        self.dim_flat = np.prod(self.dim_image)

        self.image_flat = tf.placeholder(tf.float32, [None, self.dim_flat])
        self.image = tf.reshape(self.image_flat, [-1] + self.dim_image)
        self.label = tf.placeholder(tf.float32, [None, target])

        self._loader = loader

    def next_batch(self, size):
        images, labels = self._loader.next_batch(size)
        return {self.image_flat: images, self.label: labels}

    def testing(self, transform=None):
        full = self._loader.testing()
        if transform:
            full = transform(full)
        images, labels = full
        return {self.image_flat: images, self.label: labels}

    def testing_split(self, count):
        images, labels = self._loader.testing()
        images_split = np.split(images, count)
        labels_split = np.split(labels, count)
        return [{self.image_flat: images_split[i], self.label: labels_split[i]} for i in range(count)]


def load_mnist(constant_tf_seed):
    return _MnistLoader('/data/mnist').loader(constant_tf_seed)

def load_fashion_mnist(constant_tf_seed):
    return _MnistLoader('/data/fashion-mnist').loader(constant_tf_seed)

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class _MnistLoader():
    def __init__(self, file, width, height, channels, target):
        self._data = input_data.read_data_sets(file, one_hot=True, seed=123)
        self._set = _Dataset2D(width, height, channels, target)

    def next_batch(self, size):
        images, labels = self._data.train.next_batch(size)
        return {self._set.image_flat: images, self._set.label: labels}

    def testing(self):
        return {self._set.image_flat: self._data.test.images, self._set.label: self._data.test.labels}

    def loader(self):
        return (self._set, self)


class _Dataset2D():
    def __init__(self, width, height, channels, target):
        self.width = width;
        self.height = height;
        self.target = target;
        self.dim_image = [width, height, channels]
        self.dim_flat = np.prod(self.dim_image)

        self.image_flat = tf.placeholder(tf.float32, [None, self.dim_flat])
        self.image = tf.reshape(self.image_flat, [-1] + self.dim_image)
        self.label = tf.placeholder(tf.float32, [None, target])


load_mnist = lambda: _MnistLoader('/data/MNIST-data', 28, 28, 1, 10).loader()
load_fashion_mnist = lambda: _MnistLoader('/data/FASHION-MNIST-data', 28, 28, 1, 10).loader()
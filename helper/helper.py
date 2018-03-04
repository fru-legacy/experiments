import os
import tensorflow as tf
import numpy as np
from functools import partial
import math


def get_initialized_session(disable_gpu=False):
    if disable_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    return session


def next_batch(num, full):
    """
    Return a total of `num` random samples and labels.
    """
    data, labels = full
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    return data[idx], labels[idx]


def next_batch_curry(num):
    return partial(next_batch, num)


def tests(count, full):

    data, labels = full
    len(data)
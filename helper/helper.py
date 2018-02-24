import os
import tensorflow as tf
import numpy as np


def use_constant_seed_disable_gpu(disable):
    if disable:
        tf.set_random_seed(0)
        np.random.seed(0)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

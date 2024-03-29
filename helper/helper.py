import os
import tensorflow as tf
import numpy as np
from functools import partial
import math
from tensorflow.python import debug as tf_debug
import argparse


def get_docker_args():
    parser = argparse.ArgumentParser(description="Experiment")
    parser.add_argument("--debug", action='store_true')
    args, _ = parser.parse_known_args()
    return args


def get_initialized_session(disable_gpu=False):
    if disable_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    if get_docker_args().debug:
        session = tf_debug.TensorBoardDebugWrapperSession(session, "0.0.0.0:6064")
    return session


def repeat_constant_as_shape(constant, shape):
    dims = np.repeat(1, len(shape.get_shape()))
    return tf.tile(tf.reshape(constant, dims), tf.shape(shape))
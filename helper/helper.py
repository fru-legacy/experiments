import os
import tensorflow as tf


def get_initialized_session(disable_gpu=False):
    if disable_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    return session
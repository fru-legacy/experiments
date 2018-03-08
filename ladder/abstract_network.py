import tensorflow as tf
import numpy as np
import math
import glob
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from tensorflow.examples.tutorials.mnist import input_data
import os, sys, shutil, re

def lrelu(x, rate=0.1):
    # return tf.nn.relu(x)
    return tf.maximum(tf.minimum(x * rate, 0), x)

conv2d = tf.contrib.layers.convolution2d
conv2d_t = tf.contrib.layers.convolution2d_transpose
fc_layer = tf.contrib.layers.fully_connected


def conv2d_bn_lrelu(inputs, num_outputs, kernel_size, stride, is_training=True):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv, is_training=is_training)
    conv = lrelu(conv)
    return conv


def conv2d_t_bn_relu(inputs, num_outputs, kernel_size, stride, is_training=True):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv, is_training=is_training)
    conv = lrelu(conv)
    return conv


def conv2d_t_bn(inputs, num_outputs, kernel_size, stride, is_training=True):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                                     activation_fn=tf.identity, scope=None)
    conv = tf.contrib.layers.batch_norm(conv, is_training=is_training)
    return conv


def fc_bn_lrelu(inputs, num_outputs, is_training=True):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc, is_training=is_training)
    fc = lrelu(fc)
    return fc


def fc_bn_relu(inputs, num_outputs, is_training=True):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc, is_training=is_training)
    fc = tf.nn.relu(fc)
    return fc


def _mean_squared_difference(a, b):
    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)
    return tf.reduce_mean(tf.squared_difference(a, b), 2)


def _mean_kernel(a, b=None):
    dim = float(a.get_shape()[1])
    return tf.reduce_mean(tf.exp(-_mean_squared_difference(a, b or a) / dim))


def compute_mmd(x, y):
    return _mean_kernel(x) + _mean_kernel(y) - 2 * _mean_kernel(x, y)


class Network:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name="lr_placeholder")

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        # A unique name should be given to each instance of subclasses during initialization
        self.name = "default"

        # These should be updated accordingly
        self.iteration = 0
        self.learning_rate = 0.0
        self.read_only = False

        self.do_generate_samples = False
        self.do_generate_conditional_samples = False
        self.do_generate_manifold_samples = False

    def make_model_path(self):
        if not os.path.isdir("models"):
            os.mkdir("models")
        if not os.path.isdir("models/" + self.name):
            os.mkdir("models/" + self.name)

    def print_network(self):
        self.make_model_path()
        if os.path.isdir("models/" + self.name):
            for f in os.listdir("models/" + self.name):
                if re.search(r"events.out*", f):
                    os.remove(os.path.join("models/" + self.name, f))
        self.writer = tf.summary.FileWriter("models/" + self.name, self.sess.graph)
        self.writer.flush()

    """ Save network, if network file already exists back it up to models/old folder. Only one back up will be created
    for each network """
    def save_network(self):
        if not self.read_only:
            # Saver and Summary ops cannot run in GPU
            with tf.device('/cpu:0'):
                saver = tf.train.Saver()
            self.make_model_path()
            if not os.path.isdir("models/old"):
                os.mkdir("models/old")
            file_name = "models/" + self.name + "/" + self.name + ".ckpt"
            if os.path.isfile(file_name):
                os.rename(file_name, "models/old/" + self.name + ".ckpt")
            saver.save(self.sess, file_name)

    """ Either initialize or load network from file.
    Always run this at end of initialization for every subclass to initialize Variables properly """
    def init_network(self, restart=False):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if restart:
            return
        file_name = "models/" + self.name + "/" + self.name + ".ckpt"
        if len(glob.glob(file_name + '*')) != 0:
            saver = tf.train.Saver()
            try:
                saver.restore(self.sess, file_name)
                print("Successfully restored model")
            except:
                print("Warning: network load failed, reinitializing all variables", sys.exc_info()[0])
                self.sess.run(tf.global_variables_initializer())
        else:
            print("No checkpoint file found, Initializing model from random")

    """ This function should train on the given batch and return the training loss """
    def train(self, batch_input, batch_target, labels=None):
        return None

    """ This function should take the input and return the reconstructed images """
    def test(self, batch_input, labels=None):
        return None


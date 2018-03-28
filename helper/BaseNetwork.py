import tensorflow as tf
import numpy as np
import os
from helper.selu import selu, initializer, dropout_selu
from tensorflow.core.util.event_pb2 import SessionLog
from helper.helper import get_initialized_session


class BaseNetwork:
    def __init__(self):
        np.random.seed(0)
        tf.set_random_seed(0)

        self.shape = [12, 64, 128]
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.log_name = 'log-' + self.__class__.__name__

        self.has_run_initialized = False
        self.execute = None
        self.session = None
        self.summary = None
        self.summary_writer = None
        self.saver = None
        self.saver_path = None
        self.optimizers = []

    def get_optimizers(self):
        return self.optimizers

    def alpha_dropout(self, x, keep_prob):
        return dropout_selu(x, keep_prob, training=self.is_training)

    def fully_connected(self, x, size, plain=False):
        if plain:
            return tf.layers.dense(x, size)

        return tf.layers.dense(x, size, activation=selu, kernel_initializer=initializer)
        defaults = {'activation_fn': selu, 'weights_initializer': initializer}
        return tf.contrib.layers.fully_connected(x, size, **defaults) #biases_initializer=None

    def get_current_trainable_vars(self, expected_count=None):
        prefix = tf.get_variable_scope().name
        result = [v for v in tf.trainable_variables() if v.name.startswith(prefix)]
        assert expected_count is None or len(result) == expected_count
        return result

    def run(self, data_generator, iteration, steps, is_training):
        if not self.has_run_initialized:
            self.has_run_initialized = True
            self.execute = tf.group(*self.get_optimizers())
            self.summary = tf.summary.merge_all()
            self.session = get_initialized_session(disable_gpu=False)
            self.summary_writer = tf.summary.FileWriter('./log/' + self.log_name, self.session.graph)
            #self.saver = tf.train.Saver()
            self.saver_path = './log/checkpoint_' + self.log_name

            if os.path.exists(self.saver_path):
                self.saver.restore(self.session, self.saver_path + '/model.ckpt')
            else:
                self.summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)

        for e in range(steps):
            data = data_generator()
            log, _ = self.session.run([self.summary, self.execute], {**data, self.is_training: is_training})
            self.summary_writer.add_summary(log, iteration+e)
            #self.saver.save(self.session, self.saver_path + '/model.ckpt')
            print('Iteration', e)

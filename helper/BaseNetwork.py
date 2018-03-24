import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.api.keras.initializers import lecun_normal
from helper.alpha_dropout import alpha_dropout
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
        self.optimizers = []

    def get_optimizers(self):
        return self.optimizers

    def alpha_dropout(self, x, keep_prob):
        keep_prob = tf.where(self.is_training, keep_prob, 1.0)
        return alpha_dropout(x, keep_prob)

    def fully_connected(self, x, size, plain=False):
        if plain:
            return tf.layers.dense(x, size)

        defaults = {'activation_fn': tf.nn.selu, 'weights_initializer': lecun_normal()}
        return tf.contrib.layers.fully_connected(x, size, **defaults)

    def get_current_trainable_vars(self, expected_count):
        prefix = tf.get_variable_scope().name
        result = [v for v in tf.trainable_variables() if v.name.startswith(prefix)]
        assert len(result) == expected_count
        return result

    def run(self, data_generator, iteration, steps, is_training):
        if not self.has_run_initialized:
            self.execute = self.get_optimizers()
            self.summary = tf.summary.merge_all()
            self.session = get_initialized_session(disable_gpu=False)
            self.summary_writer = tf.summary.FileWriter('./log/' + self.log_name, self.session.graph)
            self.summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)
            self.has_run_initialized = True

        for e in range(steps):
            data = data_generator()
            log, _ = self.session.run([self.summary, self.execute], {**data, self.is_training: is_training})
            self.summary_writer.add_summary(log, iteration+e)
            print('Iteration', e)





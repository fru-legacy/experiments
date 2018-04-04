import tensorflow as tf
import numpy as np
from tensorflow.core.util.event_pb2 import SessionLog
from helper.helper import get_initialized_session
from cached_property import cached_property


class BaseNetwork:
    def __init__(self, data):
        np.random.seed(0)
        tf.set_random_seed(0)

        self.data = data
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.log_name = 'log-' + self.__class__.__name__
        self.optimizers = []
        self.iteration = 0

    @staticmethod
    def get_current_trainable_vars(expected_count=None):
        prefix = tf.get_variable_scope().name
        result = [v for v in tf.trainable_variables() if v.name.startswith(prefix)]
        assert expected_count is None or len(result) == expected_count
        return result

    @cached_property
    def config(self, disable_gpu=False):
        base = self

        class RunConfig:
            def __init__(self):
                self.summary = tf.summary.merge_all()
                self.run = lambda: [self.summary, base.optimizers]
                self.session = get_initialized_session(disable_gpu)
                self.summary_writer = tf.summary.FileWriter('./log/' + base.log_name, self.session.graph)
                self.summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)
                self.session.graph.finalize()

        return RunConfig()

    def run(self, data_generator, steps, is_training):
        for e in range(steps):
            log, _ = self.config.session.run(self.config.run(), {**data_generator(), self.is_training: is_training})
            self.iteration += 1
            self.config.summary_writer.add_summary(log, self.iteration)
            print('Iteration', self.iteration)

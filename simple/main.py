import tensorflow as tf
import numpy as np

from helper.datasets2d import load_fashion_mnist
from helper.helper import get_initialized_session, next_batch_curry
from simple.conv_snn import ConvSnn
from simple.deep_snn import DeepSnn
from tensorflow.core.util.event_pb2 import SessionLog
from tensorboard.version import VERSION

data = load_fashion_mnist(constant_tf_seed=True)
model = DeepSnn(data)

#tf.summary.image('input', data.image, 1)
tf.summary.scalar('error', model.error)
tf.summary.tensor_summary('prediction', model.prediction)
summary = tf.summary.merge_all()

session = get_initialized_session(disable_gpu=False)

summary_writer = tf.summary.FileWriter('../log/log-simple-05-03-2017-l4', session.graph)
summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)

for t in range(5):
    for e in range(60000 // 128):
        _, en = session.run([model.optimize, model.cross_entropy], {**data.next_batch(128)})
        print(en)

    tests = data.testing_split(20)
    errors = []
    for i in range(20):
        error, log = session.run([model.error, summary], {**tests[i], model.dropout_enabled: False})
        if i == 0:
            summary_writer.add_summary(log, t)
        errors.append(error)
    print(errors)
    print(np.mean(errors))

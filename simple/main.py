import tensorflow as tf
import numpy as np

from helper.datasets2d import load_fashion_mnist
from helper.helper import get_initialized_session, next_batch_curry
from simple.conv_snn import ConvSnn
from tensorflow.core.util.event_pb2 import SessionLog


data = load_fashion_mnist(constant_tf_seed=True)
model = ConvSnn(data)

#tf.summary.image('input', data.image, 1)
tf.summary.scalar('error', model.error)
summary = tf.summary.merge_all()

session = get_initialized_session(disable_gpu=False)
summary_writer = tf.summary.FileWriter('log', session.graph)
summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)

for t in range(1):
    for e in range(60000 // 128):
        error, _ = session.run([model.error, model.optimize], {**data.next_batch(128)})
        print(error)

    tests = data.testing_split(20)
    errors = []
    for i in range(20):
        error, log = session.run([model.error, summary], {**tests[i]})
        if i == 0:
            summary_writer.add_summary(log, t)
        errors.append(error)
    print(errors)

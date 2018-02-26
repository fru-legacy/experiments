import tensorflow as tf
from helper.datasets2d import load_fashion_mnist
from helper.helper import get_initialized_session
from simple.fully_connected_to_label import FullyConnectedToLabel
from tensorflow.core.util.event_pb2 import SessionLog

data = load_fashion_mnist(constant_tf_seed=True)
model = FullyConnectedToLabel(data)

tf.summary.image('input', data.image, 1)
tf.summary.scalar('error', model.error)
summary = tf.summary.merge_all()

session = get_initialized_session(disable_gpu=True)
summary_writer = tf.summary.FileWriter('log', session.graph)
summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)

for t in range(10):
    epoch_size = 60
    for e in range(epoch_size):
        _, log = session.run([model.optimize, summary], {**data.next_batch(100)})
        summary_writer.add_summary(log, t * epoch_size + e)

    error = session.run(model.error, {**data.testing()})
    print('Test error {:6.2f}%'.format(100 * error))

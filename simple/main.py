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

grads_and_vars = model.optimizer.compute_gradients(model.cross_entropy)
optimize = model.optimizer.apply_gradients(grads_and_vars)

for t in range(55):
    epoch_size = 60
    for e in range(epoch_size):
        _, log = session.run([optimize, summary], {**data.next_batch(100)})
        #summary_writer.add_summary(log, t * epoch_size + e)

    error, log = session.run([model.error, summary], {**data.testing()})
    summary_writer.add_summary(log, t)
    print('Test error {:6.2f}%'.format(100 * error))

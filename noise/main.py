import tensorflow as tf
import numpy as np

from helper.datasets2d import load_fashion_mnist
from helper.helper import get_initialized_session, next_batch_curry
from noise.noise_autoencoder import NoiseAutoencoder
from tensorflow.core.util.event_pb2 import SessionLog

data = load_fashion_mnist(constant_tf_seed=True)
model = NoiseAutoencoder(data)

tf.summary.image('original', data.image, 1)
tf.summary.image('test_deconv', model.test_deconv, 1)
tf.summary.image('generated', model.generator_clipped, 1)
tf.summary.scalar('error', model.cross_entropy)
summary = tf.summary.merge_all()

session = get_initialized_session(disable_gpu=False)

summary_writer = tf.summary.FileWriter('../log/log-noise-09-03-2018', session.graph)
summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)

for e in range(6000):
    log, _ = session.run([summary, model.generator_optimize], {**data.next_batch(128)})
    summary_writer.add_summary(log, e)
    print(e)

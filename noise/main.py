import tensorflow as tf
import numpy as np

from helper.datasets2d import load_fashion_mnist
from helper.helper import get_initialized_session, next_batch_curry
from noise.noise_autoencoder import NoiseAutoencoder
from tensorflow.core.util.event_pb2 import SessionLog
#from tensorboard.plugins.beholder.beholder import Beholder

data = load_fashion_mnist(constant_tf_seed=True)
model = NoiseAutoencoder(data)

tf.summary.image('original', data.image, 1)
tf.summary.image('generated', model.restored, 1)
tf.summary.image('latent', model.latent_image, 1)
#tf.summary.image('random', model.latent_random_image, 1)
tf.summary.scalar('error', model.cross_entropy)
tf.summary.scalar('prediction', model.prediction_error)
summary = tf.summary.merge_all()

session = get_initialized_session(disable_gpu=False)

summary_writer = tf.summary.FileWriter('../log/log-noise-09-03-2018', session.graph)
summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)

for e in range(60000):
    batch = data.next_batch(128)
    session.run(model.discriminator_latent, {**batch})
    session.run(model.discriminator_random, {**batch})
    session.run([model.prediction_optimize], {**batch})
    session.run([summary, model.generator_optimize], {**batch})
    session.run([summary, model.generator_optimize], {**batch})
    log, _ = session.run([summary, model.generator_optimize], {**batch})
    summary_writer.add_summary(log, e)
    print(e)

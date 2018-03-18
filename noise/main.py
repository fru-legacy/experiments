import tensorflow as tf
import numpy as np

from helper.helper import get_initialized_session
from noise.noise_autoencoder import NoiseAutoencoder
from tensorflow.core.util.event_pb2 import SessionLog
from tensorboard.plugins.beholder.beholder import Beholder

# Only for module testing purposes
import sys
sys.path.insert(0, '/github/tf-data')
import tf_data

np.random.seed(0)
tf.set_random_seed(0)

data = tf_data.mnist('/data/tf-data').placeholder()
model = NoiseAutoencoder(data)

tf.summary.image('original', data.image, 1)
tf.summary.image('latent', model.latent_image, 1)
tf.summary.scalar('prediction', model.prediction_error)

execute = tf.group(*[model.generator_optimize, model.discriminator_latent, model.discriminator_random, model.prediction_optimize])
summary = tf.summary.merge_all()
session = get_initialized_session(disable_gpu=False)

summary_writer = tf.summary.FileWriter('../log/log-noise-09-03-2018', session.graph)
summary_writer.add_session_log(SessionLog(status=SessionLog.START), 0)

for e in range(40):
    train = data.train(batch_size=128)
    log, _ = session.run([summary, execute], {**train})
    summary_writer.add_summary(log, e)
    print(e)

# Only for module testing purposes
import sys
sys.path.insert(0, '/github/tf-data')
import tf_data
from Noise0Autoencoder import Noise0Autoencoder

data = tf_data.fashion_mnist('/data/tf-data').placeholder()
model = Noise0Autoencoder(data)

model.run(lambda: data.train(batch_size=128), 0, steps=1, is_training=True)

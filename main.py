# Only for module testing purposes
import sys
sys.path.insert(0, '/github/tf-data')

import tf_data
from NoiseAutoencoder import NoiseAutoencoder

data = tf_data.mnist('/data/tf-data').placeholder()
model = NoiseAutoencoder(data, 1)
model.run(lambda: data.train(batch_size=128), 0, steps=4000, is_training=True)

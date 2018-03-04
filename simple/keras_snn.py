# Adapted KERAS tutorial 

from __future__ import print_function
from tensorflow.python import keras
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, AlphaDropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import numpy as np


batch_size = 128
num_classes = 10
epochs = 1

tensorboard = keras.callbacks.TensorBoard(log_dir='../log', histogram_freq=0, batch_size=batch_size,
                                          write_graph=True, write_grads=False, write_images=False)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
f = np.load('/data/fashion-mnist.npz')
x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
f.close()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='selu',
                 input_shape=input_shape,kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Conv2D(64, (3, 3), activation='selu',kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AlphaDropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='selu',kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(AlphaDropout(0.5))
model.add(Dense(num_classes, activation='softmax',kernel_initializer='lecun_normal',bias_initializer='zeros'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Activation, Softmax, LeakyReLU

import math

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import Model, Sequential
from keras.datasets import mnist
from keras.utils import np_utils
import WaveletLayer as wl
from update_batch_size import update_batch_size

from keras import backend as K
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_haar_matrix(row_size):
    size = (int)(row_size / 2)
    haarM1 = np.zeros((size, row_size))
    for i in range(size):
        haarM1[i][2 * i] = math.sqrt(2) / 2
        haarM1[i][2 * i + 1] = math.sqrt(2) / 2

    haarM2 = np.zeros((size, row_size))
    for i in range(size):
        haarM2[i][2 * i] = math.sqrt(2) / 2
        haarM2[i][2 * i + 1] = -math.sqrt(2) / 2

    m = np.concatenate((haarM1, haarM2), axis=0)
    return m

batch_size = K.variable(32)

haarMatrix28 = build_haar_matrix(28)
haarMatrix14 = build_haar_matrix(14)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

number_of_classes = 10

y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)


input = Input(shape=(28, 28, 1))
conv1 = Conv2D(20, kernel_size=(5,5), padding='same')(input)
batch1 = BatchNormalization()(conv1)
#pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)
pool1 = wl.MyLayer(output_dim=(None, 14, 14, 20), haar_matrix=haarMatrix28)(batch1)

conv2 = Conv2D(50, kernel_size=(5,5), padding="same")(pool1)

#pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

pool2 = wl.MyLayer(output_dim=(None, 7, 7, 50), haar_matrix=haarMatrix14)(conv2)

batch2 = BatchNormalization()(pool2)

conv3 = Conv2D(500, kernel_size=(4,4))(batch2)
relu = LeakyReLU(alpha=0)(conv3)
conv4 = Conv2D(10, kernel_size=(1, 1))(relu)
batch3 = BatchNormalization()(conv4)
flat = Flatten()(batch3)
activ = Dense(units=10, activation='softmax')(flat)
model = Model(inputs=input, outputs=activ)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, batch_size=32)

K.set_value(batch_size, 128)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)



print(loss_and_metrics)
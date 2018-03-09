from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Activation, Softmax, LeakyReLU

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import Model, Sequential
from keras.datasets import mnist
from keras.utils import np_utils

from keras import backend as K
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# mnist = tf.contrib.learn.datasets.load_dataset("mnist")
# x_train = mnist.train.images # Returns np.array
# x_train = np.vsplit(x_train, 1)
# x_train = np.array([np.reshape(x_train[0][i], (28, 28)) for i in range(x_train[0].shape[0])])
# x_train = np.array([np.reshape(x_train[i], (28, 28, 1)) for i in range(x_train.shape[0])])
# print(x_train[4].shape)
# y_train = np.asarray(mnist.train.labels, dtype=np.int32)
# print(y_train.shape)
# x_test = mnist.test.images # Returns np.array
# y_test = np.asarray(mnist.test.labels, dtype=np.int32)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

number_of_classes = 10

y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)

print(y_train)

input = Input(shape=(28, 28, 1))
conv1 = Conv2D(20, kernel_size=(5,5))(input)
batch1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)
conv2 = Conv2D(50, kernel_size=(5,5))(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
batch2 = BatchNormalization()(pool2)
conv3 = Conv2D(500, kernel_size=(4,4))(batch2)
relu = LeakyReLU(alpha=0)(conv3)
conv4 = Conv2D(10, kernel_size=(1, 1))(relu)
batch3 = BatchNormalization()(conv4)
#conv5 = Conv2D(1, kernel_size=(1, 1), activation='softmax')(batch3)
flat = Flatten()(batch3)
activ = Dense(units=10, activation='softmax')(flat)
model = Model(inputs=input, outputs=activ)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, batch_size=32)


# 9. Fit model on training data
# model.fit(x_train, y_train,
#           batch_size=32, nb_epoch=10, verbose=1)

# 10. Evaluate model on test data
#score = model.evaluate(x_test, y, verbose=0)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

#print(score)
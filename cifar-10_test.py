import keras
from keras.datasets import cifar10
import os
import math
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Activation, Softmax, LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import Model, Sequential
from keras.datasets import mnist
from keras.utils import np_utils
import WaveletLayer as wl
from update_batch_size import update_batch_size

from keras import backend as K
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

haarMatrix32 = build_haar_matrix(32)
haarMatrix16 = build_haar_matrix(16)
haarMatrix8 = build_haar_matrix(8)

num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input = Input(shape=(32, 32, 3))

conv1 = Conv2D(32, kernel_size=(5,5), padding='same')(input)
batch1 = BatchNormalization()(conv1)
relu1 = LeakyReLU(alpha=0)(batch1)
pool1 = wl.MyLayer(output_dim=(None, 16, 16, 32), haar_matrix=haarMatrix32)(relu1)

conv2 = Conv2D(32, kernel_size=(5,5), padding='same')(pool1)
batch2 = BatchNormalization()(conv2)
relu2 = LeakyReLU(alpha=0)(batch2)
pool2 = wl.MyLayer(output_dim=(None, 8, 8, 32), haar_matrix=haarMatrix16)(relu2)


conv3 = Conv2D(32, kernel_size=(5,5), padding='same')(pool2)
batch3 = BatchNormalization()(conv3)
relu3 = LeakyReLU(alpha=0)(batch3)
pool3 = wl.MyLayer(output_dim=(None, 4, 4, 32), haar_matrix=haarMatrix8)(relu3)

drop = Dropout(rate=0.1)(pool3)
conv4 = Conv2D(64, kernel_size=(4,4), padding='same')(drop)
relu3 = LeakyReLU(alpha=0)(conv4)
drop2 = Dropout(rate=0.1)(relu3)
conv5 = Conv2D(10, kernel_size=(1,1), padding='same')(drop2)
flat = Flatten()(conv5)
activ = Dense(units=10, activation='softmax')(flat)
model = Model(inputs=input, outputs=activ)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=2, batch_size=32)


loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)



print(loss_and_metrics)
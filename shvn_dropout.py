import numpy as np
import scipy.io as sio
import keras
import os
import math
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU
from keras.layers import Dense, Flatten, Dropout
from keras import Model
import WaveletLayer as wl

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

def one_hot_encode(data, n_classes):
    n = data.shape[0]
    one_hot = np.zeros(shape=(data.shape[0], n_classes))
    for s in range(n):
        temp = np.zeros(n_classes)

        num = data[s][0]
        if num == 10:
            temp[0] = 1
        else:
            temp[num] = 1

        one_hot[s] = temp

    return one_hot


haarMatrix32 = build_haar_matrix(32)
haarMatrix16 = build_haar_matrix(16)
haarMatrix8 = build_haar_matrix(8)



train_data = sio.loadmat('train_32x32.mat')
test_data = sio.loadmat('test_32x32.mat')
num_classes = 10
# access to the dict
x_train = train_data['X'][:,:,:,:55000]
y_train = train_data['y'][:55000]

x_test = test_data['X']
y_test = test_data['y']


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int')
y_test = y_test.astype('int')

x_train = np.transpose(x_train, (3, 0, 1, 2))
x_test = np.transpose(x_test, (3, 0, 1, 2))

print(x_train.shape)
print(type(y_train[0]))


# Convert class vectors to binary class matrices.
y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)


input = Input(shape=(32, 32, 3))

conv1 = Conv2D(64, kernel_size=(5,5), padding='same')(input)
batch1 = BatchNormalization()(conv1)
relu1 = LeakyReLU(alpha=0)(batch1)
pool1 = wl.MyLayer(output_dim=(None, 16, 16, 64), haar_matrix=haarMatrix32)(relu1)

conv2 = Conv2D(64, kernel_size=(5,5), padding='same')(pool1)
batch2 = BatchNormalization()(conv2)
relu2 = LeakyReLU(alpha=0)(batch2)
pool2 = wl.MyLayer(output_dim=(None, 8, 8, 64), haar_matrix=haarMatrix16)(relu2)


conv3 = Conv2D(64, kernel_size=(5,5), padding='same')(pool2)
batch3 = BatchNormalization()(conv3)
relu3 = LeakyReLU(alpha=0)(batch3)
pool3 = wl.MyLayer(output_dim=(None, 4, 4, 64), haar_matrix=haarMatrix8)(relu3)

drop = Dropout(rate=0.1)(pool3)
conv4 = Conv2D(128, kernel_size=(4,4), padding='same')(drop)
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



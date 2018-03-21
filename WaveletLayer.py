from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, haar_matrix, batch_size, **kwargs):
        self.haarMatrix = haar_matrix
        self.batch_size = batch_size
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def dwt(self, image):
        if not (image.shape[0] % 4 == 0) or not (image.shape[1] % 4 == 0):
            print(K.int_shape(image))
            print("The size of your image must be a multiple of 4!")

        haarMatrix1 = K.variable(self.haarMatrix, dtype="float32")
        haarMatrix2 = K.repeat_elements(K.expand_dims(K.transpose(haarMatrix1), 0), int(K.get_value(self.batch_size)), 0)
        haarMatrix1 = K.repeat_elements(K.expand_dims(haarMatrix1, 0), int(K.get_value(self.batch_size)), 0)
        result = []
        nchannels = 3
        if len(image.shape) == nchannels:
            data0 = image[:, :, :, 0]
            result0 = K.batch_dot(K.batch_dot(self.haarMatrix, data0), self.haarMatrix2)
            result[:, :, :, 0] = result0

            data1 = image[:, :, :, 1]
            result1 = K.batch_dot(K.batch_dot(self.haarMatrix, data1), self.haarMatrix2)
            result[:, :, :, 1] = result1

            data2 = image[:, :, :, 2]
            result2 = K.batch_dot(K.batch_dot(self.haarMatrix, data2), self.haarMatrix2)
            result[:, :, :, 2] = result2

        else:
            nconv = K.int_shape(image)[-1]
            for i in range(nconv):
                tmp = K.batch_dot(haarMatrix1, image[:,:,:,i], axes=[1, 2])
                print(K.int_shape(tmp))
                result.append(K.batch_dot(tmp, haarMatrix2, axes=[1,2]))

        result = K.concatenate(result, -1)
        result = K.reshape(result, (-1, K.int_shape(image)[1], K.int_shape(image)[2], nconv))

        return result


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        #self.kernel = self.add_weight(name='kernel',shape=(input_shape[1], self.output_dim),initializer='uniform',trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, x):

        result = self.dwt(x)
        result = result[:, :int(K.int_shape(x)[1] / 2), : int(K.int_shape(x)[2]/2) , :]
        print(K.int_shape(result))
        return result

    def compute_output_shape(self, input_shape):
        return self.output_dim

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, haar_matrix, **kwargs):
        self.output_dim = output_dim
        self.haarMatrix = haar_matrix
        super(MyLayer, self).__init__(**kwargs)

    def dwt(self, image):
        if not (image.shape[0] % 4 == 0) or not (image.shape[1] % 4 == 0):
            print("The size of your image must bea multiple of 4!")
            return None

        haarMatrix2 = K.transpose(self.haarMatrix)
        result = K.zeros(image.shape, dtype=K.float)
        nchannels = 3
        if len(image.shape) == nchannels:
            data0 = image[:, :, 0]
            result0 = K.batch_dot(K.batch_dot(self.haarMatrix, data0), self.haarMatrix2)
            result[:, :, 0] = result0

            data1 = image[:, :, 1]
            result1 = K.batch_dot(K.batch_dot(self.haarMatrix, data0), self.haarMatrix2)
            result[:, :, 1] = result1

            data2 = image[:, :, 2]
            result2 = K.batch_dot(K.batch_dot(self.haarMatrix, data0), self.haarMatrix2)
            result[:, :, 2] = result2

        else:

            result = K.batch_dot(K.batch_dot(self.haarMatrix, image), haarMatrix2)

        return result


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, x):
        result = self.dwt(x)
        return result[:K.cast(K.int_shape(result)[0]/2, dtype="int"), :K.cast(K.int_shape(result)[1]/2, dtype= "int")]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

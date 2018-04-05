from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, haar_matrix, **kwargs):
        self.haarMatrix = haar_matrix
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def dwt(self, image):
        if not (image.shape[1] % 4 == 0) or not (image.shape[2] % 4 == 0):
            print(K.int_shape(image))
            print("The size of your image must be a multiple of 4!")

        haarMatrix1 = K.variable(self.haarMatrix, dtype="float32")
        haarMatrix2 = K.repeat_elements(K.expand_dims(K.transpose(haarMatrix1), -1), K.int_shape(image)[3], -1)
        haarMatrix1 = K.repeat_elements(K.expand_dims(haarMatrix1, -1), K.int_shape(image)[3], -1)

        input_ones_matrix = (K.abs(image) + 1) / (K.abs(image) + 1)
        haarMatrix2 = input_ones_matrix * haarMatrix2
        haarMatrix1 = input_ones_matrix * haarMatrix1

        result = []

        nconv = K.int_shape(image)[3]
        for i in range(nconv):
            tmp = K.batch_dot(haarMatrix1[:, :, :, i], image[:, :, :, i], axes=[1, 2])
            result.append(K.batch_dot(tmp, haarMatrix2[:, :, :, i], axes=[1, 2]))

        result = K.concatenate(result, -1)
        result = K.reshape(result, (-1, K.int_shape(image)[1], K.int_shape(image)[2], nconv))

        return result


    def build(self, input_shape):
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, x):
        result = self.dwt(x)
        result = result[:, :int(K.int_shape(x)[1] / 2), : int(K.int_shape(x)[2]/2) , :]
        return result

    def compute_output_shape(self, input_shape):
        return self.output_dim

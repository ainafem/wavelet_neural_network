from keras.callbacks import Callback
from keras import backend as K
import math

class update_batch_size(Callback):
    def __init__(self, batch_size, n_images):
        self.batch_size = batch_size
        self.n_images = n_images
        self.max_iterations = math.ceil(self.n_images / float(K.get_value(self.batch_size)))


    def on_batch_end(self, batch, logs=None):
        if(batch + 1 == self.max_iterations-1):
            K.set_value(self.batch_size, self.n_images - (batch+1) * int(K.get_value(self.batch_size)))



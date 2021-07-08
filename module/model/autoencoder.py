import tensorflow as tf
from tensorflow.keras.layers import *

class Encoder(tf.keras.models.Model) :
    def __init__(self, step, input_size) :
        super(Encoder, self).__init__()
        self.model = tf.keras.models.Sequential([
            InputLayer(input_shape=(input_size,)),
            Dense(input_size - step * 1, activation='relu'),
            Dense(input_size - step * 2, activation='relu'),
            Dense(input_size - step * 3, activation='relu'),
            Dense(input_size - step * 4, activation='relu'),
            Dense(input_size - step * 5),
        ])

    def call(self, x) :
        z = self.model(x)
        return z

class Decoder(tf.keras.models.Model) :
    def __init__(self, step, input_size, output_size) :
        super(Decoder, self).__init__()
        self.model = tf.keras.models.Sequential([
            InputLayer(input_shape=(input_size,)),
            Dense(output_size - step * 4, activation='relu'),
            #             Dense(output_size - step * 3, activation='relu'),
            Dense(output_size - step * 2, activation='relu'),
            #             Dense(output_size - step * 1, activation='relu'),
            Dense(output_size),
        ])

    def call(self, x) :
        z = self.model(x)
        return z

class AutoEncoder(tf.keras.models.Model) :
    def __init__(self, input_size, step) :
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(step, input_size)
        self.decoder = Decoder(step, input_size - step * 5, input_size)

    def call(self, x) :
        y = self.encoder(x)
        z = self.decoder(y)
        return z


import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,LSTM,Dense, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
import numpy as np

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

get_custom_objects().update({'gelu': tf.keras.layers.Activation(gelu)})

class CLSTM(tf.keras.Model):
    def __init__(self, frames, nfilt, nlabel, batch):
        super(CLSTM, self).__init__(self)
        self.nfilt = nfilt
        self.nlabel = nlabel
        self.frames = frames
        self.batch_size = batch
        self.hidden_size_m = 128
        self.hidden_size_s = 256
        self.hidden_size_l = 512
        self.hidden_size_xl = 1024

        self.cnnLayer1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation=None)
        self.cnnLayer2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='gelu')
        self.cnnLayer3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=None)
        self.cnnLayer4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='gelu')
        self.cnnLayer5 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=None)
        self.cnnLayer6 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='gelu')
        self.cnnLayer7 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=None)
        self.cnnLayer8 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='gelu')

        self.fcLayer1 = Dense(self.hidden_size_l, activation=None)
        self.fcLayer3 = Dense(nlabel, activation="softmax")
        self.fcLayer2 = Dense(self.hidden_size_l, activation=None)
        
        self.poolLayer1 = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid')

    def call(self, inputData):
        inputData = tf.expand_dims(inputData, axis=-1)
        hiddenOutput = self.cnnLayer1(inputData)
        hiddenOutput = self.cnnLayer2(hiddenOutput)
        hiddenOutput = tf.nn.dropout(hiddenOutput, 0.1)
        hiddenOutput = self.poolLayer1(hiddenOutput)
        hiddenOutput = self.cnnLayer3(hiddenOutput)
        hiddenOutput = self.cnnLayer4(hiddenOutput)
        hiddenOutput = tf.nn.dropout(hiddenOutput, 0.1)
        hiddenOutput = self.poolLayer1(hiddenOutput)
        hiddenOutput = self.cnnLayer5(hiddenOutput)
        hiddenOutput = self.cnnLayer6(hiddenOutput)
        hiddenOutput = tf.nn.dropout(hiddenOutput, 0.1)
        hiddenOutput = self.poolLayer1(hiddenOutput)
        hiddenOutput = self.cnnLayer7(hiddenOutput)
        hiddenOutput = self.cnnLayer8(hiddenOutput)
        hiddenOutput = tf.reshape(hiddenOutput, [self.batch_size, -1, self.nfilt*128])
        #hiddenOutput = self.lstmLayer(hiddenOutput)
        output = self.fcLayer1(hiddenOutput)
        #output = self.fcLayer2(output)
        pred = self.fcLayer3(output)
        return  pred

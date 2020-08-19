
import tensorflow as tf
from tensorflow import keras

def get_activation(name):
    if name == "sigmoid":
        return keras.activations.sigmoid
    if name == "relu":
        return keras.activations.relu

class conv_bn_layer():
    def __init__(self, filter_num, filter_size, stride=(1, 1), padding="same", activation=None):
        self.conv_layer = keras.layers.Conv2D(filter_num, filter_size, strides=stride, padding=padding, activation=None)
        self.normal_layer = keras.layers.BatchNormalization()
        self.activation = get_activation(activation)
    def __call__(self, input):
        conv_output = self.conv_layer(input)
        normalize_output = self.normal_layer(conv_output)
        return self.activation(normalize_output)

class gru_bn_layer():
    def __init__(self, size, activation):
        self.input_proj_forward_layer = keras.layers.Dense(size*3, activation=None)
        self.input_proj_reverse_layer = keras.layers.Dense(size*3, activation=None)
        self.forward_normal_layer = keras.layers.BatchNormalization()
        self.reverse_normal_layer = keras.layers.BatchNormalization()
        self.forward_gru_layer = keras.layers.GRU(size, return_sequences=True, activation="sigmoid")
        self.reverse_gru_layer = keras.layers.GRU(size, return_sequences=True, activation="sigmoid", go_backwards=True)

    def __call__(self, input):
        input_proj_forward = self.input_proj_forward_layer(input)
        input_proj_reverse = self.input_proj_reverse_layer(input)
        input_proj_bn_forward = self.forward_normal_layer(input_proj_forward)
        input_proj_bn_reverse = self.reverse_normal_layer(input_proj_reverse)
        forward_gru = self.forward_gru_layer(input_proj_bn_forward)
        reverse_gru = self.reverse_gru_layer(input_proj_bn_reverse)
        return tf.concat([forward_gru, reverse_gru], axis=-1)

class  conv_group():
    def __init__(self, stack_num):
        self.conv_start = conv_bn_layer(32, (41, 11), stride=(2, 3), padding='same', activation="relu")
        self.conv_stack = []
        for i in range(stack_num):
            conv_layer = conv_bn_layer(32, (21, 11), stride=(2, 1), padding='same', activation="relu")
            self.conv_stack.append(conv_layer)
    def __call__(self, input):
        conv = self.conv_start(input)
        for layer in self.conv_stack:
            conv = layer(conv)
        return conv

class  rnn_group():
    def __init__(self, stack_num, size):
        self.rnn_layers = []
        for i in range(stack_num):
            self.rnn_layers.append(gru_bn_layer(size, "relu"))
    def __call__(self, input):
        output = input
        for layer in self.rnn_layers:
            output = layer(output)
        return output

class deep_speech_network(keras.Model):
    def __init__(self, num_conv_layers=2, num_rnn_layers=3,rnn_size=256, nlabels=1300, share_rnn_weights=True):
        super(deep_speech_network, self).__init__(self)
        self.conv_group = conv_group(num_conv_layers)
        self.rnn_group = rnn_group(num_rnn_layers, rnn_size)
        self.fc_layer = keras.layers.Dense(nlabels+1, activation=None)
    def call(self, audio_data):
        audio_data = tf.expand_dims(audio_data, -1)
        output = self.conv_group(audio_data)
        output = tf.reshape(output, [output.shape[0], output.shape[1], -1])
        #print(output.shape)
        output = self.rnn_group(output)
        output = self.fc_layer(output)
        return tf.nn.softmax(output, axis=-1)

import numpy as np
import matplotlib.pyplot as plt

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs, training=True):
        if training:
            self.mask = (np.random.rand(*inputs.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            return inputs * self.mask
        return inputs

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask

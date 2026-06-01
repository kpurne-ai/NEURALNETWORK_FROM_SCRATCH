import numpy as np
import matplotlib.pyplot as plt
class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient, learning_rate):
        relu_gradient = self.inputs > 0
        return output_gradient * relu_gradient

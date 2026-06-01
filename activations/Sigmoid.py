import numpy as np
import matplotlib.pyplot as plt
class sigmoid:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs
    def backward(self, output_gradient, learning_rate):
        sigmoid_gradient = self.outputs * (1 - self.outputs)
        return output_gradient * sigmoid_gradient

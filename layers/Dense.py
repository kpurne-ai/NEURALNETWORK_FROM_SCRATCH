import numpy as np
import matplotlib.pyplot as plt
class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.inputs.T, output_gradient)
        inputs_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return inputs_gradient
    

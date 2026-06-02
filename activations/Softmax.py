import numpy as np
import matplotlib.pyplot as plt

class Softmax:
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        return probabilities

    def backward(self, output_gradient, learning_rate):
        # Note: This is a simplified version often used when combined with Cross-Entropy
        # For a general Softmax backward, the Jacobian matrix is required.
        return output_gradient

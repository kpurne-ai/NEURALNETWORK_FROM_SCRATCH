import numpy as np
import matplotlib.pyplot as plt

class Mse:
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.outputs = None

    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.outputs = np.mean((inputs - targets) ** 2)
        return self.outputs

    def backward(self):
        mse_gradient = 2 * (self.inputs - self.targets) / self.inputs.shape[0]
        return mse_gradient


import numpy as np
import matplotlib.pyplot as plt
class MSE:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.outputs = np.mean((inputs - targets) ** 2)
        return self.outputs
    def backward(self):
        mse_gradient = 2 * (self.inputs - self.targets) / self.inputs.shape[0]
        return mse_gradient
class BinaryCrossEntropy:
    def __init__(self):
        self.inputs = None
        self.targets = None

    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        epsilon = 1e-15
        inputs_clipped = np.clip(inputs, epsilon, 1 - epsilon)
        self.outputs = -np.mean(targets * np.log(inputs_clipped) + (1 - targets) * np.log(1 - inputs_clipped))
        return self.outputs

    def backward(self):
        epsilon = 1e-15
        inputs_clipped = np.clip(self.inputs, epsilon, 1 - epsilon)
        bce_gradient = -(self.targets / inputs_clipped) + ((1 - self.targets) / (1 - inputs_clipped))
        return bce_gradient / self.inputs.shape[0]
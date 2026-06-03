import numpy as np
import matplotlib.pyplot as plt

class MAE:
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.outputs = None

    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.outputs = np.mean(np.abs(inputs - targets))
        return self.outputs

    def backward(self):
        mae_gradient = np.sign(self.inputs - self.targets) / self.inputs.shape[0]
        return mae_gradient

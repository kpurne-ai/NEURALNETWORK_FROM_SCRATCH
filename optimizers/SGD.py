import numpy as np
class SGD:

    def __init__(
        self,
        learning_rate=0.01
    ):

        self.learning_rate = learning_rate

    def step(self, layer):

        layer.weights -= (
            self.learning_rate
            *
            layer.d_weights
        )

        layer.bias -= (
            self.learning_rate
            *
            layer.d_bias
        )
        return layer.weights, layer.bias
    def update_parameters(self, layer):
        layer.weights -= (
            self.learning_rate
            *
            layer.d_weights
        )

        layer.bias -= (
            self.learning_rate
            *
            layer.d_bias
        )
        return layer.weights, layer.bias
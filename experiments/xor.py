import numpy as np

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from activations.Sigmoid import sigmoid
from activations.ReLu import ReLU
from layers.Dense import Dense
from losses.BinaryCrossEntropy import BinaryCrossEntropy
from losses.MSE import Mse
from losses.MAE import MAE
from activations.Softmax import Softmax


from network.NeuralNetwork import NeuralNetwork
nn = NeuralNetwork()

#inputs with xor output
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])
nn.add(Dense(2, 4))
nn.add(sigmoid())
nn.add(Dense(4, 1))
nn.add(sigmoid())


nn.fit(inputs, targets, epochs=1000, learning_rate=0.01, loss_function=BinaryCrossEntropy(), test_size=0.2)

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

#inputs with huge dataset
X = np.random.rand(1000, 2)
y = np.array([[1] if x[0] > 0.5 and x[1] > 0.5 else [0] for x in X])

nn.add(Dense(2, 8))
nn.add(ReLU())
nn.add(Dense(8, 4))
nn.add(ReLU())
nn.add(Dense(4, 1))
nn.add(sigmoid())

nn.fit(X, y, epochs=1000, learning_rate=0.01, loss_function=BinaryCrossEntropy(), test_size=0.2)

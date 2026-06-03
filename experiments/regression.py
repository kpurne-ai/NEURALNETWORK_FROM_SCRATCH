import numpy as np
import matplotlib.pyplot as plt


import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from activations.Sigmoid import sigmoid 
from activations.ReLu import ReLU
from activations.Softmax import Softmax
from layers.Dense import Dense
from losses.BinaryCrossEntropy import BinaryCrossEntropy
from losses.MSE import Mse
from losses.MAE import MAE


from network.NeuralNetwork import NeuralNetwork
nn = NeuralNetwork()

#data for regresson nn
X = np.random.rand(1000, 6)
y = (3*X[:, 0] + 2*X[:, 1] + 4*X[:, 2] + 1*X[:, 3] ).reshape(-1, 1)

nn.add(Dense(6, 16))

nn.add(Dense(16, 8))

nn.add(Dense(8, 1))
nn.batch_train(X, y, epochs=1000, learning_rate=0.01, batch_size=32, loss_function=Mse())

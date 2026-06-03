import numpy as np

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

#inputs with huge dataset
X = np.random.rand(1000, 8)
y = np.array([[1] if x[0] > 0.5 and x[1] > 0.5 else [0] for x in X])

nn.add(Dense(8, 16))
nn.add(ReLU())
nn.add(Dense(16, 8))
nn.add(ReLU())
nn.add(Dense(8, 1))
nn.add(sigmoid())


#batch size wise training
batch_size = 32
nn.batch_train(X, y, epochs=1000, learning_rate=0.01, batch_size=batch_size, loss_function=BinaryCrossEntropy(), test_size=0.2)

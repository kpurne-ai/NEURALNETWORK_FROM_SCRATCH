import numpy as np
import matplotlib.pyplot as plt
from activations.ReLu import ReLU
from layers.Dense import Dense
from activations.Sigmoid import sigmoid
from losses.BinaryCrossEntropy import BinaryCrossEntropy
from activations.Softmax import Softmax
from losses.MSE import Mse
from losses.MAE import MAE


class NeuralNetwork:

    def __init__(self):

        self.layers = []


    def add(self, layer):

        self.layers.append(layer)
    
    def forward(self, X):

        output = X

        for layer in self.layers:

            output = layer.forward(output)

        return output
    def backward(self, output_gradient, learning_rate):

        for layer in reversed(self.layers):

            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient
    
    def predict(self, X):       
        return self.forward(X)
    


    def evaluate(self, X, y):

        predictions = self.predict(X)
        predictions_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions_binary == y)
        predictions = self.predict(X)
        predictions_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions_binary == y)
        return accuracy

    def plot_actual_vs_predicted(self, X, y):
        predictions = self.forward(X)
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), marker='o', label='Actual')
        plt.scatter(X[:, 0], X[:, 1], c=predictions.flatten(), marker='x', label='Predicted')
        plt.title('Actual vs Predicted')
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.legend()
        plt.show()

    def plottrainlossVStestloss(self, train_losses, test_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def fit(self, X, y, epochs, learning_rate, loss_function, batch_size=None, test_size=0.0):

        if loss_function is None:
            raise ValueError("fit requires a loss_function")

        if test_size > 0.0:
            split_index = int((1 - test_size) * len(X))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            if batch_size is None:
                predictions = self.forward(X_train)
                epoch_loss = loss_function.forward(predictions, y_train)
                self.backward(loss_function.backward(), learning_rate)
            else:
                epoch_loss = 0.0
                num_batches = 0
                for start in range(0, len(X_train), batch_size):
                    end = start + batch_size
                    X_batch = X_train[start:end]
                    y_batch = y_train[start:end]

                    predictions = self.forward(X_batch)
                    batch_loss = loss_function.forward(predictions, y_batch)
                    self.backward(loss_function.backward(), learning_rate)

                    epoch_loss += batch_loss
                    num_batches += 1

                epoch_loss /= num_batches

            train_losses.append(epoch_loss)

            test_predictions = self.forward(X_test)
            test_loss = loss_function.forward(test_predictions, y_test)
            test_losses.append(test_loss)

            if epoch % 100 == 0:
                if test_size > 0.0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss}, Test Loss: {test_loss}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

        final_predictions = self.forward(X)
        self.train_losses = train_losses
        self.test_losses = test_losses
        print("Accuracy: ", self.evaluate(X, y))
        self.plottrainlossVStestloss(train_losses, test_losses)
        return final_predictions



    
    def batch_train(self, X, y, epochs, learning_rate, batch_size, loss_function=None, test_size=0.0):
        return self.fit(X, y, epochs, learning_rate, loss_function, batch_size=batch_size, test_size=test_size)

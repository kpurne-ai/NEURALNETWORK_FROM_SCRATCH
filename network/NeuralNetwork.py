import numpy as np
import matplotlib.pyplot as plt
from activations.ReLu import ReLU
from layers.Dense import Dense
from activations.Sigmoid import sigmoid
from losses.BinaryCrossEntropy import BinaryCrossEntropy



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

        predictions = self.forward(X)
        predicted_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_classes == y)
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

    def fit(self, X, y, epochs, learning_rate):

        for epoch in range(epochs):

            predictions = self.forward(X)

            loss = BinaryCrossEntropy()
            loss_value = loss.forward(predictions, y)
            self.backward(loss.backward(), learning_rate) 
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value}")
            final_predictions = self.forward(X)
        print("Accuracy: ", self.evaluate(X, y))
        self.plot_actual_vs_predicted(X, y)

        return final_predictions



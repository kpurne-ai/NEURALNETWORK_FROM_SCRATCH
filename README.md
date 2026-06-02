# NeuralNetwork_From_Scratch

## Overview

This project is a small neural network implementation from scratch using `numpy`. It includes core components such as dense layers, activation functions, loss functions, an optimizer, and an XOR, deep_netowrk training example.

## Project Structure

- `network/NeuralNetwork.py`
  - `NeuralNetwork` class with support for adding layers, forward propagation, backward propagation, training, and evaluation.
- `layers/Dense.py`
  - Dense (fully connected) layer implementation.
- `activations/Sigmoid.py`
  - Sigmoid activation layer.
- `activations/ReLu.py`
  - ReLU activation layer.
- `losses/MSE.py`
  - Mean Squared Error loss function.
- `losses/BinaryCrossEntropy.py`
  - Binary Cross-Entropy loss function.
- `optimizers/SGD.py`
  - Stochastic Gradient Descent optimizer.
- `experiments/xor.py`
  - XOR dataset training example.
  - DeepNN training example

## Requirements

- Python 3.8+
- `numpy`

Install dependencies with:

```bash
pip install numpy
```

## Usage

From the project root, run:

```bash
python experiments/xor.py
```

This script trains a simple neural network on the XOR dataset and prints training loss and accuracy.

## Notes

- Each layer should implement `forward()` and `backward()`.
- Activation layers should accept the `learning_rate` parameter in `backward()` even if they do not use it.
- The training loop in `NeuralNetwork.fit()` computes loss and updates parameters through backward propagation.
# NEURALNETWORK_FROM_SCRATCH

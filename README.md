# NeuralNetwork_From_Scratch

## Overview

This project is a small neural network implementation from scratch using `numpy`. It includes core components such as dense and dropout layers, activation functions, loss functions, a training loop, and example scripts for XOR, deep network, regression, and batch-size experiments.

## Project Structure

- `network/NeuralNetwork.py`
  - `NeuralNetwork` class with support for adding layers, forward propagation, backward propagation, training, evaluation, and loss plotting.
- `layers/Dense.py`
  - Dense (fully connected) layer implementation.
- `layers/Dropout.py`
  - Dropout layer implementation.
- `activations/Sigmoid.py`
  - Sigmoid activation layer.
- `activations/ReLu.py`
  - ReLU activation layer.
- `activations/Softmax.py`
  - Softmax activation layer.
- `losses/MSE.py`
  - Mean Squared Error loss function.
- `losses/MAE.py`
  - Mean Absolute Error loss function.
- `losses/BinaryCrossEntropy.py`
  - Binary Cross-Entropy loss function.
- `experiments/xor.py`
  - XOR dataset training example.
- `experiments/deep_network.py`
  - Example deep network training script.
- `experiments/regression.py`
  - Regression example using mean squared error.
- `experiments/exp4batch_size.py`
  - Batch size experiment script.

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

From the project root, run one of the example scripts:

```bash
python experiments/xor.py
python experiments/deep_network.py
python experiments/regression.py
python experiments/exp4batch_size.py
```

### Example scripts

- `experiments/xor.py`
  - Trains a small network on the XOR dataset.
  - Uses `BinaryCrossEntropy` loss and a sigmoid output layer.
- `experiments/deep_network.py`
  - Demonstrates a deeper model with ReLU hidden layers.
  - Uses synthetic classification data and `BinaryCrossEntropy`.
- `experiments/regression.py`
  - Demonstrates a regression task with mean squared error.
  - Uses a simple linear target generated from random input features.
- `experiments/exp4batch_size.py`
  - Designed to explore the effect of different `batch_size` values.

### Training API

The core `NeuralNetwork` API supports these arguments for `fit()` and `batch_train()`:

- `epochs`: number of training epochs.
- `learning_rate`: step size for gradient updates.
- `loss_function`: loss object such as `Mse()`, `MAE()`, or `BinaryCrossEntropy()`.
- `batch_size`: optional mini-batch size; when omitted, training runs on the full dataset.
- `test_size`: optional fraction of data held out for validation.

### Example usage

```bash
python experiments/xor.py
```

This runs the XOR example and prints loss/accuracy during training.

## Notes

- Each layer implements `forward()` and `backward()`.
- `NeuralNetwork.fit()` supports `batch_size` and `test_size` for flexible training.
- `NeuralNetwork.batch_train()` is an alias for `fit()` with `batch_size`.
- The network currently uses a simple manual training loop without a separate optimizer module.

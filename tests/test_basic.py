import numpy as np
import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from core.network import NeuralNetwork
from layers.dense import DenseLayer
from layers.activations import ReLU, Sigmoid
from losses.loss_functions import MeanSquaredError, SoftmaxCrossEntropy
from optimizers.optimizer_classes import SGD


def test_forward_pass():
    print("Testing forward pass...")

    net = NeuralNetwork()
    net.add_layer(DenseLayer(10, 5))
    net.add_layer(ReLU())
    net.add_layer(DenseLayer(5, 3))

    X = np.random.randn(4, 10)
    output = net.predict(X)

    assert output.shape == (4, 3), f"Expected shape (4, 3), got {output.shape}"
    print("Forward pass test passed!")


def test_backward_pass():
    print("Testing backward pass...")

    net = NeuralNetwork()
    net.add_layer(DenseLayer(10, 5))
    net.add_layer(ReLU())
    net.add_layer(DenseLayer(5, 3))
    net.set_loss(SoftmaxCrossEntropy())

    X = np.random.randn(4, 10)
    y = np.array([0, 1, 2, 1])

    loss = net.forward(X, y)
    net.backward()

    params, grads = net.get_params_and_grads()

    assert len(params) == 2, "Expected 2 layers with parameters"
    assert len(grads) == 2, "Expected 2 layers with gradients"

    print("Backward pass test passed!")


def test_optimizer():
    print("Testing optimizer...")

    net = NeuralNetwork()
    net.add_layer(DenseLayer(10, 5))
    net.add_layer(ReLU())
    net.add_layer(DenseLayer(5, 3))
    net.set_loss(SoftmaxCrossEntropy())

    X = np.random.randn(4, 10)
    y = np.array([0, 1, 2, 1])

    optimizer = SGD(lr=0.01)

    loss_before = net.forward(X, y)
    net.backward()

    params, grads = net.get_params_and_grads()
    for p, g in zip(params, grads):
        optimizer.update(p, g)

    loss_after = net.forward(X, y)

    print(f"Loss before: {loss_before:.4f}")
    print(f"Loss after: {loss_after:.4f}")
    print("Optimizer test passed!")


if __name__ == "__main__":
    test_forward_pass()
    test_backward_pass()
    test_optimizer()
    print("All basic tests passed!")


def test_init_weights():
    print("Testing init_weights...")

    net = NeuralNetwork()
    net.add_layer(DenseLayer(10, 5))
    net.add_layer(ReLU())
    net.add_layer(DenseLayer(5, 3))

    def w_init(shape):
        return np.full(shape, 0.123)

    def b_init(shape):
        return np.full(shape, 0.456)

    net.init_weights(w_init=w_init, b_init=b_init)

    params, grads = net.get_params_and_grads()
    assert len(params) == 2, "Expected 2 layers with parameters"

    for p in params:
        assert np.allclose(p['W'], 0.123), "Weights not initialized correctly"
        assert np.allclose(p['b'], 0.456), "Biases not initialized correctly"

    print("init_weights test passed!")
from training.trainer import Trainer
from optimizers.optimizer_classes import SGD, Adam
from losses.loss_functions import MeanSquaredError
from layers.activations import ReLU, LinearActivation
from layers.dense import DenseLayer
from core.network import NeuralNetwork
import numpy as np
import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)


def test_simple_regression():
    print("Testing simple regression problem...")

    np.random.seed(42)
    X = np.random.randn(200, 5)
    true_weights = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = X.dot(true_weights) + 0.1 * np.random.randn(200)
    y = y.reshape(-1, 1)

    split_idx = 160
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    network = NeuralNetwork()
    network.add_layer(DenseLayer(5, 20))
    network.add_layer(ReLU())
    network.add_layer(DenseLayer(20, 10))
    network.add_layer(ReLU())
    network.add_layer(DenseLayer(10, 1))
    network.add_layer(LinearActivation())
    network.set_loss(MeanSquaredError())

    optimizer = Adam(lr=0.01)

    print("Training network...")

    for epoch in range(100):
        network.set_train_mode(True)
        loss = network.forward(X_train, y_train)
        network.backward()

        params, grads = network.get_params_and_grads()
        for p, g in zip(params, grads):
            optimizer.update(p, g)

        if (epoch + 1) % 20 == 0:
            network.set_train_mode(False)
            test_loss = network.forward(X_test, y_test)
            print(
                f"Epoch {epoch + 1}/100 - Train Loss: {loss:.4f} - Test Loss: {test_loss:.4f}")

    network.set_train_mode(False)
    final_test_loss = network.forward(X_test, y_test)

    print(f"Final Test Loss (MSE): {final_test_loss:.4f}")

    assert final_test_loss < 0.5, f"Test loss too high: {final_test_loss:.4f}"
    print("Regression test passed!")


if __name__ == "__main__":
    test_simple_regression()

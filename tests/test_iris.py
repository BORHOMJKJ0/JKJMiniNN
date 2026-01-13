from training.trainer import Trainer
from optimizers.optimizer_classes import Adam
from losses.loss_functions import SoftmaxCrossEntropy
from layers.normalization import BatchNorm
from layers.activations import ReLU, Sigmoid
from layers.dense import DenseLayer
from core.network import NeuralNetwork
from utils.data_utils import load_data, split_data, normalize_data
import numpy as np
import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)


def test_iris_classification():
    print("Testing on Iris dataset...")

    X, y = load_data('iris')

    X, mean, std = normalize_data(X)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42)

    network = NeuralNetwork()
    network.add_layer(DenseLayer(4, 16))
    network.add_layer(Sigmoid())
    network.add_layer(BatchNorm(16))
    network.add_layer(DenseLayer(16, 8))
    network.add_layer(ReLU())
    network.add_layer(DenseLayer(8, 3))
    network.set_loss(SoftmaxCrossEntropy())

    optimizer = Adam(lr=0.01)
    trainer = Trainer(network, optimizer)

    print("Training network...")
    trainer.fit(X_train, y_train, X_test, y_test,
                epochs=100, batch_size=16, verbose=True)

    final_train_acc = network.accuracy(X_train, y_train)
    final_test_acc = network.accuracy(X_test, y_test)

    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")

    assert final_test_acc > 0.85, f"Test accuracy too low: {final_test_acc:.4f}"
    print("Iris test passed!")


if __name__ == "__main__":
    test_iris_classification()

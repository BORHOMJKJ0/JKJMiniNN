from training.trainer import Trainer
from optimizers.optimizer_classes import Adam
from losses.loss_functions import SoftmaxCrossEntropy
from layers.normalization import Dropout
from layers.activations import ReLU, Tanh
from layers.dense import DenseLayer
from core.network import NeuralNetwork
from utils.data_utils import load_data, split_data, normalize_data
import numpy as np
import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)


def test_digits_classification():
    print("Testing on Digits dataset...")

    X, y = load_data('digits')

    X, mean, std = normalize_data(X)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42)

    network = NeuralNetwork()
    network.add_layer(DenseLayer(64, 128))
    network.add_layer(ReLU())
    network.add_layer(Dropout(0.3))
    network.add_layer(DenseLayer(128, 64))
    network.add_layer(Tanh())
    network.add_layer(DenseLayer(64, 10))
    network.set_loss(SoftmaxCrossEntropy())

    optimizer = Adam(lr=0.001)
    trainer = Trainer(network, optimizer)

    print("Training network...")
    trainer.fit(X_train, y_train, X_test, y_test,
                epochs=50, batch_size=32, verbose=True)

    final_train_acc = network.accuracy(X_train, y_train)
    final_test_acc = network.accuracy(X_test, y_test)

    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")

    assert final_test_acc > 0.90, f"Test accuracy too low: {final_test_acc:.4f}"
    print("Digits test passed!")


if __name__ == "__main__":
    test_digits_classification()

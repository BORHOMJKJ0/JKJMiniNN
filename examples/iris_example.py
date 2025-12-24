import numpy as np
import sys
sys.path.append('..')

from utils.data_utils import load_data, split_data, normalize_data

from core.network import NeuralNetwork
from layers.dense import DenseLayer
from layers.activations import ReLU, Sigmoid
from layers.normalization import BatchNorm
from losses.loss_functions import SoftmaxCrossEntropy
from optimizers.optimizer_classes import Adam
from training.trainer import Trainer

X, y = load_data('iris')

X, mean, std = normalize_data(X)

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

print("Building network architecture...")
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

print("Training on Iris dataset...")
trainer.fit(X_train, y_train, X_test, y_test, epochs=100, batch_size=16, verbose=True)

print(f"Final Training Accuracy: {network.accuracy(X_train, y_train):.4f}")
print(f"Final Test Accuracy: {network.accuracy(X_test, y_test):.4f}")

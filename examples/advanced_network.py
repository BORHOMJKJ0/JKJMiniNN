import numpy as np
import sys
sys.path.append('..')

from utils.data_utils import load_data, split_data, normalize_data

from core.network import NeuralNetwork
from layers.dense import DenseLayer
from layers.activations import ReLU, Tanh
from layers.normalization import Dropout, BatchNorm
from losses.loss_functions import SoftmaxCrossEntropy
from optimizers.optimizer_classes import Adam, Momentum
from training.trainer import Trainer

X, y = load_data('digits')

X, mean, std = normalize_data(X)

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

print("Building advanced network with regularization...")
network = NeuralNetwork()
network.add_layer(DenseLayer(64, 256))
network.add_layer(ReLU())
network.add_layer(BatchNorm(256))
network.add_layer(Dropout(0.3))

network.add_layer(DenseLayer(256, 128))
network.add_layer(Tanh())
network.add_layer(BatchNorm(128))
network.add_layer(Dropout(0.3))

network.add_layer(DenseLayer(128, 64))
network.add_layer(ReLU())
network.add_layer(Dropout(0.2))

network.add_layer(DenseLayer(64, 10))
network.set_loss(SoftmaxCrossEntropy())

optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
trainer = Trainer(network, optimizer)

print("Training advanced network on Digits dataset...")
trainer.fit(X_train, y_train, X_test, y_test, epochs=100, batch_size=64, verbose=True)

print("" + "="*60)
print(f"Final Training Accuracy: {network.accuracy(X_train, y_train):.4f}")
print(f"Final Test Accuracy: {network.accuracy(X_test, y_test):.4f}")
print("="*60)
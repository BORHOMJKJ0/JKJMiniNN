import numpy as np
import sys
sys.path.append('..')

from core.network import NeuralNetwork
from layers.dense import DenseLayer
from layers.activations import ReLU
from losses.loss_functions import SoftmaxCrossEntropy
from optimizers.optimizer_classes import SGD
from training.trainer import Trainer

np.random.seed(42)

n_samples = 300
n_features = 2
n_classes = 3

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

network = NeuralNetwork()
network.add_layer(DenseLayer(2, 10))
network.add_layer(ReLU())
network.add_layer(DenseLayer(10, 5))
network.add_layer(ReLU())
network.add_layer(DenseLayer(5, 3))
network.set_loss(SoftmaxCrossEntropy())

optimizer = SGD(lr=0.1)
trainer = Trainer(network, optimizer)

print("Training simple classification network...")
trainer.fit(X_train, y_train, X_test, y_test, epochs=50, batch_size=32, verbose=True)
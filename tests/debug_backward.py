import numpy as np

from core.network import NeuralNetwork
from layers.dense import DenseLayer
from layers.activations import ReLU, Sigmoid
from layers.normalization import BatchNorm
from losses.loss_functions import SoftmaxCrossEntropy

net = NeuralNetwork()
net.add_layer(DenseLayer(4, 16))
net.add_layer(Sigmoid())
net.add_layer(BatchNorm(16))
net.add_layer(DenseLayer(16, 8))
net.add_layer(ReLU())
net.add_layer(DenseLayer(8, 3))
net.set_loss(SoftmaxCrossEntropy())

X = np.random.randn(5, 4)
y = np.array([0, 1, 2, 1, 0])

loss = net.forward(X, y)
print('Loss:', loss)
net.backward()

for i, layer in enumerate(net.layers):
    print('Layer', i, type(layer).__name__)
    print(' params keys:', list(getattr(layer, 'params', {}).keys()))
    print(' grads keys:', list(getattr(layer, 'grads', {}).keys()))
    print('')

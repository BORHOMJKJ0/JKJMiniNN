
from training.trainer import Trainer
from optimizers.optimizer_classes import Adam
from losses.loss_functions import SoftmaxCrossEntropy
from layers.normalization import Dropout
from layers.activations import ReLU
from layers.dense import DenseLayer
from core.network import NeuralNetwork
from utils.data_utils import load_data, split_data, normalize_data
import numpy as np
import sys
sys.path.append('..')


X, y = load_data('iris')
X, mean, std = normalize_data(X)
X_train, X_test, y_train, y_test = split_data(
    X, y, test_size=0.2, random_state=42)

print("=" * 70)
print("Iris Classification")
print("=" * 70)

network = NeuralNetwork()
network.add_layer(DenseLayer(4, 8, weight_init='he'))
network.add_layer(ReLU())
network.add_layer(Dropout(0.3))
network.add_layer(DenseLayer(8, 3, weight_init='he'))
network.set_loss(SoftmaxCrossEntropy())

optimizer = Adam(lr=0.003, beta1=0.9, beta2=0.999)
trainer = Trainer(network, optimizer, weight_decay=0.001)

trainer.fit(
    X_train, y_train,
    X_test, y_test,
    epochs=150,
    batch_size=16,
    verbose=True,
    early_stopping=True,
    patience=20,
    lr_decay=True,
    decay_rate=0.95,
    decay_every=30
)

train_acc = network.accuracy(X_train, y_train)
test_acc = network.accuracy(X_test, y_test)
gap = train_acc - test_acc

print(f"Training Accuracy:   {train_acc:.4f}")
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Overfitting Gap:     {gap:.4f}")

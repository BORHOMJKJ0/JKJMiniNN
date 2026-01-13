from training.tuner import HyperparameterTuner
from optimizers.optimizer_classes import Adam, SGD
from losses.loss_functions import SoftmaxCrossEntropy
from layers.activations import ReLU
from layers.dense import DenseLayer
from core.network import NeuralNetwork
from utils.data_utils import load_data, split_data, normalize_data
import numpy as np
import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)


def build_model(params):
    network = NeuralNetwork()

    network.add_layer(DenseLayer(4, params['hidden_size']))
    network.add_layer(ReLU())
    network.add_layer(DenseLayer(params['hidden_size'], 3))
    network.set_loss(SoftmaxCrossEntropy())

    if params['optimizer'] == 'adam':
        optimizer = Adam(lr=params['learning_rate'])
    else:
        optimizer = SGD(lr=params['learning_rate'])

    return network, optimizer


def test_grid_search():
    print("Testing Grid Search for hyperparameter tuning...")

    X, y = load_data('iris')

    X, mean, std = normalize_data(X)

    X_train, X_temp, y_train, y_temp = split_data(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = split_data(
        X_temp, y_temp, test_size=0.5, random_state=42)

    param_grid = {
        'hidden_size': [8, 16],
        'learning_rate': [0.01, 0.001],
        'optimizer': ['adam', 'sgd']
    }

    tuner = HyperparameterTuner()
    best_params, best_score = tuner.grid_search(
        build_model,
        X_train, y_train,
        X_val, y_val,
        param_grid,
        epochs=30,
        batch_size=16
    )

    print(f"Best parameters found: {best_params}")
    print(f"Best validation score: {best_score:.4f}")

    final_model, final_optimizer = build_model(best_params)
    from training.trainer import Trainer
    trainer = Trainer(final_model, final_optimizer)
    trainer.fit(X_train, y_train, X_test, y_test,
                epochs=50, batch_size=16, verbose=False)

    test_acc = final_model.accuracy(X_test, y_test)
    print(f"Final test accuracy with best params: {test_acc:.4f}")


if __name__ == "__main__":
    test_grid_search()

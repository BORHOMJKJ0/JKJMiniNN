import numpy as np
from core.network import NeuralNetwork
from core.base_layer import Layer

from layers.dense import DenseLayer
from layers.activations import ReLU, Sigmoid, Tanh, LinearActivation
from layers.normalization import Dropout, BatchNorm

from losses.loss_functions import MeanSquaredError, SoftmaxCrossEntropy

from optimizers.optimizer_classes import SGD, Momentum, AdaGrad, Adam

from training.trainer import Trainer
from training.tuner import HyperparameterTuner


def test_dense_and_activation_forward_backward():
    x = np.random.randn(6, 4)
    d = DenseLayer(4, 3)
    out = d.forward(x)
    assert out.shape == (6, 3)

    dout = np.random.randn(6, 3)
    dx = d.backward(dout)
    assert dx.shape == x.shape

    relu = ReLU()
    r = relu.forward(out)
    assert r.shape == out.shape
    assert relu.backward(dout).shape == dout.shape

    sig = Sigmoid()
    s = sig.forward(out)
    assert s.shape == out.shape
    assert sig.backward(dout).shape == dout.shape

    tan = Tanh()
    t = tan.forward(out)
    assert t.shape == out.shape
    assert tan.backward(dout).shape == dout.shape


def test_dropout_and_batchnorm():
    x = np.random.randn(5, 4)
    drop = Dropout(0.5)
    drop.train_mode = True
    y = drop.forward(x)
    assert y.shape == x.shape
    drop.backward(np.ones_like(x))

    bn = BatchNorm(4)
    bn.train_mode = True
    y2 = bn.forward(x)
    assert y2.shape == x.shape
    dx = bn.backward(np.ones_like(x))
    assert dx.shape == x.shape


def test_losses():
    y_true = np.array([0, 1, 2])
    x = np.random.randn(3, 3)
    sce = SoftmaxCrossEntropy()
    loss = sce.forward(x, y_true)
    assert isinstance(loss, float)
    dx = sce.backward()
    assert dx.shape == x.shape

    y_pred = np.random.randn(4, 2)
    y_t = np.random.randn(4, 2)
    mse = MeanSquaredError()
    loss2 = mse.forward(y_pred, y_t)
    assert isinstance(loss2, float)
    dx2 = mse.backward()
    assert dx2.shape == y_pred.shape


def test_optimizers_update():
    params = {'W': np.ones((2, 2)), 'b': np.zeros((1, 2))}
    grads = {'W': np.full((2, 2), 0.1), 'b': np.full((1, 2), 0.1)}

    for Optim in (SGD, Momentum, AdaGrad, Adam):
        opt = Optim()
        p_before = {k: v.copy() for k, v in params.items()}
        opt.update(params, grads)
        for k in params:
            assert params[k].shape == p_before[k].shape


def test_network_trainer_basic_flow():
    nn = NeuralNetwork()
    nn.add_layer(DenseLayer(4, 5))
    nn.add_layer(ReLU())
    nn.add_layer(DenseLayer(5, 3))
    nn.set_loss(SoftmaxCrossEntropy())

    x = np.random.randn(10, 4)
    y = np.random.randint(0, 3, size=(10,))

    loss = nn.forward(x, y)
    assert isinstance(loss, float)
    nn.backward()

    params, grads = nn.get_params_and_grads()
    assert isinstance(params, list)
    assert isinstance(grads, list)

    trainer = Trainer(nn, SGD(lr=0.01))
    loss2 = trainer.train_step(x[:4], y[:4])
    assert isinstance(loss2, float)


def test_hyperparameter_tuner_smoke():
    def build_model(params):
        nn = NeuralNetwork()
        nn.add_layer(DenseLayer(4, params.get('hidden', 5)))
        nn.add_layer(ReLU())
        nn.add_layer(DenseLayer(params.get('hidden', 5), 3))
        opt = SGD(lr=params.get('lr', 0.01))
        nn.set_loss(SoftmaxCrossEntropy())
        return nn, opt

    x = np.random.randn(20, 4)
    y = np.random.randint(0, 3, size=(20,))

    tuner = HyperparameterTuner()
    best_params, best_score = tuner.grid_search(
        build_model, x, y, x, y, {'hidden': [4], 'lr': [0.01]}, epochs=1, batch_size=8)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)

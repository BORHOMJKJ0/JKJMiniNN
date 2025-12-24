import numpy as np

from core.network import NeuralNetwork
from layers.dense import DenseLayer
from layers.activations import ReLU, Sigmoid
from layers.normalization import BatchNorm
from losses.loss_functions import SoftmaxCrossEntropy
from optimizers.optimizer_classes import Adam
from training.trainer import Trainer

np.random.seed(0)

X = np.random.randn(20, 4)
y = np.random.randint(0,3,size=(20,))

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

for step in range(10):
    try:
        loss = trainer.train_step(X, y)
        params_list, grads_list = network.get_params_and_grads()
        for i, (params, grads) in enumerate(zip(params_list, grads_list)):
            print(f"Layer {i} param keys: {list(params.keys())}")
            print(f"Layer {i} grad keys: {list(grads.keys())}")
        print('Step', step, 'loss', loss)
    except Exception as e:
        print('Exception at step', step, type(e), e)
        params_list, grads_list = network.get_params_and_grads()
        for i, (params, grads) in enumerate(zip(params_list, grads_list)):
            print(f"Layer {i} param keys: {list(params.keys())}")
            print(f"Layer {i} grad keys: {list(grads.keys())}")
        raise

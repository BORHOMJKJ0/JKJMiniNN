import numpy as np
from core.base_layer import Layer


class DenseLayer(Layer):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W = np.random.randn(n_in, n_out) * 0.01
        self.b = np.zeros((1, n_out))
        self.params = {'W': self.W, 'b': self.b}
        self.x_input = None

    def forward(self, x):
        self.x_input = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.x_input.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)

        self.grads = {'W': dW, 'b': db}
        return dx
import numpy as np
from core.base_layer import Layer


class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)

    def backward(self, dout):
        return dout * self.mask


class BatchNorm(Layer):
    def __init__(self, input_dim, momentum=0.9, eps=1e-5):
        super().__init__()
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.params = {'gamma': self.gamma, 'beta': self.beta}

        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))

        self.train_mode = True
        self.xc = None
        self.std = None
        self.xn = None

    def forward(self, x):
        if self.train_mode:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.xc = x - mu
            self.std = np.sqrt(var + self.eps)
            self.xn = self.xc / self.std

            out = self.gamma * self.xn + self.beta
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + self.eps)
            out = self.gamma * xn + self.beta

        return out

    def backward(self, dout):
        N = dout.shape[0]

        dbeta = np.sum(dout, axis=0, keepdims=True)
        dgamma = np.sum(dout * self.xn, axis=0, keepdims=True)

        dx = (1.0 / N) * (self.gamma / self.std) * (
            N * dout - np.sum(dout, axis=0, keepdims=True) - self.xn * np.sum(dout * self.xn, axis=0, keepdims=True)
        )

        self.grads = {'gamma': dgamma, 'beta': dbeta}
        return dx
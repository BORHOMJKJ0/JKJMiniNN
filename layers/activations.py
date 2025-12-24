import numpy as np
from core.base_layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)
        return dx


class LinearActivation(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout
import numpy as np
from core.base_layer import Layer


class MeanSquaredError(Layer):
    def __init__(self):
        super().__init__()
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred

        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self, dout=1):
        batch_size = self.y_true.shape[0]
        dx = 2 * (self.y_pred - self.y_true) / batch_size
        return dx


class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        super().__init__()
        self.y_true = None
        self.y_pred = None

    def forward(self, x, y_true):
        self.y_true = y_true

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        batch_size = y_true.shape[0]

        if y_true.ndim == 1:
            log_probs = - \
                np.log(self.y_pred[np.arange(batch_size), y_true] + 1e-7)
        else:
            log_probs = -np.sum(y_true * np.log(self.y_pred + 1e-7), axis=1)

        loss = np.mean(log_probs)
        return loss

    def backward(self, dout=1):
        batch_size = self.y_true.shape[0]
        dx = self.y_pred.copy()

        if self.y_true.ndim == 1:
            dx[np.arange(batch_size), self.y_true] -= 1
        else:
            dx -= self.y_true

        dx = dx / batch_size
        return dx

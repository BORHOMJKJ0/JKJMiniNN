import numpy as np


class Optimizer:
    def update(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update(self, params, grads):
        pid = id(params)
        if pid not in self.v:
            self.v[pid] = {}
            for key, val in params.items():
                self.v[pid][key] = np.zeros_like(val)

        for key in params.keys():
            self.v[pid][key] = self.momentum * self.v[pid][key] - self.lr * grads[key]
            params[key] += self.v[pid][key]


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = {}

    def update(self, params, grads):
        pid = id(params)
        if pid not in self.h:
            self.h[pid] = {}
            for key, val in params.items():
                self.h[pid][key] = np.zeros_like(val)

        for key in params.keys():
            self.h[pid][key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[pid][key]) + 1e-7)


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = {}
        self.v = {}

    def update(self, params, grads):
        pid = id(params)
        if pid not in self.m:
            self.m[pid] = {}
            self.v[pid] = {}
            for key, val in params.items():
                self.m[pid][key] = np.zeros_like(val)
                self.v[pid][key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[pid][key] = self.beta1 * self.m[pid][key] + (1 - self.beta1) * grads[key]
            self.v[pid][key] = self.beta2 * self.v[pid][key] + (1 - self.beta2) * (grads[key] ** 2)

            params[key] -= lr_t * self.m[pid][key] / (np.sqrt(self.v[pid][key]) + 1e-7)
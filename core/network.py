import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_layer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss_layer):
        self.loss_layer = loss_layer

    def init_weights(self, w_init=None, b_init=None):
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                w_shape = layer.W.shape
                b_shape = layer.b.shape

                if w_init is None:
                    layer.W = np.random.randn(*w_shape) * 0.01
                else:
                    layer.W = w_init(w_shape)

                if b_init is None:
                    layer.b = np.zeros(b_shape)
                else:
                    layer.b = b_init(b_shape)

                layer.params = {'W': layer.W, 'b': layer.b}

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, y):
        output = self.predict(x)
        loss = self.loss_layer.forward(output, y)
        return loss

    def backward(self):
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_params_and_grads(self):
        params = []
        grads = []

        for layer in self.layers:
            if hasattr(layer, 'params') and getattr(layer, 'params'):
                params.append(layer.params)
                grads.append(getattr(layer, 'grads', {}))

        return params, grads

    def accuracy(self, x, y):
        output = self.predict(x)

        if output.ndim == 1:
            output = output.reshape(1, output.size)

        pred = np.argmax(output, axis=1)

        if y.ndim != 1:
            y = np.argmax(y, axis=1)

        acc = np.sum(pred == y) / float(x.shape[0])
        return acc

    def set_train_mode(self, mode=True):
        for layer in self.layers:
            if hasattr(layer, 'train_mode'):
                layer.train_mode = mode
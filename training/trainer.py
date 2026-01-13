import numpy as np


class Trainer:
    def __init__(self, network, optimizer, weight_decay=0.0):
        self.network = network
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        self.best_test_acc = 0.0
        self.best_params = None

    def train_step(self, x_batch, y_batch):
        loss = self.network.forward(x_batch, y_batch)
        self.network.backward()

        params_list, grads_list = self.network.get_params_and_grads()

        if self.weight_decay > 0:
            for params, grads in zip(params_list, grads_list):
                if 'W' in grads:
                    grads['W'] += self.weight_decay * params['W']

        for params, grads in zip(params_list, grads_list):
            self.optimizer.update(params, grads)

        if self.weight_decay > 0:
            l2_loss = 0
            for params in params_list:
                if 'W' in params:
                    l2_loss += np.sum(params['W'] ** 2)
            loss += 0.5 * self.weight_decay * l2_loss

        return loss

    def fit(self, x_train, y_train, x_test=None, y_test=None,
            epochs=10, batch_size=32, verbose=True,
            early_stopping=False, patience=10,
            lr_decay=False, decay_rate=0.95, decay_every=10):
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)

        patience_counter = 0
        self.best_test_acc = 0.0

        for epoch in range(epochs):
            self.network.set_train_mode(True)

            idx = np.random.permutation(train_size)
            x_shuffled = x_train[idx]
            y_shuffled = y_train[idx]

            epoch_loss = 0
            for i in range(iter_per_epoch):
                start = i * batch_size
                end = start + batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss

            avg_loss = epoch_loss / iter_per_epoch
            self.train_loss_list.append(avg_loss)

            self.network.set_train_mode(False)
            train_acc = self.network.accuracy(x_train, y_train)
            self.train_acc_list.append(train_acc)

            if x_test is not None and y_test is not None:
                test_acc = self.network.accuracy(x_test, y_test)
                self.test_acc_list.append(test_acc)

                if early_stopping:
                    if test_acc > self.best_test_acc:
                        self.best_test_acc = test_acc
                        patience_counter = 0
                        self.best_params = self._save_params()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"\n Early stopping at epoch {epoch + 1}")
                            print(
                                f"Best test accuracy: {self.best_test_acc:.4f}")
                        self._restore_params(self.best_params)
                        break

                if verbose:
                    gap = train_acc - test_acc
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - "
                          f"Train Acc: {train_acc:.4f} - "
                          f"Test Acc: {test_acc:.4f} - "
                          f"Gap: {gap:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - "
                          f"Train Acc: {train_acc:.4f}")

            if lr_decay and (epoch + 1) % decay_every == 0:
                if hasattr(self.optimizer, 'lr'):
                    self.optimizer.lr *= decay_rate
                    if verbose:
                        print(
                            f"Learning rate decayed to {self.optimizer.lr:.6f}")

    def _save_params(self):
        saved = []
        for layer in self.network.layers:
            if hasattr(layer, 'params') and layer.params:
                layer_params = {}
                for key, val in layer.params.items():
                    layer_params[key] = val.copy()
                saved.append(layer_params)
            else:
                saved.append(None)
        return saved

    def _restore_params(self, saved_params):
        for layer, params in zip(self.network.layers, saved_params):
            if params is not None:
                for key, val in params.items():
                    layer.params[key] = val.copy()
                    if hasattr(layer, key):
                        setattr(layer, key, val.copy())

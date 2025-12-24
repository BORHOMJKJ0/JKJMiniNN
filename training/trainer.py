import numpy as np


class Trainer:
    def __init__(self, network, optimizer):
        self.network = network
        self.optimizer = optimizer
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self, x_batch, y_batch):
        loss = self.network.forward(x_batch, y_batch)
        self.network.backward()

        params_list, grads_list = self.network.get_params_and_grads()

        for params, grads in zip(params_list, grads_list):
            self.optimizer.update(params, grads)

        return loss

    def fit(self, x_train, y_train, x_test=None, y_test=None, epochs=10, batch_size=32, verbose=True):
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)

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

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f}")
import numpy as np
from itertools import product


class HyperparameterTuner:
    def __init__(self):
        self.best_params = None
        self.best_score = -np.inf
        self.results = []

    def grid_search(self, build_model_fn, x_train, y_train, x_val, y_val, param_grid, epochs=10, batch_size=32):
        from training.trainer import Trainer

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        for combo in product(*values):
            params = dict(zip(keys, combo))

            print(f"Trying parameters: {params}")

            model, optimizer = build_model_fn(params)

            trainer = Trainer(model, optimizer)
            trainer.fit(x_train, y_train, x_val, y_val,
                        epochs=epochs, batch_size=batch_size, verbose=False)

            val_acc = model.accuracy(x_val, y_val)

            self.results.append({
                'params': params,
                'val_acc': val_acc
            })

            print(f"Validation Accuracy: {val_acc:.4f}")

            if val_acc > self.best_score:
                self.best_score = val_acc
                self.best_params = params
                print(f"New best score: {val_acc:.4f}")

        print(f"Best parameters: {self.best_params}")
        print(f"Best validation accuracy: {self.best_score:.4f}")

        return self.best_params, self.best_score

    def random_search(self, build_model_fn, x_train, y_train, x_val, y_val, param_distributions, n_iter=10, epochs=10,
                      batch_size=32):
        from training.trainer import Trainer

        for i in range(n_iter):
            params = {}
            for key, values in param_distributions.items():
                if isinstance(values, list):
                    params[key] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    if isinstance(values[0], int):
                        params[key] = np.random.randint(values[0], values[1])
                    else:
                        params[key] = np.random.uniform(values[0], values[1])

            print(f"Iteration {i + 1}/{n_iter}")
            print(f"Trying parameters: {params}")

            model, optimizer = build_model_fn(params)

            trainer = Trainer(model, optimizer)
            trainer.fit(x_train, y_train, x_val, y_val,
                        epochs=epochs, batch_size=batch_size, verbose=False)

            val_acc = model.accuracy(x_val, y_val)

            self.results.append({
                'params': params,
                'val_acc': val_acc
            })

            print(f"Validation Accuracy: {val_acc:.4f}")

            if val_acc > self.best_score:
                self.best_score = val_acc
                self.best_params = params
                print(f"New best score: {val_acc:.4f}")

        print(f"Best parameters: {self.best_params}")
        print(f"Best validation accuracy: {self.best_score:.4f}")

        return self.best_params, self.best_score

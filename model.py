"""The example ML model to train in a federated setting."""
import numpy as np
from flwr.common import NDArrays
from sklearn.neural_network import MLPClassifier


class MyClassifier:
    def __init__(self, batch_size):
        """init."""
        self.module = MLPClassifier(
            solver="adam",
            alpha=1e-5,
            hidden_layer_sizes=(5, 2),
            batch_size=batch_size,
            verbose=True,
        )

    def fit(self, X, y, epochs):
        """Fit the model on a dataset."""
        for _ in range(epochs):
            self.module.partial_fit(X, y)
        return self.module

    def predict_proba(self, X):
        """Predict for a dataset."""
        return self.module.predict_proba(X)

    def get_parameters(self) -> NDArrays:
        """Get the model's parameters as a list of numpy arrays."""
        return [val for pair in zip(self.module.coefs_, self.module.intercepts_) for val in pair]

    def set_parameters(self, parameters: NDArrays):
        """Set the model's parameters from a list of numpy arrays."""
        self.module.coefs_ = parameters[0::2]
        self.module.intercepts_ = parameters[1::2]
        return self

    def initialize_parameters(self, input_dim, output_dim):
        """Initialize the model parameters to start training."""
        dims = (input_dim, *self.module.hidden_layer_sizes, output_dim)
        self.module.coefs_ = [np.random.rand(dims[i], dims[i + 1]) for i in range(len(dims[:-1]))]
        self.module.intercepts_ = [np.random.rand(dims[i + 1]) for i in range(len(dims[:-1]))]
        return self

"""The example ML model to train in a federated setting."""
import numpy as np
from flwr.common import NDArrays
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_is_fitted


class MyClassifier:
    def __init__(self, batch_size):
        """init."""
        self.module = MLPClassifier(
            solver="adam",
            alpha=1e-5,
            hidden_layer_sizes=(5, 2),
            batch_size=batch_size,
            verbose=True,
            warm_start=True,
            max_iter=1,
        )

    def fit(self, X, y, epochs):
        """Fit the model on a dataset."""
        for _ in range(epochs):
            self.module.fit(X, y)
        return self.module

    def predict_proba(self, X):
        """Predict for a dataset."""
        return self.module.predict_proba(X)

    def get_parameters(self) -> NDArrays:
        """Get the model's parameters as a list of numpy arrays."""
        try:
            check_is_fitted(self.module)
            coefs = self.module.coefs_
            intercepts = self.module.intercepts_
        except NotFittedError:
            # generate random parameters for initialization
            dims = (784, *self.module.hidden_layer_sizes, 10)
            coefs = [np.random.rand(dims[i], dims[i + 1]) for i in range(len(dims[:-1]))]
            intercepts = [np.random.rand(dims[i + 1]) for i in range(len(dims[:-1]))]
        return [val for pair in zip(coefs, intercepts) for val in pair]

    def set_parameters(self, parameters: NDArrays):
        """Set the model's parameters from a list of numpy arrays."""
        self.module.coefs_ = parameters[0::2]
        self.module.intercepts_ = parameters[1::2]
        return self

    def initialize_parameters(self, input_dim, output_dim):
        """Initialize the model parameters to start training."""
        self._initialize()
        dims = (input_dim, *self.module.hidden_layer_sizes, output_dim)
        self.module.coefs_ = [np.random.rand(dims[i], dims[i + 1]) for i in range(len(dims[:-1]))]
        self.module.intercepts_ = [np.random.rand(dims[i + 1]) for i in range(len(dims[:-1]))]
        self.module.n_layers_ = len(self.module.coefs_) + 1
        self.module.out_activation_ = "softmax"
        self.module.n_iter_ = 0
        self.module.t_ = 0
        self.module.loss_curve_ = []
        self.module.loss_ = np.inf
        self.module.best_loss_ = np.inf
        return self

"""The example ML model to train in a federated setting."""
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

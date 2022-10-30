# %%
import argparse

import numpy as np
from flwr.client import NumPyClient, start_numpy_client
from flwr.common.typing import Config, Dict, NDArrays, Scalar, Tuple
from loguru import logger
from sklearn.datasets import fetch_openml
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import MyClassifier


class MyClient(NumPyClient):
    def __init__(self, model, cid, train_data, test_data, epochs):
        self.model = model
        self.cid = cid
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.epochs = epochs

    def get_parameters(self, config=None):
        return self.model.get_parameters()

    def fit(self, parameters, config=None):
        self.model.set_parameters(parameters)
        for _ in tqdm(range(self.epochs)):
            self.model.fit(self.X_train, self.y_train, self.epochs)
        return self.model.get_parameters(), len(self.y_train), {}

    def evaluate(self, parameters, config=None):
        self.model.set_parameters(parameters)
        loss = log_loss(np.int32(self.y_test), self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return float(loss), len(self.y_test), {"accuracy": float(accuracy)}


def load_data(cid: int, num_clients: int):
    """Get a subset of the training data for one client."""
    logger.info("Fetching data...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X = X / 255.0  # normalize the pixel intensities to range [0,1]
    X_splits = np.array_split(X, num_clients)
    y_splits = np.array_split(y, num_clients)
    return X_splits[cid], y_splits[cid]


def start_client(cid: int, num_clients: int, batch_size: int, epochs: int):
    """Start a client for training."""
    model = MyClassifier(batch_size)
    data = load_data(cid, num_clients)
    # model.initialize_parameters(784, 10)
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2)
    client = MyClient(model, cid, (X_train, y_train), (X_test, y_test), epochs)
    logger.info("Starting client # {}", cid)
    start_numpy_client(server_address="0.0.0.0:8080", client=client)


def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(description="Flower Client for demo.")
    parser.add_argument(
        "cid",
        type=int,
        help="The current client ID. Used for splitting the dataset.",
    )
    parser.add_argument(
        "--num-clients",
        default=2,
        type=int,
        help="Total number of clients for federated training.",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--epochs", default=10, type=int, help="Epochs to run on each client round."
    )
    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = get_args()
    cid = int(args.cid)
    num_clients = int(args.num_clients)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    start_client(cid, num_clients, batch_size, epochs)

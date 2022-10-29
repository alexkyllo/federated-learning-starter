import argparse

from flwr.client import NumPyClient
from flwr.common.typing import Config, Dict, NDArrays, Scalar, Tuple
from loguru import logger
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from model import MyClassifier


class MyClient(NumPyClient):
    """A simple flower client for demonstration purposes."""

    def __init__(self, model, cid: int) -> None:
        self.model = model
        self.cid = cid

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set the PyTorch module parameters from a list of NumPy arrays.
        Parameters
        ----------
        parameters: List[numpy.ndarray]
            The desired local model parameters as a list of NumPy ndarrays.
        """
        # TODO

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : NDArrays
            The local model parameters as a list of NumPy ndarrays.
        """
        return

    def get_properties(self, config: Config) -> Dict[str, Scalar]:

        return

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        return

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        return


def load_data(client_id: int):
    """Get a subset of the training data for one client."""
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    return ((train_x, train_y), (test_x, test_y))


def start_client(cid: int, batch_size: int):
    """Start a client for training."""
    model = MyClassifier(batch_size)
    train_data, test_data = load_data(cid)
    client = MyClient(model, cid)
    logger.info("Starting client # {}", cid)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cid = int(args.cid)
    batch_size = int(args.batch_size)
    start_client(cid, batch_size)

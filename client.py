from flwr.client import NumPyClient
from flwr.common.typing import Config, Dict, NDArrays, Scalar, Tuple


class MyClient(NumPyClient):
    """A simple flower client for demonstration purposes."""

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


def main():
    client = MyClient()


if __name__ == "__main__":
    main()

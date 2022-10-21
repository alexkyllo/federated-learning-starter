"""A Flower Server."""
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import Parameters
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg


def get_initial_params() -> Parameters:
    """Get initial model parameters (weights)."""
    init_weights = []  # TODO, get these from the model
    init_param = ndarrays_to_parameters(init_weights)
    return init_param


def main():
    """Start the server."""
    rounds = 10
    min_fit_clients = 2
    min_available_clients = 2
    init_params = get_initial_params()
    strategy = FedAvg(
        fraction_fit=float(min_fit_clients / min_available_clients),
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        initial_parameters=init_params,
    )
    start_server(
        server_address="[::]:8080",
        config=ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

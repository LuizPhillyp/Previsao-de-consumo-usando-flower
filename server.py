import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Tuple
from flwr.common import FitRes


def main() -> None:

    # Define strategy
    strategy = fl.server.strategy.FedAvg(min_available_clients=4)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
"""adultnoniid: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from adultnoniid.task import load_model, get_num_clients, save_global_model, set_seed, SEED

def on_fit_config(server_round: int):
    """Passes the current round number to clients."""
    return {"server_round": server_round}

def on_evaluate(server_round, parameters, config):
    """Save the global model at the end of each round."""
    if server_round == 0:
        return 0.0, {"accuracy": 0.0}
    model = load_model()  # Load base model architecture
    model.set_weights(parameters)  # Convert Parameters to NumPy
    save_global_model(model, get_num_clients(), server_round)  # Save model

    return 0.0, {"accuracy": 0.0}

def server_fn(context: Context):
    set_seed(SEED)
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    model = load_model()
    save_global_model(model, get_num_clients(), 0)  # Save initial model

    # Define strategy
    strategy = strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_evaluate_config_fn=lambda r: {},
        on_fit_config_fn=on_fit_config,
        evaluate_fn=on_evaluate,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)

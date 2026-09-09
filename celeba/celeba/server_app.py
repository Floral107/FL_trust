"""celeba: A Flower / TensorFlow app."""
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from celeba.task import load_model, get_num_clients, save_global_model, set_seed, SEED

set_seed(SEED)

_STRATEGY = "fedavg"


def on_fit_config(server_round: int):
    return {"server_round": server_round}


def on_evaluate(server_round, parameters, config):
    if server_round == 0:
        return 0.0, {"accuracy": 0.0}

    client_num = get_num_clients()
    model = load_model()
    model.set_weights(parameters)

    save_global_model(model, client_num, server_round, strategy=_STRATEGY)

    return 0.0, {"accuracy": 0.0}


def server_fn(context):
    set_seed(SEED)
    num_rounds = context.run_config["num-server-rounds"]
    strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()
    proximal_mu = float(context.run_config.get("proximal-mu", 0.0))

    global _STRATEGY
    _STRATEGY = strategy_name

    parameters = ndarrays_to_parameters(load_model().get_weights())

    client_num = get_num_clients()
    model = load_model()
    save_global_model(model, client_num, 0, strategy=strategy_name)

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_evaluate_config_fn=lambda r: {},
        on_fit_config_fn=on_fit_config,
        evaluate_fn=on_evaluate,
    )
    if strategy_name == "fedprox":
        strategy = FedProx(proximal_mu=proximal_mu, **common_kwargs)
    else:
        strategy = FedAvg(**common_kwargs)

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)

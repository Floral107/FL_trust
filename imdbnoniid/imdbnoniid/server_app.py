"""imdbnoniid: A Flower / TensorFlow app."""
import sys
from pathlib import Path

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from imdbnoniid.task import (
    load_model, get_num_clients, save_global_model, load_data, process_text,
)
import logging

# Make sibling repo-root modules importable (weighted_strategy etc.)
_PROJ_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# Suppress Ray logs
logging.getLogger("ray").setLevel(logging.ERROR)

_STRATEGY = "fedavg"
_ET_MODE = "none"
_ET_CONFIG = {}
_WEIGHT_MODE = "none"
_WEIGHT_METRIC = ""


def on_fit_config(server_round: int):
    cfg = {"server_round": server_round}
    cfg.update(_ET_CONFIG)
    cfg["weight_mode"] = _WEIGHT_MODE
    cfg["weight_metric"] = _WEIGHT_METRIC
    return cfg


def on_evaluate(server_round, parameters, config):
    """Save the global model at the end of each round."""
    if server_round == 0:
        return 0.0, {"accuracy": 0.0}
    model = load_model()
    model.set_weights(parameters)
    save_global_model(model, get_num_clients(), server_round,
                      strategy=_STRATEGY, et_mode=_ET_MODE,
                      weight_mode=_WEIGHT_MODE, weight_metric=_WEIGHT_METRIC)
    return 0.0, {"accuracy": 0.0}


def _with_agg_save(base_cls):
    """Wrap a strategy so each round's global model is ALSO saved in
    aggregate_fit. aggregate_fit is guaranteed to run whenever a round
    produces results; the centralized evaluate callback (on_evaluate) --
    previously the ONLY place the global ckpt was written -- can be skipped
    near round_timeout, which produced transient 'FAIL -- no valid ckpt
    (rc=0)' false-failures on heavy configs. on_evaluate's save is kept;
    this just makes the write unconditional."""
    class _AggSave(base_cls):
        def aggregate_fit(self, server_round, results, failures):
            agg_params, metrics = super().aggregate_fit(server_round, results, failures)
            if agg_params is not None:
                try:
                    from flwr.common import parameters_to_ndarrays as _p2n
                    model = load_model()
                    model.set_weights(_p2n(agg_params))
                    save_global_model(model, get_num_clients(), server_round,
                                      strategy=_STRATEGY, et_mode=_ET_MODE,
                                      weight_mode=_WEIGHT_MODE, weight_metric=_WEIGHT_METRIC)
                except Exception as e:
                    print(f"[agg-save] round {server_round} save failed: {e}")
            return agg_params, metrics
    return _AggSave


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()
    proximal_mu = float(context.run_config.get("proximal-mu", 0.0))
    et_mode = str(context.run_config.get("et-mode", "none")).lower()
    weight_mode = str(context.run_config.get("weight-mode", "none")).lower()
    weight_metric = str(context.run_config.get("weight-metric", "")).lower()
    weight_beta = float(context.run_config.get("weight-beta", 1.0))
    weight_cap = float(context.run_config.get("weight-cap", 0.0))

    global _STRATEGY, _ET_MODE, _ET_CONFIG, _WEIGHT_MODE, _WEIGHT_METRIC
    _STRATEGY = strategy_name
    _ET_MODE = et_mode
    _WEIGHT_MODE = weight_mode
    _WEIGHT_METRIC = weight_metric
    _ET_CONFIG = {
        "et_mode": et_mode,
        "proximal_mu": proximal_mu,
        "et_lambda_fair": float(context.run_config.get("et-lambda-fair", 0.1)),
        "et_adv_ratio":   float(context.run_config.get("et-adv-ratio", 0.5)),
        "et_adv_eps":     float(context.run_config.get("et-adv-eps", 0.0015)),
        "et_adv_alpha":   float(context.run_config.get("et-adv-alpha", 0.0015)),
        "et_adv_steps":   int(context.run_config.get("et-adv-steps", 7)),
        "et_dp_clip":     float(context.run_config.get("et-dp-clip", 1.0)),
        "et_dp_sigma":    float(context.run_config.get("et-dp-sigma", 1.0)),
    }

    parameters = ndarrays_to_parameters(load_model().get_weights())

    model = load_model()
    save_global_model(model, get_num_clients(), 0,
                      strategy=strategy_name, et_mode=et_mode,
                      weight_mode=weight_mode, weight_metric=weight_metric)

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_evaluate_config_fn=lambda r: {},
        on_fit_config_fn=on_fit_config,
        evaluate_fn=on_evaluate,
    )

    # Strategy precedence: ET > weight-mode (ST/DY) > fedprox > fedavg
    if et_mode in ("fair", "adv", "dp"):
        strategy = _with_agg_save(FedAvg)(**common_kwargs)
    elif weight_mode in ("st", "dy"):
        from weighted_strategy import WeightedFedAvg
        from inloop_scoring import build_gtg_score_provider
        from score_metrics import build_metric

        # IMDB load_data signature: (partition_id, num_partitions) -> tuple
        # CRITICAL: IMDB x_test is raw TEXT — must be embedded via process_text
        # before the model can consume it. Other datasets pass x_test as-is.
        data = load_data(0, 1)
        x_test_raw, y_test = data[2], data[3]
        x_test = process_text(x_test_raw)  # text -> embeddings
        # IMDB has no sensitive attribute; fairness metrics will be skipped at metric build
        s_test = None

        metric_fn = build_metric(weight_metric, x_test, y_test, s_test=s_test,
                                 dataset="imdbnoniid")
        score_provider = build_gtg_score_provider(
            metric_fn=metric_fn,
            model_factory=load_model,
            max_permutations=30,
        )
        strategy = _with_agg_save(WeightedFedAvg)(
            weight_mode=weight_mode,
            score_provider=score_provider,
            shrink_beta=weight_beta,
            shrink_cap=weight_cap,
            **common_kwargs,
        )
    elif strategy_name == "fedprox":
        strategy = _with_agg_save(FedProx)(proximal_mu=proximal_mu, **common_kwargs)
    else:
        strategy = _with_agg_save(FedAvg)(**common_kwargs)

    config = ServerConfig(num_rounds=num_rounds, round_timeout=7200)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)

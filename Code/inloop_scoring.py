"""inloop_scoring.py — compute per-client GTG-Shapley scores DURING FL training.

Unlike the post-hoc scoring in robustness.py (which loads ckpts and re-evaluates
afterwards), this module is invoked from inside a WeightedFedAvg.aggregate_fit
to produce the scores needed for ST (round 2) and DY (every round).

Design:
    build_score_provider(eval_set, metric_fn, model_factory, num_clients, ...)
        returns a callable suitable for WeightedFedAvg.score_provider:
            (server_round, [(params, num_examples), ...]) -> [score_0, ..., score_{N-1}]

The callable:
    1. Builds the GTG-Shapley utility function `u(subset)`:
         - aggregate the subset's params via size-weighted FedAvg
         - set them on a fresh model
         - score the model via metric_fn(model, eval_set)
    2. Runs GTGShapley.compute(round_num) -> per-client raw scores
    3. Returns the scores in the same order as the input client list

The model_factory must return a *fresh* Keras model with the architecture
the clients trained — it just needs the weights set onto it.

The metric_fn signature is:
    metric_fn(model, eval_set) -> float

For accuracy metric_fn, eval_set is (x_test, y_test); for fairness it's
(x_test, y_test, s_test); etc. The factory hides the signature variance from
the strategy.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple
import logging

import numpy as np

from gtg_shap import GTGShapley
from weighted_utils import fedavg_aggregate

log = logging.getLogger("InloopScoring")
log.setLevel(logging.INFO)


# Type aliases (avoid flwr dependency here — params are list-of-ndarrays)
NDArrays = List[np.ndarray]


def build_gtg_score_provider(
    metric_fn: Callable[[object], float],
    model_factory: Callable[[], object],
    *,
    last_round_utility: float = 0.0,
    loss_flag: bool = False,
    max_permutations: int = 30,
    converge_min: int = 30,
    eps: float = 0.001,
    round_trunc_threshold: float = 0.002,
    convergence_criteria: float = 0.05,
) -> Callable[[int, Sequence[Tuple[NDArrays, int]]], List[float]]:
    """Build a score_provider closure for WeightedFedAvg.

    Args:
        metric_fn:       callable(model) -> float; evaluation of a model on the
                         held-out set for the target metric. The caller wires up
                         the data/labels/sensitive attr inside this closure so
                         the in-loop machinery sees only `model -> float`.
        model_factory:   callable() -> fresh Keras model; used to instantiate a
                         model and set weights for each coalition evaluation.
        last_round_utility: passed to GTGShapley to enable cross-round
                         truncation. Kept stateful across rounds via the
                         closure's `state` dict.
        loss_flag:       True if metric is loss (lower = better). GTGShapley
                         uses this to flip the "best subset" selection.
        max_permutations, converge_min, eps, round_trunc_threshold,
        convergence_criteria: GTGShapley knobs; defaults match the paper.

    Returns:
        score_provider closure: (server_round, payload) -> per-client scores.
        - server_round: int
        - payload: list of (params, num_examples) in the order the strategy
          uses for aggregation.
    """
    state = {"last_round_utility": float(last_round_utility)}

    def score_provider(
        server_round: int,
        payload: Sequence[Tuple[NDArrays, int]],
    ) -> List[float]:
        client_params: List[NDArrays] = [p for p, _ in payload]
        client_ns: List[int] = [n for _, n in payload]
        N = len(client_params)
        if N == 0:
            return []

        # Build utility function for GTG-Shapley
        # u(subset) := metric_fn(model with params = FedAvg(subset))
        # Empty subset: use a freshly-initialised model (the "no-client" baseline).
        # Keep the same model instance across calls to avoid re-allocation.
        model = model_factory()
        # Save the freshly-initialised weights for u({})
        init_weights = model.get_weights()

        def utility_function(subset: List[int]) -> float:
            if not subset:
                # Empty subset = freshly-initialised model.
                model.set_weights(init_weights)
                return float(metric_fn(model))
            subset_params = [client_params[i] for i in subset]
            subset_ns = [client_ns[i] for i in subset]
            agg = fedavg_aggregate(subset_params, subset_ns)
            model.set_weights(agg)
            return float(metric_fn(model))

        gtg = GTGShapley(
            num_players=N,
            last_round_utility=state["last_round_utility"],
            eps=eps,
            round_trunc_threshold=round_trunc_threshold,
            convergence_criteria=convergence_criteria,
            max_permutations=max_permutations,
            converge_min=converge_min,
            loss_flag=loss_flag,
        )
        gtg.set_utility_function(utility_function)
        # return_raw=True: every client gets a Shapley estimate (negative ones
        # included). ST/DY want full ranking across clients, not best-subset
        # truncation (which would zero out non-top clients and degenerate the
        # weighted aggregation to "only top client contributes").
        scores = gtg.compute(round_num=server_round, return_raw=True)
        state["last_round_utility"] = float(gtg.last_round_utility)
        log.info(
            f"[GTG-raw] round={server_round} scores={[f'{s:.4f}' for s in scores]} "
            f"last_utility={state['last_round_utility']:.4f}"
        )
        return list(scores)

    return score_provider

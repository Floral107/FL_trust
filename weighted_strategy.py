"""weighted_strategy.py — Score-driven FedAvg variants for ST and DY.

Implements the algorithm described in MAIN.tex §Score Utilization:
  ST: at end of round 2, compute scores → derive weights → freeze.
      Rounds 1, 2 use standard FedAvg.
      Rounds 3..N use frozen weights.
  DY: at end of round X (X≥2), compute fresh scores from round-X params →
      derive weights → use them for round X's aggregation. Weights are not
      carried over. Round 1 uses standard FedAvg.

Weight transform (paper eq.):
    s' = s + |min(0, min_k s_k)|     # shift non-negative
    w  = s' / mean(s')               # normalise to ≈1

Aggregation (Option (a), size × weight):
    θ_X = Σ_k (n_k · w_k / Σ_j n_j w_j) · θ_k^X

Implementation note: the strategy holds a `score_provider` callable that,
given (server_round, list-of-(params, num_examples)), returns per-client raw
scores. The provider is responsible for in-loop GTG/LOO score computation
(see inloop_scoring.py). This keeps the strategy itself agnostic to the
underlying scoring method or target metric.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import logging

import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Pure-math helpers are in weighted_utils so they're importable without flwr/TF.
from weighted_utils import (
    shift_and_normalize,
    weighted_aggregate,
    fedavg_aggregate,
)

log = logging.getLogger("WeightedFedAvg")
log.setLevel(logging.INFO)


# ─── Flower strategy ─────────────────────────────────────────────────────────

ScoreProvider = Callable[
    [int, List[Tuple[NDArrays, int]]],  # (server_round, [(params, num_examples), ...])
    List[float],                         # per-client raw scores aligned to input list
]


class WeightedFedAvg(FedAvg):
    """FedAvg with optional score-driven weighting (BL / ST / DY).

    Args:
        weight_mode: "none" | "st" | "dy"
        score_provider: callable returning per-client raw scores given round +
            client params; only invoked when weighting is needed:
              ST → invoked once, at end of round 2
              DY → invoked every round X ≥ 2
              BL ("none") → never invoked
        **fedavg_kwargs: passed through to FedAvg
    """

    def __init__(
        self,
        *,
        weight_mode: str = "none",
        score_provider: Optional[ScoreProvider] = None,
        shrink_beta: float = 1.0,
        shrink_cap: float = 0.0,
        **fedavg_kwargs,
    ):
        super().__init__(**fedavg_kwargs)
        wm = (weight_mode or "none").lower()
        if wm not in ("none", "st", "dy"):
            raise ValueError(f"weight_mode must be one of none/st/dy, got {wm!r}")
        if wm != "none" and score_provider is None:
            raise ValueError(f"weight_mode={wm!r} requires a score_provider")
        if not (0.0 <= shrink_beta <= 1.0):
            raise ValueError(f"shrink_beta must be in [0,1], got {shrink_beta!r}")
        if shrink_cap and shrink_cap < 1.0:
            raise ValueError(f"shrink_cap must be >= 1 (or 0 to disable), got {shrink_cap!r}")
        self.weight_mode = wm
        self.score_provider = score_provider
        # Shrinkage toward uniform: w ← (1-β)+β·w. β=1 → legacy raw transform,
        # β=0 → plain FedAvg. Floors every weight at (1-β) so GTG one-hot scores
        # can no longer collapse the aggregation onto a single client. Replaces
        # the post-hoc do-no-harm guard with a collapse-proof transform.
        self.shrink_beta = shrink_beta
        # Hard multiplicative cap (>=1, or 0 to disable). Clips each client weight
        # to [1/cap, cap] AFTER shrinkage. Unlike beta (which only floors the low
        # end — the top weight can still reach (1-beta)+beta·K), the cap bounds
        # the *top* weight too, so a near-one-hot GTG score can never let a single
        # client capture the aggregation regardless of K. cap=2 → top client
        # gets at most 2× the uniform share.
        self.shrink_cap = shrink_cap
        # ST state: weights computed at end of round 2 and frozen.
        # Indexed by client_id (the partition_id from ClientProxy.cid / metrics).
        self._frozen_weights: Optional[List[float]] = None
        self._frozen_client_order: Optional[List[str]] = None

    # ---------------------------------------------------------------------
    # FedAvg.aggregate_fit override
    # ---------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}

        # Sort by client cid for determinism. Flower doesn't guarantee result order.
        results_sorted = sorted(results, key=lambda r: r[0].cid)

        # Pull params and counts in the sorted order
        client_params: List[NDArrays] = [
            parameters_to_ndarrays(fitres.parameters) for _, fitres in results_sorted
        ]
        client_num_examples: List[int] = [fitres.num_examples for _, fitres in results_sorted]
        client_cids: List[str] = [c.cid for c, _ in results_sorted]

        # Decide what weights to use this round
        weights_used: Optional[List[float]] = None
        if self.weight_mode == "none":
            # plain FedAvg — let parent handle it
            pass

        elif self.weight_mode == "st":
            if server_round <= 2:
                # Rounds 1, 2: standard FedAvg
                # At END of round 2: compute scores, freeze weights for rounds 3+.
                if server_round == 2 and self._frozen_weights is None:
                    log.info("[ST] computing scores at round 2 to freeze weights…")
                    raw_scores = self.score_provider(
                        server_round,
                        list(zip(client_params, client_num_examples)),
                    )
                    self._frozen_weights = shift_and_normalize(
                        raw_scores, beta=self.shrink_beta, cap=self.shrink_cap)
                    self._frozen_client_order = client_cids
                    log.info(f"[ST] frozen weights (beta={self.shrink_beta}, cap={self.shrink_cap}) = {self._frozen_weights}")
                # but for round 2 itself, still use standard FedAvg
            else:
                # Round ≥ 3: use frozen weights
                if self._frozen_weights is None:
                    log.warning(
                        f"[ST] round {server_round}: no frozen weights (round-2 likely "
                        "produced no scores). Falling back to standard FedAvg."
                    )
                else:
                    # Re-align frozen weights to current client order
                    weights_used = self._align_weights(
                        self._frozen_weights, self._frozen_client_order, client_cids
                    )

        elif self.weight_mode == "dy":
            if server_round >= 2:
                log.info(f"[DY] round {server_round}: computing fresh scores…")
                raw_scores = self.score_provider(
                    server_round,
                    list(zip(client_params, client_num_examples)),
                )
                weights_used = shift_and_normalize(
                    raw_scores, beta=self.shrink_beta, cap=self.shrink_cap)
                log.info(f"[DY] round {server_round} weights (beta={self.shrink_beta}, cap={self.shrink_cap}) = {weights_used}")
            # else: round 1 → standard FedAvg

        # ---- aggregate ----
        # Unconditional stdout diagnostic (NOT logging — several apps' logging
        # configs swallow this module's log.info lines, which made the first
        # sweep's dropout invisible). One line per round in every cell log:
        # participation + weight-vector length is the ground truth for the
        # ST/DY definition audit.
        print(f"[WFA] round={server_round} mode={self.weight_mode} "
              f"n_results={len(results)} cids={client_cids[:3]}... "
              f"wlen={len(weights_used) if weights_used is not None else 0} "
              f"w={weights_used}", flush=True)
        if weights_used is None:
            # Standard FedAvg
            agg_params = self._aggregate_fedavg(client_params, client_num_examples)
        else:
            agg_params = weighted_aggregate(client_params, client_num_examples, weights_used)

        # Build metrics dict (forward any client-side fit_metrics if needed; for
        # now we just report the weights for diagnostics).
        metrics: dict = {}
        if weights_used is not None:
            metrics["weights"] = ",".join(f"{w:.4f}" for w in weights_used)
        return ndarrays_to_parameters(agg_params), metrics

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _aggregate_fedavg(
        client_params: List[NDArrays], client_num_examples: List[int]
    ) -> NDArrays:
        """Plain size-weighted FedAvg (the default when no score weighting)."""
        total = float(sum(client_num_examples))
        coeffs = [float(n) / total for n in client_num_examples]
        out = []
        for layer_idx in range(len(client_params[0])):
            layer_acc = sum(
                coeffs[k] * client_params[k][layer_idx] for k in range(len(client_params))
            )
            out.append(layer_acc)
        return out

    @staticmethod
    def adopt_reweighted_or_fallback(
        reweighted_acc: float, baseline_acc: float, delta: float = 0.1
    ) -> bool:
        """Do-no-harm safeguard for the ST/DY scenarios (post-training selection).

        The score transformation (``shift_and_normalize``) is unbounded, so in the
        degenerate limit where GTG truncation drives the contribution scores toward
        one-hot, almost all aggregation weight lands on a single client and the
        reweighted global model can collapse. To keep the intervention safe by
        construction, the trustworthiness-reweighted model is ADOPTED only when its
        held-out accuracy stays within ``delta`` of the size-weighted baseline;
        otherwise the baseline (uniform FedAvg) model is reported for that setting.

        Returns True to adopt the reweighted model, False to fall back to baseline.
        The decision depends only on the two final-model accuracies, so it is
        applied deterministically at model-selection / reporting time (the Tab.4
        application lives in ``build_tab4_final.py`` using the same rule).
        """
        return (baseline_acc - reweighted_acc) <= delta

    @staticmethod
    def _align_weights(
        frozen_weights: List[float],
        frozen_order: List[str],
        current_order: List[str],
    ) -> List[float]:
        """Reorder frozen_weights to match current_order. Falls back to 1.0 for
        any cid not in the frozen ordering (defensive — shouldn't happen)."""
        lookup = dict(zip(frozen_order, frozen_weights))
        return [lookup.get(cid, 1.0) for cid in current_order]

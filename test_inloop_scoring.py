"""test_inloop_scoring.py — TF-free smoke test of inloop_scoring.

We define a *fake* "model" that's just a numpy vector (its "weights"), and a
metric_fn that computes a deterministic toy utility from those weights. This
exercises GTGShapley + fedavg_aggregate + the score_provider plumbing without
touching TensorFlow.
"""
import numpy as np

from inloop_scoring import build_gtg_score_provider


class FakeModel:
    """Minimal stand-in for a Keras model: stores one weight array."""

    def __init__(self, init_weights):
        # init_weights: list with single ndarray (mimics keras "layers")
        self._w = [np.array(init_weights[0], dtype=float)]

    def get_weights(self):
        return [w.copy() for w in self._w]

    def set_weights(self, weights):
        self._w = [np.array(w, dtype=float) for w in weights]


def _factory_initial_zero(dim):
    def _f():
        return FakeModel([np.zeros(dim)])
    return _f


def test_gtg_provider_single_client():
    """1 client: GTG should return a single score for that client.

    Setup: client-0 brings params=[1,1,1]. Empty utility = 0 (zero model).
    Full utility = metric(model with [1,1,1]) = 3 (just sum elements).
    So contribution of client-0 = 3 - 0 = 3.
    """
    def metric_fn(model):
        # toy metric: sum of weights (deterministic, lower = worse)
        return float(np.sum(model.get_weights()[0]))

    provider = build_gtg_score_provider(
        metric_fn=metric_fn,
        model_factory=_factory_initial_zero(3),
        max_permutations=10,    # plenty for 1 player
        converge_min=2,
    )
    payload = [(np.array([[1.0, 1.0, 1.0]], dtype=object).tolist()[0:1], 100)]
    # payload format: [(params, num_examples), ...] — params is "list of layers"
    payload = [([np.array([1.0, 1.0, 1.0])], 100)]
    scores = provider(server_round=1, payload=payload)
    assert len(scores) == 1
    assert abs(scores[0] - 3.0) < 1e-6, f"expected ~3.0, got {scores[0]}"
    print(f"OK test_gtg_provider_single_client: scores={scores}")


def test_gtg_provider_two_clients_symmetric_raw():
    """2 clients with IDENTICAL params — raw Shapley should be symmetric.

    Both alone give util=3, together also util=3 (FedAvg of identical params).
    Marginal contributions average out: each client gets ~1.5 raw Shapley.
    (No best-subset truncation since return_raw=True.)
    """
    def metric_fn(model):
        return float(np.sum(model.get_weights()[0]))

    p = [np.array([1.0, 1.0, 1.0])]
    provider = build_gtg_score_provider(
        metric_fn=metric_fn,
        model_factory=_factory_initial_zero(3),
        max_permutations=10,
        converge_min=2,
    )
    payload = [(p, 100), (p, 100)]
    scores = provider(server_round=1, payload=payload)
    assert len(scores) == 2
    # With raw Shapley, identical clients should get equal scores
    assert abs(scores[0] - scores[1]) < 1e-6, \
        f"identical clients should have equal raw Shapley, got {scores}"
    assert abs(scores[0] - 1.5) < 0.5, \
        f"expected ~1.5 per client, got {scores[0]}"
    print(f"OK test_gtg_provider_two_clients_symmetric_raw: scores={scores}")


def test_gtg_provider_two_clients_asymmetric():
    """2 clients with DIFFERENT params: high-utility client outscores low-utility.

    Client A: [3,0,0] -> alone utility = 3 (high)
    Client B: [0,0,0] -> alone utility = 0 (none)
    Raw Shapley:
      φ(A) = 1/2 * (u({A}) - u({})) + 1/2 * (u({A,B}) - u({B}))
           = 1/2 * 3 + 1/2 * 1.5 = 2.25
      φ(B) = 1/2 * (u({B}) - u({})) + 1/2 * (u({A,B}) - u({A}))
           = 1/2 * 0 + 1/2 * (-1.5) = -0.75
      Sum = 1.5 = u({A,B}) - u({}) ✓ (efficiency)
    With return_raw=True, B's score CAN be negative now.
    """
    def metric_fn(model):
        return float(np.sum(model.get_weights()[0]))

    pA = [np.array([3.0, 0.0, 0.0])]
    pB = [np.array([0.0, 0.0, 0.0])]
    provider = build_gtg_score_provider(
        metric_fn=metric_fn,
        model_factory=_factory_initial_zero(3),
        max_permutations=20,
        converge_min=4,
    )
    payload = [(pA, 100), (pB, 100)]
    scores = provider(server_round=1, payload=payload)
    assert len(scores) == 2
    assert scores[0] > scores[1], (
        f"client A (high-utility) should outscore client B, got {scores}"
    )
    # Efficiency: sum should be ~marginal_gain = 1.5
    assert abs(scores[0] + scores[1] - 1.5) < 0.3, \
        f"sum should approximate marginal gain 1.5, got {sum(scores)}"
    print(f"OK test_gtg_provider_two_clients_asymmetric: "
          f"scores=[{scores[0]:.4f}, {scores[1]:.4f}] sum={sum(scores):.4f} "
          f"(efficiency OK)")


def test_gtg_provider_three_clients_ordering():
    """3 clients with monotonically increasing utility contributions.

    With raw Shapley, every client gets a value reflecting their relative
    contribution. Ranking A < B < C should hold.
    """
    def metric_fn(model):
        return float(np.sum(model.get_weights()[0]))

    pA = [np.array([0.0, 0.0, 0.0])]  # adds nothing
    pB = [np.array([1.0, 1.0, 1.0])]  # mid
    pC = [np.array([3.0, 3.0, 3.0])]  # most
    provider = build_gtg_score_provider(
        metric_fn=metric_fn,
        model_factory=_factory_initial_zero(3),
        max_permutations=30,
        converge_min=6,
    )
    payload = [(pA, 100), (pB, 100), (pC, 100)]
    scores = provider(server_round=1, payload=payload)
    assert len(scores) == 3
    # Expected ordering: A < B < C (low-contributor smallest score)
    # With raw Shapley, A may be 0 or negative; the inequalities should still hold.
    assert scores[0] < scores[1], f"A should rank below B, got {scores}"
    assert scores[1] < scores[2], f"B should rank below C, got {scores}"
    print(f"OK test_gtg_provider_three_clients_ordering: scores={[f'{s:.4f}' for s in scores]} (A<B<C)")


def test_round_truncation():
    """When the round's utility doesn't move appreciably, all scores -> 0.

    Set last_round_utility = current utility (≈ no change). GTGShapley
    short-circuits and returns zeros.
    """
    def metric_fn(model):
        return float(np.sum(model.get_weights()[0]))

    pA = [np.array([1.0, 1.0, 1.0])]
    pB = [np.array([1.0, 1.0, 1.0])]
    # current_utility for the full coalition will be 3.0 (avg = [1,1,1])
    provider = build_gtg_score_provider(
        metric_fn=metric_fn,
        model_factory=_factory_initial_zero(3),
        last_round_utility=3.0,         # same as current → trigger truncation
        round_trunc_threshold=0.01,
        max_permutations=10,
        converge_min=2,
    )
    payload = [(pA, 100), (pB, 100)]
    scores = provider(server_round=2, payload=payload)
    assert all(abs(s) < 1e-9 for s in scores), \
        f"expected all zeros under round-trunc, got {scores}"
    print(f"OK test_round_truncation: scores={scores}")


if __name__ == "__main__":
    print("--- inloop_scoring tests ---")
    test_gtg_provider_single_client()
    test_gtg_provider_two_clients_symmetric_raw()
    test_gtg_provider_two_clients_asymmetric()
    test_gtg_provider_three_clients_ordering()
    test_round_truncation()
    print()
    print("All tests passed OK")

"""test_weighted_strategy.py — unit tests for ST/DY helpers + strategy.

Pure-math tests import from weighted_utils (no flwr/TF needed → run on any box).
Strategy state-machine tests import from weighted_strategy (needs flwr + TF).
On boxes where flwr/TF can't import (this laptop has a TF DLL issue), the
strategy tests gracefully skip; the math tests still run.

Run with:  python test_weighted_strategy.py
or:        python -m pytest test_weighted_strategy.py
"""
import numpy as np
import sys

# Pure-math helpers — always importable
from weighted_utils import (
    shift_and_normalize,
    weighted_aggregate,
    fedavg_aggregate,
)

# Strategy — may fail if flwr/TF unavailable. Gracefully degrade.
try:
    from weighted_strategy import WeightedFedAvg
    _STRATEGY_AVAILABLE = True
except Exception as e:
    print(f"[note] WeightedFedAvg import failed ({type(e).__name__}: {e}); "
          "strategy state-machine tests will be SKIPPED.")
    _STRATEGY_AVAILABLE = False
    WeightedFedAvg = None  # placeholder so name resolves


def _assert_close(a, b, tol=1e-6, msg=""):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise AssertionError(f"{msg}: shape mismatch {a.shape} vs {b.shape}")
    diff = float(np.max(np.abs(a - b)))
    if diff > tol:
        raise AssertionError(f"{msg}: max diff {diff} > {tol}\n  got: {a}\n  exp: {b}")


def test_shift_and_normalize_positive_scores():
    """Already-positive scores: just divide by mean."""
    out = shift_and_normalize([1.0, 2.0, 3.0])
    # mean = 2.0 → [0.5, 1.0, 1.5]
    _assert_close(out, [0.5, 1.0, 1.5], msg="positive_scores")
    print("OK test_shift_and_normalize_positive_scores")


def test_shift_and_normalize_mixed_signs():
    """Negative + positive: shift so min becomes 0, then divide by mean."""
    out = shift_and_normalize([-1.0, 0.0, 1.0, 2.0])
    # shift by 1 → [0, 1, 2, 3], mean = 1.5 → [0, 0.667, 1.333, 2.0]
    _assert_close(out, [0.0, 2/3, 4/3, 2.0], msg="mixed_signs")
    print("OK test_shift_and_normalize_mixed_signs")


def test_shift_and_normalize_all_zero():
    """All zero → uniform 1.0 (degenerate case → fall back to no-op)."""
    out = shift_and_normalize([0.0, 0.0, 0.0])
    _assert_close(out, [1.0, 1.0, 1.0], msg="all_zero")
    print("OK test_shift_and_normalize_all_zero")


def test_shift_and_normalize_all_negative():
    """All-negative scores → shift all to non-negative."""
    out = shift_and_normalize([-3.0, -2.0, -1.0])
    # shift by 3 → [0, 1, 2], mean = 1 → [0, 1, 2]
    _assert_close(out, [0.0, 1.0, 2.0], msg="all_negative")
    print("OK test_shift_and_normalize_all_negative")


def test_cap_bounds_one_hot_scores():
    """The cap must keep one-hot GTG scores from collapsing the aggregation.

    Simulates the failure mode: K=20 clients, GTG truncation gives one client a
    big score and the rest ~0. Without a cap the top weight is ~K; with cap=2 the
    top:bottom ratio is bounded by cap^2=4 and the mean stays ~1.
    """
    K = 20
    scores = [0.0] * K
    scores[7] = 1.0  # one-hot: only client 7 has positive Shapley

    # uncapped (legacy beta=1): top weight blows up toward K
    raw = shift_and_normalize(scores, beta=1.0, cap=0.0)
    assert max(raw) > 10.0, f"expected blow-up without cap, got max={max(raw)}"

    # capped at 2: top client gets at most 2x uniform, bottom at least 0.5x.
    # (No renorm — aggregation is scale-invariant — so the bound is exact.)
    capped = shift_and_normalize(scores, beta=1.0, cap=2.0)
    assert max(capped) <= 2.0 + 1e-9, f"cap violated on top: {max(capped)}"
    assert min(capped) >= 0.5 - 1e-9, f"cap violated on bottom: {min(capped)}"
    # top:bottom ratio bounded by cap^2 = 4
    assert max(capped) / min(capped) <= 4.0 + 1e-9
    print(f"OK test_cap_bounds_one_hot_scores (uncapped max={max(raw):.2f} -> "
          f"capped max={max(capped):.3f}, ratio={max(capped)/min(capped):.2f})")


def test_cap_disabled_is_noop():
    """cap=0 must reproduce the legacy transform exactly (no behavior change)."""
    s = [-1.0, 0.0, 1.0, 2.0]
    _assert_close(shift_and_normalize(s, beta=1.0, cap=0.0),
                  shift_and_normalize(s, beta=1.0), msg="cap=0 noop")
    print("OK test_cap_disabled_is_noop")


def test_weighted_aggregate_uniform_weights_equals_fedavg():
    """When all w_k = 1, weighted_aggregate must reduce to standard size-weighted FedAvg."""
    # 3 clients, single-layer "weights" (just arrays)
    p1 = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    p2 = [np.array([[5.0, 6.0], [7.0, 8.0]])]
    p3 = [np.array([[9.0, 10.0], [11.0, 12.0]])]
    params = [p1, p2, p3]
    ns = [10, 20, 30]  # totals 60
    ws = [1.0, 1.0, 1.0]

    out = weighted_aggregate(params, ns, ws)
    # expected: (10·p1 + 20·p2 + 30·p3) / 60
    exp = (10 * p1[0] + 20 * p2[0] + 30 * p3[0]) / 60.0
    _assert_close(out[0], exp, msg="uniform_eq_fedavg")
    print("OK test_weighted_aggregate_uniform_weights_equals_fedavg")


def test_weighted_aggregate_doubles_weight_doubles_share():
    """If one client's w doubles relative to others, its share roughly doubles."""
    p1 = [np.array([1.0, 0.0])]  # contributes "1" to dim 0
    p2 = [np.array([0.0, 1.0])]  # contributes "1" to dim 1
    params = [p1, p2]
    ns = [100, 100]  # equal sizes, so n won't bias

    # uniform weights → equal split, output should be [0.5, 0.5]
    out_eq = weighted_aggregate(params, ns, [1.0, 1.0])
    _assert_close(out_eq[0], [0.5, 0.5], msg="weighted_aggregate eq weights")

    # double weight on client 1 → its share is ~2/3
    out_db = weighted_aggregate(params, ns, [2.0, 1.0])
    # coeffs = [200/300, 100/300] = [2/3, 1/3]
    _assert_close(out_db[0], [2/3, 1/3], msg="weighted_aggregate doubled weight")
    print("OK test_weighted_aggregate_doubles_weight_doubles_share")


def test_weighted_aggregate_zero_weights_falls_back_uniform():
    """All weights zero → must NOT divide by zero; falls back to equal coefficients."""
    p1 = [np.array([1.0, 1.0])]
    p2 = [np.array([3.0, 3.0])]
    out = weighted_aggregate([p1, p2], [50, 50], [0.0, 0.0])
    # equal coefficients → (1+3)/2 = 2
    _assert_close(out[0], [2.0, 2.0], msg="zero_weights_fallback")
    print("OK test_weighted_aggregate_zero_weights_falls_back_uniform")


def test_size_x_weight_modulation():
    """Option (a): w_k > 1 amplifies the natural size-weighted share."""
    # Client A: small data + low score → tiny influence
    # Client B: large data + high score → dominates
    p_A = [np.array([0.0])]
    p_B = [np.array([1.0])]
    ns = [10, 90]   # B dominates by 9×
    ws = [0.5, 1.5]  # B's score is 3× A's

    out = weighted_aggregate([p_A, p_B], ns, ws)
    # raw = [10*0.5, 90*1.5] = [5, 135]; total = 140; coeffs = [5/140, 135/140]
    exp = (5/140) * 0.0 + (135/140) * 1.0
    _assert_close(out[0], [exp], msg="size_x_weight")
    # Check that B's share is now even larger than under plain FedAvg (90/100=0.9)
    assert exp > 0.9, f"Expected B's share > 0.9 under boosted weights, got {exp}"
    print(f"OK test_size_x_weight_modulation (B's share went from 0.900 -> {exp:.4f})")


# ─── strategy-state tests (mock score_provider; no Flower-server boot) ───

class _MockResult:
    """Minimal stand-in for (ClientProxy, FitRes) tuple to drive aggregate_fit."""
    def __init__(self, cid, params, num_examples):
        from flwr.common import ndarrays_to_parameters, Code, Status
        # FitRes wants: status, parameters, num_examples, metrics
        try:
            from flwr.common import FitRes
            self.fitres = FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(params),
                num_examples=num_examples,
                metrics={},
            )
        except Exception:
            self.fitres = None
        class _CP:
            def __init__(self, cid): self.cid = cid
        self.client = _CP(cid)

    def as_tuple(self):
        return (self.client, self.fitres)


def test_strategy_st_freezes_at_round_2():
    """ST: scores fired only at round 2; rounds 3+ use frozen weights."""
    if not _STRATEGY_AVAILABLE:
        print("SKIP test_strategy_st_freezes_at_round_2 (flwr unavailable)")
        return
    calls = []
    def score_provider(server_round, payload):
        calls.append(server_round)
        # Pretend client-0 has score 0.0 and client-1 has score 1.0
        return [0.0, 1.0]

    strat = WeightedFedAvg(
        weight_mode="st",
        score_provider=score_provider,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_available_clients=2,
    )
    # 2 clients, simple 1-layer params
    p0 = [np.zeros(3)]
    p1 = [np.ones(3)]
    results = [_MockResult("0", p0, 100).as_tuple(),
               _MockResult("1", p1, 100).as_tuple()]
    if any(r[1] is None for r in results):
        print("SKIP test_strategy_st_freezes_at_round_2 (flwr.common.FitRes unavailable)")
        return

    # Round 1: no score call (round ≤ 2 of ST uses plain FedAvg, score only at round 2)
    strat.aggregate_fit(1, results, [])
    assert calls == [], f"round 1 must not call provider, got {calls}"

    # Round 2: should call provider, freeze weights
    strat.aggregate_fit(2, results, [])
    assert calls == [2], f"round 2 should call provider once, got {calls}"
    assert strat._frozen_weights is not None
    # After shift_and_normalize([0,1]): mean=0.5 → [0.0, 2.0]
    _assert_close(strat._frozen_weights, [0.0, 2.0], msg="st_frozen_weights")

    # Round 3: should NOT call provider again; uses frozen weights
    strat.aggregate_fit(3, results, [])
    assert calls == [2], f"round 3 must not call provider, got {calls}"
    print("OK test_strategy_st_freezes_at_round_2")


def test_strategy_dy_fresh_each_round():
    """DY: provider called every round ≥ 2; no frozen state."""
    if not _STRATEGY_AVAILABLE:
        print("SKIP test_strategy_dy_fresh_each_round (flwr unavailable)")
        return
    call_log = []
    def score_provider(server_round, payload):
        call_log.append(server_round)
        return [float(server_round), 1.0]  # vary by round

    strat = WeightedFedAvg(
        weight_mode="dy",
        score_provider=score_provider,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_available_clients=2,
    )
    p0 = [np.zeros(3)]
    p1 = [np.ones(3)]
    results = [_MockResult("0", p0, 100).as_tuple(),
               _MockResult("1", p1, 100).as_tuple()]
    if any(r[1] is None for r in results):
        print("SKIP test_strategy_dy_fresh_each_round (flwr.common.FitRes unavailable)")
        return

    strat.aggregate_fit(1, results, [])
    assert call_log == [], f"round 1 must not call provider, got {call_log}"

    strat.aggregate_fit(2, results, [])
    assert call_log == [2]
    strat.aggregate_fit(3, results, [])
    assert call_log == [2, 3]
    strat.aggregate_fit(5, results, [])
    assert call_log == [2, 3, 5]
    assert strat._frozen_weights is None, "DY must never freeze"
    print("OK test_strategy_dy_fresh_each_round")


def test_strategy_none_never_calls_provider():
    """BL (weight_mode='none'): provider should never be invoked."""
    if not _STRATEGY_AVAILABLE:
        print("SKIP test_strategy_none_never_calls_provider (flwr unavailable)")
        return
    calls = []
    def score_provider(server_round, payload):
        calls.append(server_round)
        return [1.0, 1.0]

    # When mode is 'none', score_provider should be optional; pass anyway to be sure
    # it's not invoked
    strat = WeightedFedAvg(
        weight_mode="none",
        score_provider=score_provider,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_available_clients=2,
    )
    p0 = [np.zeros(3)]
    p1 = [np.ones(3)]
    results = [_MockResult("0", p0, 100).as_tuple(),
               _MockResult("1", p1, 100).as_tuple()]
    if any(r[1] is None for r in results):
        print("SKIP test_strategy_none_never_calls_provider (flwr.common.FitRes unavailable)")
        return

    for r in [1, 2, 3, 5, 10]:
        strat.aggregate_fit(r, results, [])
    assert calls == [], f"BL must not call provider, got {calls}"
    print("OK test_strategy_none_never_calls_provider")


if __name__ == "__main__":
    print("--- pure-functional helpers ---")
    test_shift_and_normalize_positive_scores()
    test_shift_and_normalize_mixed_signs()
    test_shift_and_normalize_all_zero()
    test_shift_and_normalize_all_negative()
    test_cap_bounds_one_hot_scores()
    test_cap_disabled_is_noop()
    test_weighted_aggregate_uniform_weights_equals_fedavg()
    test_weighted_aggregate_doubles_weight_doubles_share()
    test_weighted_aggregate_zero_weights_falls_back_uniform()
    test_size_x_weight_modulation()
    print()
    print("--- strategy state machine ---")
    test_strategy_st_freezes_at_round_2()
    test_strategy_dy_fresh_each_round()
    test_strategy_none_never_calls_provider()
    print()
    print("All tests passed OK")

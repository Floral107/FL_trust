"""Smoke tests for the RobShap demo: loader normalization + live L1O scoring."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robshap_demo.loader import DIMENSIONS, TRADEOFF_DIMS, load_results, tradeoff_matrix
from robshap_demo.scoring import score_clients

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "clients_demo.csv")


# ------------------------------------------------------------------ loader
def test_load_results_tidy_shape():
    df = load_results(DATA_DIR)
    assert not df.empty
    assert set(df.columns) >= {"dataset", "partition", "num_clients", "seed",
                               "round", "client_id", "method", "dimension",
                               "global", "contribution"}
    assert set(df["method"].unique()) <= {"l1o", "gtg"}
    assert set(df["dimension"].unique()) <= set(DIMENSIONS)
    # All eight dataset files should have survived normalization.
    assert df["dataset"].nunique() >= 4


def test_tradeoff_matrix_one_row_per_client():
    df = load_results(DATA_DIR)
    sub = df[(df["dataset"] == "adult") & (df["method"] == "l1o")
             & (df["num_clients"] == 4) & (df["round"] == 10)]
    wide = tradeoff_matrix(sub)
    assert len(wide) == 4
    assert [c for c in wide.columns if c != "client_id"] == TRADEOFF_DIMS


# ------------------------------------------------------------------ live scoring
@pytest.fixture(scope="module")
def sample_df():
    if not os.path.exists(SAMPLE):
        pytest.skip("sample data not generated")
    return pd.read_csv(SAMPLE)


def test_score_clients_shapes(sample_df):
    res = score_clients(sample_df, "client", "income", "sex")
    assert len(res.contributions) == 5
    for m in ("accuracy", "neg_log_loss", "noise_robustness", "fairness_dp", "fairness_eo"):
        assert m in res.contributions.columns, m
    assert res.n_test > 0
    assert 0.0 <= res.global_metrics["accuracy"] <= 1.0


def test_scoring_flags_biased_client(sample_df):
    """client_4's group-biased labels should make it the clear fairness villain.

    This is the demo's headline result and it is robust by a wide margin: removing the
    biased client measurably improves demographic parity, so its DP contribution is the
    most negative one, separated from every other client.
    """
    c = score_clients(sample_df, "client", "income", "sex").contributions
    dp = c["fairness_dp"]
    assert dp.idxmin() == "client_4"
    assert dp["client_4"] < 0
    # Separated from the pack, not a coin-flip winner.
    assert dp["client_4"] < dp.drop("client_4").min() - 0.03


def test_scoring_flags_noisy_client(sample_df):
    """client_3's random labels barely move accuracy but dominate the loss swing.

    Illustrates why accuracy alone is insufficient: the noisy client is the largest
    contributor (by magnitude) to the model's loss/calibration.
    """
    c = score_clients(sample_df, "client", "income", "sex").contributions
    loss_swing = c["neg_log_loss"].abs()
    assert loss_swing.idxmax() == "client_3"


def test_score_clients_deterministic(sample_df):
    a = score_clients(sample_df, "client", "income", "sex").contributions
    b = score_clients(sample_df, "client", "income", "sex").contributions
    pd.testing.assert_frame_equal(a, b)


def test_score_clients_validates_input():
    df = pd.DataFrame({"client": ["a"] * 10, "x": range(10), "y": [0, 1] * 5})
    with pytest.raises(ValueError, match="at least 2 clients"):
        score_clients(df, "client", "y")

    df2 = pd.DataFrame({"client": ["a", "b"] * 5, "x": range(10), "y": [1] * 10})
    with pytest.raises(ValueError, match="at least 2 classes"):
        score_clients(df2, "client", "y")

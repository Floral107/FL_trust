"""
reweight_eval.py
================
Post-hoc BL / ST / DY reweighting evaluation for Tab. 4.

For each (dataset, noniid, num_clients, seed, target_dim):
  - BL  : standard equal-weight FedAvg (loads existing global_model_round_10)
  - ST  : fixed round-2 weights applied to round-10 client models
  - DY  : cumulative-score weights (rounds 2-9) applied to round-10 client models

Results are averaged over 5 seeds and written to:
  results_reweight_<dataset>_<nc>.csv

Then run  python reweight_eval.py --latex  to print the LaTeX table rows.

Usage
-----
# Run experiments (one dataset at a time to keep memory manageable):
python reweight_eval.py --dataset adultnoniid    --num_clients 4
python reweight_eval.py --dataset adultnoniid    --num_clients 20
python reweight_eval.py --dataset celebanoniid   --num_clients 4
python reweight_eval.py --dataset celebanoniid   --num_clients 20
python reweight_eval.py --dataset imdbnoniid     --num_clients 4
python reweight_eval.py --dataset imdbnoniid     --num_clients 20

# Generate LaTeX table:
python reweight_eval.py --latex
"""

import os, sys, argparse, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("seed", "42")  # task.py modules read this at import time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(BASE_DIR, "420")
CSV_ROOT   = os.path.join(BASE_DIR, "data", "combo")

# ET checkpoint roots written by run_et_experiments.{sh,ps1}.
ET_MODEL_ROOTS = {
    "fair": os.path.join(BASE_DIR, "420_et_fair"),
    "adv":  os.path.join(BASE_DIR, "420_et_adv"),
    "dp":   os.path.join(BASE_DIR, "420_et_dp"),
}
# Which Tab. 4 target_group each ET mode fills in.
ET_MODE_TO_TARGET_GROUP = {
    "fair": "fairness",
    "adv":  "robustness",
    "dp":   "privacy",
}

SEEDS = [42, 107, 123, 2025, 9928]
ROUNDS = list(range(1, 11))   # 1-10
EVAL_ROUND = 10

# Score columns in the CSV keyed by our metric names
# CSV dataset label mapping. We now have a real celebanoniid combo CSV
# (from the CelebA re-run), so point at it; legacy cifar combo is kept
# around only as a fallback for older analyses.
CSV_DATASET_MAP = {
    "adultnoniid":  "results_adultnoniid_gtg.csv",
    "celebanoniid": "results_celebanoniid_gtg.csv",
    "imdbnoniid":   "results_imdbnoniid_gtg.csv",
}

SCORE_COL = {
    "acc":    "acc_contribution",
    "loss":   "loss_contribution",
    "fairDP": "gtg_fair_contribution",
    "fairEO": "gtg_fair_eo_contribution",
    "rel":    "gtg_rob_contribution",
    "res":    "gtg_adv_contribution",
    "priv":   "gtg_priv_contribution",
}

# Which target dimension is used for each "column group" in Tab. 4
TARGET_DIMS = {
    "fairness":   "fairDP",
    "robustness": "res",
    "privacy":    "priv",
}

ALL_METRICS = ["loss", "acc", "fairDP", "fairEO", "rel", "res", "priv"]

# ── helpers ────────────────────────────────────────────────────────────────────

def compute_weights(scores: np.ndarray) -> np.ndarray:
    """Shift to non-negative, normalise by mean (paper eq.)."""
    scores = np.asarray(scores, dtype=float)
    min_s = min(0.0, float(np.min(scores)))
    shifted = scores - min_s          # CS'  = CS + |min(0, min_k CS)|
    mean_s = float(np.mean(shifted))
    if mean_s < 1e-12:
        return np.ones(len(scores)) / len(scores)
    return shifted / mean_s           # CS'' = CS' / mean(CS')


def weighted_avg_weights(param_list, weights):
    """Weighted average of a list of Keras weight arrays."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    result = []
    for layer_idx in range(len(param_list[0])):
        avg = sum(wk * param_list[k][layer_idx] for k, wk in enumerate(w))
        result.append(avg)
    return result


def load_client_weights(model_dir, num_clients, round_num, dataset):
    """Load Keras model weights for all clients at a given round."""
    loss_fn = ("binary_crossentropy" if "adult" in dataset.lower()
               else "sparse_categorical_crossentropy")
    weights_list = []
    for cid in range(num_clients):
        path = os.path.join(model_dir, f"client_{cid}_round_{round_num}.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        m = tf.keras.models.load_model(path, compile=False)
        m.compile(loss=loss_fn)
        weights_list.append(m.get_weights())
    return weights_list


def load_global_model(model_dir, round_num, dataset):
    loss_fn = ("binary_crossentropy" if "adult" in dataset.lower()
               else "sparse_categorical_crossentropy")
    path = os.path.join(model_dir, f"global_model_round_{round_num}.keras")
    m = tf.keras.models.load_model(path, compile=False)
    m.compile(loss=loss_fn)
    return m


def set_model_weights(template_model, weights):
    """Return a fresh copy of template with given weights (avoids mutation)."""
    m = tf.keras.models.clone_model(template_model)
    m.set_weights(weights)
    return m


# ── data loading ───────────────────────────────────────────────────────────────

def load_test_data(dataset: str):
    """Returns (x_test, y_test, s_test_or_None)."""
    d = dataset.lower()
    if "adult" in d:
        from adultnoniid.adultnoniid.task import load_data as ld
        _, _, x_test, y_test, _, s_test = ld(0, 1, return_sensitive=True)
        return x_test, y_test, s_test

    elif "celeba" in d:
        from celebanoniid.celebanoniid.task import load_data as ld
        _, _, x_test, y_test, _, s_test = ld(0, 1, return_sensitive=True, test_only=False)
        return x_test, y_test, s_test

    elif "imdb" in d:
        from imdbnoniid.imdbnoniid.task import load_data as ld
        _, _, x_test, y_test = ld(0, 1)
        return x_test, y_test, None

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── metric evaluation ──────────────────────────────────────────────────────────

def _majority_pred_fraction(model, x_test):
    """Fraction of inputs assigned the single most-common predicted class.

    ~1.0 => the model is a (near-)constant predictor that ignores its input.
    Healthy classifiers spread predictions across classes, so this stays well
    below 1 (e.g. ~0.5 on balanced IMDB, ~0.76 on class-skewed ADULT).
    """
    import numpy as np
    p = model(x_test, training=False)
    p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
    if p.ndim == 2 and p.shape[-1] > 1:
        yhat = p.argmax(axis=-1)
    else:
        yhat = (np.asarray(p).reshape(-1) >= 0.5).astype(int)
    if len(yhat) == 0:
        return 1.0
    _, cnt = np.unique(yhat, return_counts=True)
    return float(cnt.max()) / float(len(yhat))


# A model is "res-collapsed" when it predicts one class for >= this fraction of
# inputs: resilience (1 - PGD success) is then a meaningless ~1.0 artifact
# (PGD cannot flip a constant output), so we invalidate res for it.
RES_COLLAPSE_FRACTION = 0.98


def res_is_collapsed(model, x_test):
    return _majority_pred_fraction(model, x_test) >= RES_COLLAPSE_FRACTION


def eval_all_metrics(model, x_test, y_test, s_test, dataset):
    """Evaluate all 7 metrics on a model. Returns dict metric→float."""
    from cont_evals import _accuracy_score, _loss_score
    from robustness_metric import calculate_robustness_score
    from attack_metric import calculate_pgd_score
    from fairness_metric import calculate_fairness_score

    d = dataset.lower()

    scores = {}

    # acc + loss
    scores["acc"]  = _accuracy_score(model, x_test, y_test, d)
    scores["loss"] = _loss_score(model, x_test, y_test, d)

    # reliability
    scores["rel"] = float(calculate_robustness_score(model, x_test, d))

    # resilience (PGD)
    scores["res"] = float(calculate_pgd_score(model, x_test, d, y_test))
    # RES-COLLAPSE GUARD: a near-constant predictor yields a spurious res~1.0
    # (PGD can't flip a constant output; ~no clean-correct samples move). That's
    # a metric artifact, not robustness, so we null res (NaN) for such models —
    # it's then dropped from the Tab.5 res-row mean rather than inflating it.
    if "imdb" in d:  # imdb x_test is raw text; embed for the model forward (cached)
        from imdbnoniid.imdbnoniid.task import process_text as _pt
        _x_model = _pt(x_test)
    else:
        _x_model = x_test
    _majfrac = _majority_pred_fraction(model, _x_model)
    if _majfrac >= RES_COLLAPSE_FRACTION:
        print(f"    [res-guard] constant predictor (majority-class frac={_majfrac:.3f}, "
              f"acc={scores['acc']:.3f}): res={scores['res']:.3f} -> NaN", flush=True)
        scores["res"] = float("nan")

    # fairness (only for datasets with sensitive attr)
    if s_test is not None:
        scores["fairDP"] = float(
            calculate_fairness_score(model, x_test, s_test, d,
                                     y_test=y_test, fairness_metric="dp"))
        scores["fairEO"] = float(
            calculate_fairness_score(model, x_test, s_test, d,
                                     y_test=y_test, fairness_metric="eo"))
    else:
        scores["fairDP"] = float("nan")
        scores["fairEO"] = float("nan")

    # privacy (MIA) — shadow fine-tune protocol (same as cont_evals)
    from privacy_metric import MIAConfig, make_mia_splits
    from cont_evals import _build_privacy_value
    if "imdb" in d:
        from imdb.imdb.task import process_text  # lazy: not all hosts have imdb/
        x_priv = process_text(x_test)
    else:
        x_priv = x_test
    # CelebA is image data: pass clip_values like "cifar" path in _build_privacy_value
    is_image = ("cifar" in d or "celeba" in d)
    mia_cfg = MIAConfig(clip_values=(0.0, 1.0) if is_image else None)
    mia_splits = make_mia_splits(y_test, seed=42, round_num=EVAL_ROUND,
                                  member_ratio=0.5, attack_train_ratio=0.5, balance=True)
    priv_fn = _build_privacy_value(
        "gtg_priv", x_priv, y_test, d, EVAL_ROUND,
        mia_seed=42, mia_splits=mia_splits, mia_config=mia_cfg,
        reference_model=model,
    )
    scores["priv"] = float(priv_fn(model))

    return scores


# ── main experiment loop ────────────────────────────────────────────────────────

def run_experiment(dataset: str, num_clients: int):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}  |  num_clients: {num_clients}")
    print('='*60)

    # Load CSV scores
    csv_path = os.path.join(CSV_ROOT, CSV_DATASET_MAP[dataset])
    if not os.path.exists(csv_path):
        # fallback: look in current dir
        csv_path = os.path.join(BASE_DIR, CSV_DATASET_MAP[dataset])
    df_scores = pd.read_csv(csv_path)
    df_scores = df_scores[df_scores["num_clients"] == num_clients].copy()

    # Load test data once (same for all seeds)
    print("Loading test data...")
    x_test, y_test, s_test = load_test_data(dataset)

    # Template model for cloning
    model_dir_template = os.path.join(MODEL_ROOT, dataset, str(num_clients), str(SEEDS[0]))
    template_model = load_global_model(model_dir_template, EVAL_ROUND, dataset)

    results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}")
        model_dir = os.path.join(MODEL_ROOT, dataset, str(num_clients), str(seed))

        # ── BL: existing global model ─────────────────────────────────────────
        print("    BL...", end=" ", flush=True)
        bl_model = load_global_model(model_dir, EVAL_ROUND, dataset)
        bl_scores = eval_all_metrics(bl_model, x_test, y_test, s_test, dataset)
        print("done")
        for tgt_group in TARGET_DIMS:
            for metric, val in bl_scores.items():
                results.append(dict(dataset=dataset, num_clients=num_clients,
                                    seed=seed, target_group=tgt_group,
                                    condition="BL", eval_metric=metric, value=val))

        # Load client weights at round 10 (used for ST and DY)
        client_w10 = load_client_weights(model_dir, num_clients, EVAL_ROUND, dataset)

        # ── Per target group: ST and DY ───────────────────────────────────────
        for tgt_group, tgt_dim in TARGET_DIMS.items():
            score_col = SCORE_COL[tgt_dim]
            df_s = df_scores[df_scores["seed"] == seed]

            if score_col not in df_s.columns:
                print(f"    [{tgt_group}] score column '{score_col}' missing, skipping")
                continue

            # ── ST weights: from round 2 ──────────────────────────────────────
            r2 = df_s[df_s["round"] == 2].sort_values("client_id")
            if len(r2) < num_clients:
                print(f"    [{tgt_group}] round-2 data incomplete, skipping ST/DY")
                continue

            st_raw = r2[score_col].values[:num_clients]
            # Replace NaN with 0
            st_raw = np.where(np.isnan(st_raw), 0.0, st_raw)
            st_weights = compute_weights(st_raw)

            print(f"    ST [{tgt_group}]...", end=" ", flush=True)
            st_params = weighted_avg_weights(client_w10, st_weights)
            st_model  = set_model_weights(template_model, st_params)
            st_scores = eval_all_metrics(st_model, x_test, y_test, s_test, dataset)
            print("done")
            for metric, val in st_scores.items():
                results.append(dict(dataset=dataset, num_clients=num_clients,
                                    seed=seed, target_group=tgt_group,
                                    condition="ST", eval_metric=metric, value=val))

            # ── DY weights: cumulative sum over rounds 2-9 ────────────────────
            rounds_used = [r for r in ROUNDS if 2 <= r <= EVAL_ROUND - 1]
            cum_scores = np.zeros(num_clients)
            for rnd in rounds_used:
                rdf = df_s[df_s["round"] == rnd].sort_values("client_id")
                if len(rdf) < num_clients:
                    continue
                vals = rdf[score_col].values[:num_clients]
                vals = np.where(np.isnan(vals), 0.0, vals)
                cum_scores += vals

            dy_weights = compute_weights(cum_scores)

            print(f"    DY [{tgt_group}]...", end=" ", flush=True)
            dy_params = weighted_avg_weights(client_w10, dy_weights)
            dy_model  = set_model_weights(template_model, dy_params)
            dy_scores = eval_all_metrics(dy_model, x_test, y_test, s_test, dataset)
            print("done")
            for metric, val in dy_scores.items():
                results.append(dict(dataset=dataset, num_clients=num_clients,
                                    seed=seed, target_group=tgt_group,
                                    condition="DY", eval_metric=metric, value=val))

        tf.keras.backend.clear_session()

    # Save raw results
    out_df = pd.DataFrame(results)
    out_path = os.path.join(BASE_DIR, f"results_reweight_{dataset}_{num_clients}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return out_df


# ── ET evaluation ──────────────────────────────────────────────────────────────

def run_et_evaluation(dataset: str, num_clients: int, et_mode: str):
    """Evaluate the ET-trained round-10 global model for each seed and save CSV.

    Reads 420_et_<et_mode>/<dataset>/<num_clients>/<seed>/global_model_round_10.keras.
    Writes results_et_<et_mode>_<dataset>_<num_clients>.csv with one row per
    (seed, eval_metric).
    """
    if et_mode not in ET_MODEL_ROOTS:
        raise ValueError(f"et_mode must be one of {list(ET_MODEL_ROOTS)}, got {et_mode!r}")

    root = ET_MODEL_ROOTS[et_mode]
    print(f"\n{'='*60}")
    print(f"ET eval | dataset={dataset}  nc={num_clients}  et_mode={et_mode}")
    print(f"Model root: {root}")
    print('='*60)

    # Skip imdbnoniid + fair (no sensitive attribute → ET-fair is a BL no-op).
    if dataset == "imdbnoniid" and et_mode == "fair":
        print("[SKIP] imdbnoniid + fair (no sensitive attribute)")
        return None

    print("Loading test data...")
    x_test, y_test, s_test = load_test_data(dataset)

    results = []
    for seed in SEEDS:
        model_dir = os.path.join(root, dataset, str(num_clients), str(seed))
        path = os.path.join(model_dir, f"global_model_round_{EVAL_ROUND}.keras")
        if not os.path.exists(path):
            print(f"  [skip seed={seed}] missing {path}")
            continue

        print(f"  Seed {seed} ...", end=" ", flush=True)
        loss_fn = ("binary_crossentropy" if "adult" in dataset.lower()
                   else "sparse_categorical_crossentropy")
        et_model = tf.keras.models.load_model(path, compile=False)
        et_model.compile(loss=loss_fn)
        scores = eval_all_metrics(et_model, x_test, y_test, s_test, dataset)
        print("done")

        for metric, val in scores.items():
            results.append(dict(
                dataset=dataset, num_clients=num_clients, seed=seed,
                et_mode=et_mode, eval_metric=metric, value=val,
            ))
        tf.keras.backend.clear_session()

    if not results:
        print("No seeds had ET checkpoints; skipping save.")
        return None

    out_df = pd.DataFrame(results)
    out_path = os.path.join(BASE_DIR, f"results_et_{et_mode}_{dataset}_{num_clients}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return out_df


# ── LaTeX table generation ─────────────────────────────────────────────────────

def load_all_reweight_results():
    dfs = []
    for ds in ["adultnoniid", "celebanoniid", "imdbnoniid"]:
        for nc in [4, 20]:
            p = os.path.join(BASE_DIR, f"results_reweight_{ds}_{nc}.csv")
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
    if not dfs:
        print("No result files found. Run experiments first.")
        return None
    return pd.concat(dfs, ignore_index=True)


def load_all_et_results():
    """Returns one combined DataFrame across all results_et_*_*_*.csv files."""
    dfs = []
    for em in ET_MODEL_ROOTS:
        for ds in ["adultnoniid", "celebanoniid", "imdbnoniid"]:
            for nc in [4, 20]:
                p = os.path.join(BASE_DIR, f"results_et_{em}_{ds}_{nc}.csv")
                if os.path.exists(p):
                    dfs.append(pd.read_csv(p))
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def fmt(v):
    """Format a float as .XX for the table."""
    if pd.isna(v):
        return r"\textemdash"
    return f"{v:.2f}"


def generate_latex_table(df, et_df=None):
    """Print the LaTeX rows for Tab. 4.

    `df`     — combined results_reweight_*.csv (BL/ST/DY rows)
    `et_df`  — combined results_et_*.csv (ET rows). If None, ET rows fall back to $.XX$.
    """
    # Average over seeds
    agg = (df.groupby(["dataset", "num_clients", "target_group", "condition", "eval_metric"])
             ["value"].mean().reset_index())
    et_agg = None
    if et_df is not None:
        et_agg = (et_df.groupby(["dataset", "num_clients", "et_mode", "eval_metric"])
                       ["value"].mean().reset_index())

    # Dataset × num_clients columns in table order
    # Tab. 4 columns: ADULT-4, ADULT-20, CelebA-4, CelebA-20, IMDB-4, IMDB-20
    COL_ORDER = [
        ("adultnoniid",  4),
        ("adultnoniid",  20),
        ("celebanoniid", 4),
        ("celebanoniid", 20),
        ("imdbnoniid",   4),
        ("imdbnoniid",   20),
    ]

    TARGET_GROUPS = ["fairness", "robustness", "privacy"]
    CONDITIONS    = ["BL", "ST", "DY"]  # ET left blank
    EVAL_METRICS  = ["loss", "acc", "fairDP", "fairEO", "rel", "res", "priv"]

    METRIC_LABEL = {
        "loss":   r"$\mathtt{loss}$",
        "acc":    r"$\mathtt{acc}$",
        "fairDP": r"$\mathtt{fairDP}$",
        "fairEO": r"$\mathtt{fairEO}$",
        "rel":    r"$\mathtt{rel}$",
        "res":    r"$\mathtt{res}$",
        "priv":   r"$\mathtt{priv}$",
    }

    def get_val(ds, nc, tg, cond, metric):
        sub = agg[(agg.dataset == ds) & (agg.num_clients == nc) &
                  (agg.target_group == tg) & (agg.condition == cond) &
                  (agg.eval_metric == metric)]
        if len(sub) == 0:
            return float("nan")
        return float(sub["value"].iloc[0])

    # Target_group -> et_mode mapping for ET rows.
    tg_to_et = {v: k for k, v in ET_MODE_TO_TARGET_GROUP.items()}

    def get_et_val(ds, nc, tg, metric):
        if et_agg is None:
            return float("nan")
        em = tg_to_et.get(tg)
        if em is None:
            return float("nan")
        sub = et_agg[(et_agg.dataset == ds) & (et_agg.num_clients == nc) &
                     (et_agg.et_mode == em) & (et_agg.eval_metric == metric)]
        if len(sub) == 0:
            return float("nan")
        return float(sub["value"].iloc[0])

    if et_agg is None:
        print("% ---- Tab. 4 BL/ST/DY rows (ET rows remain $.XX$) ----")
    else:
        print("% ---- Tab. 4 BL/ST/DY/ET rows (ET filled from results_et_*.csv) ----")
    print(r"\hline\hline")

    for eval_metric in EVAL_METRICS:
        print(f"        \\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{{METRIC_LABEL[eval_metric]}}}}}", end="")

        for i, cond in enumerate(CONDITIONS + ["ET"]):
            if i == 0:
                print(f" & {cond}", end="")
            else:
                print(f"\n        & {cond}", end="")

            for tg_idx, tg in enumerate(TARGET_GROUPS):
                for ds, nc in COL_ORDER:
                    if cond == "ET":
                        v = get_et_val(ds, nc, tg, eval_metric)
                        if pd.isna(v):
                            print(r" & $.XX$", end="")
                        else:
                            print(f" & {fmt(v)}", end="")
                    else:
                        v = get_val(ds, nc, tg, cond, eval_metric)
                        print(f" & {fmt(v)}", end="")

            print(r" \\")
        print(r"        \hline")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=str, default=None,
                        choices=list(CSV_DATASET_MAP.keys()))
    parser.add_argument("--num_clients", type=int, default=None, choices=[4, 20])
    parser.add_argument("--latex",       action="store_true",
                        help="Load saved results and print LaTeX rows")
    parser.add_argument("--et-mode",     type=str, default=None,
                        choices=list(ET_MODEL_ROOTS.keys()),
                        help="Evaluate ET-trained checkpoints in 420_et_<mode>/ "
                             "(no reweighting) and write results_et_<mode>_<ds>_<nc>.csv")
    args = parser.parse_args()

    if args.latex:
        df = load_all_reweight_results()
        et_df = load_all_et_results()
        if df is not None:
            generate_latex_table(df, et_df=et_df)
    elif args.et_mode:
        if args.dataset and args.num_clients:
            run_et_evaluation(args.dataset, args.num_clients, args.et_mode)
        else:
            # Loop over all combos for this ET mode.
            for ds in CSV_DATASET_MAP:
                for nc in [4, 20]:
                    run_et_evaluation(ds, nc, args.et_mode)
    elif args.dataset and args.num_clients:
        run_experiment(args.dataset, args.num_clients)
    else:
        # Run all BL/ST/DY combinations
        for ds in CSV_DATASET_MAP:
            for nc in [4, 20]:
                run_experiment(ds, nc)

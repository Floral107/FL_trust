import argparse
import os
import sys

# CPU force: hardcoded historically because TF's PTX-JIT path is broken on
# RTX 5090 (compute capability 12.0) and CW attack corrupts the CUDA context
# on that same card. On a vast.ai 3090/4090/A100 the GPU path works fine, so
# gate the disable behind FLR_FORCE_CPU=1. eval_tmux.sh still sets this for
# priv and cw methods (MIA fine-tune + CW context issues persist regardless
# of card), so those stay on CPU even on the GPU instance.
if os.environ.get("FLR_FORCE_CPU", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Mixed precision for GPU efficiency. RTX 4090 (and Ampere generally) has
# 4-8× higher bf16 matmul throughput than fp32. Inference-only — no gradient
# accumulation issues. set_global_policy must run BEFORE any keras op so we
# do it here at module load. Skipped when FLR_FORCE_CPU is set (CPU doesn't
# benefit, MIA fine-tune in privacy_metric.py has dtype assumptions).
if os.environ.get("FLR_MIXED_PRECISION", "0") == "1" and os.environ.get("FLR_FORCE_CPU", "0") != "1":
    import tensorflow as _tf
    try:
        _tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print(f"[mixed precision] policy = mixed_bfloat16")
    except Exception as e:
        print(f"[mixed precision] failed to set policy: {e}")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

# Optional app dirs: a compute box only deploys the datasets it runs (e.g. this
# non-IID box has no adult/celeba/cifar iid dirs). Import lazily so a missing app
# dir yields a skipped dataloader key instead of crashing every scoring job at
# import time (the 2026-07-09 silent-zero-scoring bug).
def _opt(modpath, attr):
    import importlib
    try:
        return getattr(importlib.import_module(modpath), attr)
    except ModuleNotFoundError:
        return None

load_adult        = _opt("adult.adult.task", "load_data")
load_adultnoniid  = _opt("adultnoniid.adultnoniid.task", "load_data")
load_cifar        = _opt("cifar.cifar.task", "load_data")
load_cifarnoniid  = _opt("cifarnoniid.cifarnoniid.task", "load_data")
load_imdb         = _opt("imdb.imdb.task", "load_data")
load_imdbnoniid   = _opt("imdbnoniid.imdbnoniid.task", "load_data")
load_celeba       = _opt("celeba.celeba.task", "load_data")
load_celebanoniid = _opt("celebanoniid.celebanoniid.task", "load_data")

# set_seed is byte-identical across every task.py; take it from whichever app
# dir this box happens to have deployed.
set_seed = next(fn for fn in (
    _opt("adultnoniid.adultnoniid.task", "set_seed"),
    _opt("imdbnoniid.imdbnoniid.task", "set_seed"),
    _opt("adult.adult.task", "set_seed"),
    _opt("celebanoniid.celebanoniid.task", "set_seed"),
) if fn is not None)

from cont_evals import evaluate_round
from privacy_metric import make_mia_splits, MIAConfig


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TF: memory growth enabled")
    except RuntimeError as e:
        print("TF: cannot set memory growth after initialization:", e)

dataloaders = {name: fn for name, fn in {
    "cifar": load_cifar,
    "imdb": load_imdb,
    "adult": load_adult,
    "imdbnoniid": load_imdbnoniid,
    "adultnoniid": load_adultnoniid,
    "cifarnoniid": load_cifarnoniid,
    "celeba": load_celeba,
    "celebanoniid": load_celebanoniid,
}.items() if fn is not None}

BASE_DIR = "./420"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate federated learning metrics (LOO/GTG).")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar", "imdb", "adult", "cifarnoniid", "imdbnoniid", "adultnoniid",
                                 "celeba", "celebanoniid"])
    parser.add_argument("--partition", type=str, required=True, choices=["iid", "noniid"])
    parser.add_argument("--num_rounds", type=int, required=True)

    # (javított) seed: 1 seed vagy 'all'
    parser.add_argument("--seed", type=str, required=True,
                        choices=["all", "42", "107", "123", "2025", "9928"])

    parser.add_argument("--method", type=str, required=True,
                        choices=[
                            "l1o_rob", "l1o_adv", "l1o_adv_pgd", "l1o_adv_cw",
                            "l1o_fair", "l1o_fair_eo", "l1o_priv",
                            "l1o_acc", "l1o_loss",
                            "gtg_rob", "gtg_adv", "gtg_adv_pgd", "gtg_adv_cw",
                            "gtg_fair", "gtg_fair_eo", "gtg_priv",
                            "gtg_acc", "gtg_loss",
                        ])

    parser.add_argument("--root_dir", type=str, default=None,
                        help="Root directory containing experiment results. "
                             "Defaults to ./420 for fedavg, ./420_fedprox for fedprox.")
    parser.add_argument("--strategy", type=str, default="fedavg",
                        choices=["fedavg", "fedprox"],
                        help="Training strategy whose saved models to evaluate.")
    # --- sampling controls -------------------------------------------------
    # At full settings (5000 test × 100 noise samples per coalition × 100
    # GTG-Shapley permutations × 20 subsets/perm) the GTG variants of rob /
    # adv / priv / cw are ~weeks/seed even on RTX 4090. These flags trade
    # some statistical precision for tractable runtime. Defaults of None
    # mean "use the metric module's full settings" (5000 / 100 / etc.).
    parser.add_argument("--num_clients", type=int, default=None,
                        choices=[None, 4, 20],
                        help="Restrict eval to one client count (4 or 20). None = both [20, 4].")
    parser.add_argument("--eval_subset", type=int, default=None,
                        help="Subsample the test set to this size. None=full. "
                             "Recommended 1000 for GTG-heavy methods (±1.5%% CI).")
    parser.add_argument("--noise_samples", type=int, default=None,
                        help="L for certified rob (number of noise samples per "
                             "test point). None=metric default (100). 30 gives "
                             "~3× speedup with modest CI widening.")
    return parser.parse_args()


def load_test_data(dataset_key: str):
    k = dataset_key.lower()
    if "adult" in k:
        _, _, x_test, y_test, _, s_test = dataloaders[dataset_key](
            0, 1, return_sensitive=True, sensitive_attr="sex"
        )
    elif "celeba" in k:
        _, _, x_test, y_test, _, s_test = dataloaders[dataset_key](
            0, 1, return_sensitive=True, sensitive_attr="Male", test_only=True,
        )
    else:
        _, _, x_test, y_test = dataloaders[dataset_key](0, 1)
        s_test = None
    return x_test, y_test, s_test


def privacy_config_for_dataset(dataset_key: str) -> MIAConfig:
    """Single source of truth for MIA attack configuration."""
    return MIAConfig(
        finetune_epochs=5,
        finetune_lr=5e-4,
        batch_size=16,
        feature_type="combined_delta",
        attack_model_type="gb",
        clip_values=(0.0, 1.0) if "cifar" in dataset_key.lower() else None,
        use_logits=False,
        use_auc_for_privacy=True,
    )


def _load_done_keys(out_path: str) -> set[tuple[int, int, int]]:
    """Return the set of (seed, num_clients, round) tuples already written to CSV.

    Used to make `robustness.py` resume-safe: on restart, rows already present
    in the output CSV are skipped, so a multi-day run that gets interrupted by
    tmux kill / OOM / disk-full / SSH drop only loses whatever round was
    in-flight, not the whole sweep.

    Returns empty set if the CSV doesn't exist or can't be parsed.
    """
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        return set()
    try:
        df = pd.read_csv(out_path, usecols=["seed", "num_clients", "round"])
        return set(zip(df["seed"].astype(int), df["num_clients"].astype(int), df["round"].astype(int)))
    except Exception as e:
        # Corrupt header / partial write — fall back to no-resume rather than
        # silently re-running everything. Safer than crashing.
        print(f"[resume] could not parse {out_path} ({e}); starting fresh.")
        return set()


def evaluate_all(root_dir: str, dataset: str, partition: str, num_rounds: int, seeds: list[int], method: str, out_path: str, eval_subset: int = None, num_clients_only: int = None):
    x_test, y_test, s_test = load_test_data(dataset)

    # Optional test-set subsampling for tractable GTG runtime. Deterministic
    # via fixed seed=42 so all (seed, method) runs see the same subset
    # (otherwise cross-method comparisons drift). Stratified by label.
    if eval_subset is not None and eval_subset > 0 and eval_subset < len(x_test):
        try:
            from sklearn.model_selection import train_test_split
            y_arr = y_test.numpy() if hasattr(y_test, "numpy") else np.asarray(y_test)
            idx, _ = train_test_split(
                np.arange(len(x_test)), train_size=eval_subset,
                random_state=42, stratify=y_arr,
            )
            idx = np.sort(idx)
            x_test = tf.gather(x_test, idx) if hasattr(x_test, "numpy") else x_test[idx]
            y_test = tf.gather(y_test, idx) if hasattr(y_test, "numpy") else y_test[idx]
            if s_test is not None:
                s_test = s_test[idx]
            print(f"[eval_subset] subsampled test set from full -> {eval_subset} (stratified, seed=42)")
        except Exception as e:
            print(f"[eval_subset] stratified subsample failed ({e}); falling back to first {eval_subset}")
            x_test = x_test[:eval_subset]; y_test = y_test[:eval_subset]
            if s_test is not None:
                s_test = s_test[:eval_subset]

    header_written = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    done_keys = _load_done_keys(out_path)
    if done_keys:
        print(f"[resume] skipping {len(done_keys)} already-computed (seed, num_clients, round) rows from {out_path}")

    # Custom round iteration order (outside-in by default per user spec).
    # FLR_ROUND_ORDER overrides; format is comma-separated, e.g. "1,10,2,9,3,4,5,6,7,8".
    # The intent: cover endpoints early (round 1 and the LAST round are usually
    # what we want first for trustworthiness analysis), then fill in middle rounds.
    _round_order_env = os.environ.get("FLR_ROUND_ORDER", "").strip()
    if _round_order_env:
        try:
            _round_order = [int(x) for x in _round_order_env.split(",") if x.strip()]
            # Clip to valid range and filter dupes preserving order.
            # IMPORTANT: do NOT auto-append missing rounds — the caller might
            # be running just one pair per invocation (e.g. FLR_ROUND_ORDER="1,10"),
            # in which case appending the remaining rounds defeats the per-pair scope.
            seen = set()
            _round_order = [r for r in _round_order if 1 <= r <= num_rounds and not (r in seen or seen.add(r))]
            print(f"[round_order] using custom order: {_round_order}")
        except Exception as e:
            print(f"[round_order] env parse failed ({e}); falling back to 1..N")
            _round_order = list(range(1, num_rounds + 1))
    else:
        _round_order = list(range(1, num_rounds + 1))

    for seed in seeds:
        set_seed(seed)

        nc_list = [num_clients_only] if num_clients_only is not None else [20, 4]
        for num_clients in nc_list:
            exp_dir = os.path.join(root_dir, dataset, str(num_clients), str(seed))
            if not os.path.exists(exp_dir):
                print(f"Directory not found: {exp_dir}")
                continue

            for round_num in tqdm(_round_order, desc=f"{dataset} | {num_clients} clients | seed={seed}"):
                # Resume-from-CSV: skip rounds already in the output. Cheap
                # because done_keys is a hashset and we test before any model load.
                if (seed, num_clients, round_num) in done_keys:
                    continue
                try:
                    mia_splits = None
                    mia_cfg = None
                    if method in ("l1o_priv", "gtg_priv"):
                        mia_splits = make_mia_splits(
                            y_test, seed=seed, round_num=round_num,
                            member_ratio=0.2, attack_train_ratio=0.7, balance=True
                        )
                        mia_cfg = privacy_config_for_dataset(dataset)

                    global_metric, contributions = evaluate_round(
                        exp_dir, num_clients,
                        x_test, y_test, s_test,
                        round_num, dataset,
                        method=method,
                        mia_splits=mia_splits,
                        mia_config=mia_cfg,
                        mia_seed=seed,
                    )

                    if global_metric is None or contributions is None:
                        continue

                    rows = []
                    for client_id, contribution in contributions.items():
                        rows.append({
                            "dataset": dataset.replace("noniid", ""),
                            "partition": partition,
                            "num_clients": num_clients,
                            "seed": seed,
                            "round": round_num,
                            "client_id": client_id,
                            f"global_{method}": float(global_metric),
                            f"{method}_contribution": float(contribution),
                            "abs_contribution": float(abs(contribution)),
                        })

                    # Flush to CSV after each round — crash-safe + frees memory
                    chunk = pd.DataFrame(rows)
                    chunk.to_csv(out_path, mode="a", index=False, header=not header_written)
                    header_written = True

                except Exception as e:
                    print(f"Error evaluating {exp_dir} round {round_num}: {e}")


def main():
    args = parse_arguments()
    seeds = [42, 107, 123, 2025, 9928] if args.seed == "all" else [int(args.seed)]

    root_dir = args.root_dir or (BASE_DIR if args.strategy == "fedavg" else "./420_fedprox")
    # Optional output filename suffix (env-driven). Used when two GPU
    # instances split the same (dataset, seed, method) job across different
    # round orders — each writes its own CSV; merge by (round, num_clients,
    # client_id) at analysis time. Empty by default = standard filename.
    _out_suffix = os.environ.get("FLR_OUT_SUFFIX", "")
    out = f"results_{args.dataset}_{args.seed}_{args.method}{_out_suffix}_{args.strategy}.csv"

    evaluate_all(
        root_dir=root_dir,
        dataset=args.dataset,
        partition=args.partition,
        num_rounds=args.num_rounds,
        seeds=seeds,
        method=args.method,
        out_path=out,
        eval_subset=args.eval_subset,
        num_clients_only=args.num_clients,
    )

    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
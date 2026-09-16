"""gtg_check.py - do the LOO and GTG scores compute, and mean, what they claim?

Runs the production scorer (cont_evals coalition averaging, gtg_evals, l1o_evals and
evaluate_round itself) on one dataset, K and seed, and reports per round and method:

  agg      FedAvg coefficients recovered from the checkpoints vs the clients' real training
           sizes, and whether they rebuild the saved global model (AUDIT A36)
  base     K <= 6: an independent exact Shapley enumeration vs `_shapley` (must match) and
           vs `_contribution` (GTG with truncation and best-subset step, the paper's numbers,
           AUDIT A37); per-group Shapley for gap methods; LOO.
           K > 6: `_shapley` (balanced sampling) vs a long sampling reference (--ref_perms)
  e2e      evaluate_round for the gtg_ and l1o_ method must reproduce the harness
  dup      (K <= 6) client 0 entered twice: exact Shapley gives both copies one value
  planted  (K <= 6) an extra client trained like client_app.fit on only one group of
           client 0's data: its score under the base metric vs under the gap

Deterministic metrics only (acc, loss, fair, res and their gaps).

  seed=42 python Code/gtg_check.py --dataset adultnoniid --K 4 --seed 42 --rounds 2 10
  seed=42 python Code/gtg_check.py --dataset adultnoniid --K 20 --seed 42 --rounds 10 --ref_perms 200
"""
import argparse
import importlib
import itertools
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU eval, see PIPELINE.md gotchas
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

import robustness as R
from cont_evals import (_average, _coalition_coeffs, _gap_base_fn, _get_metric_fun, _values,
                        evaluate_round, gtg_evals, l1o_evals)
from fairness_metric import calculate_fairness_score
from gap_metrics import _take, make_gap_value, parse as parse_gap_method
from gtg_shap import GTGShapley, shapley_values
from robustness_metric import load_client_models

ROOT = Path(__file__).resolve().parent.parent


def metric_fun_for(method, dataset, x, y, s):
    gap = parse_gap_method(method)
    gap_value = make_gap_value(_gap_base_fn(gap[0], dataset), x, y, s, gap[1]) if gap else None
    fair = lambda model, X, fm: calculate_fairness_score(model, X, s, dataset, y_test=y, fairness_metric=fm)
    return _get_metric_fun(method, fair, None, gap_value)


def exact_shapley(u, n):
    """Plain enumeration over all coalitions, independent of gtg_shap."""
    f = math.factorial
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            w = f(r) * f(n - r - 1) / f(n)
            phi[i] += w * sum(u(S + (i,)) - u(S) for S in itertools.combinations(others, r))
    return phi


def rho(a, b):
    return spearmanr(a, b).correlation if np.std(a) > 0 and np.std(b) > 0 else float("nan")


def check(label, method, weights, coeffs, ctx, table=False):
    """Truth (exact or long reference) vs production _shapley, _contribution and LOO.

    coeffs: FedAvg coefficient per player index, or None for a plain mean.
    """
    n = len(weights)
    sign = -1.0 if method.endswith("_loss") else 1.0  # evaluate_round negates raw-loss contributions
    ds, x, y = ctx["dataset"], ctx["x"], ctx["y"]
    mf = metric_fun_for(method, ds, x, y, ctx["s"])
    cache = {}

    def game(subset):  # the production coalition game: previous global for {}, else the coalition average
        key = tuple(sorted(subset))
        if key not in cache:
            model = ctx["prev_model"]
            if key:
                ctx["eval_model"].set_weights(_average([weights[i] for i in key],
                                                       [coeffs[i] for i in key] if coeffs else None))
                model = ctx["eval_model"]
            cache[key] = _values(mf, model, x, ds, y)
        return cache[key]

    u = lambda S: float(game(S)[0])
    everyone = tuple(range(n))
    rng = ctx["seed"] * 10_000 + n * 100 + ctx["round"]  # robustness.py's per-round seed for gap methods
    R.set_seed(rng)
    g = GTGShapley(num_players=n, loss_flag=sign < 0)
    g.set_utility_function(u)
    gtg_harness = sign * np.array(g.compute())
    R.set_seed(rng)
    prod, extras = gtg_evals(mf, ctx["model_dir"], n, x, y, ctx["round"], ds, dict(enumerate(weights)),
                             {i: i for i in everyone}, loss_flag=sign < 0, coeffs=coeffs)
    final = sign * np.array([prod[i] for i in everyone])
    shap = sign * np.array([extras["shapley"][i] for i in everyone])
    loo, _ = l1o_evals(mf, game(everyone), x, y, ds, dict(enumerate(weights)), ctx["global_model"], coeffs=coeffs)
    loo = sign * np.array([loo[i] for i in everyone])
    v_saved = float(_values(mf, ctx["global_model"], x, ds, y)[0])
    if n <= 6:
        truth, truth_name = sign * exact_shapley(u, n), "exact"
    else:
        truth, truth_name = sign * shapley_values(game, n, min_perms=ctx["ref_perms"])[:, 0], f"ref{ctx['ref_perms']}"

    print(f"\n[{label}] round {ctx['round']} | {method} | {n} clients")
    print(f"  u(all clients)={u(everyone):.4f}  u(saved global)={v_saved:.4f}  u(prev global)={u(()):.4f}"
          f"  coalitions scored={len(cache) - 1}/{2 ** n - 1}"
          f"  production _contribution == harness GTG: {np.allclose(final, gtg_harness, atol=1e-9)}")
    if table:
        print(f"  client  {truth_name:>9}   _shapley  _contribution        loo")
        for i in everyone:
            print(f"  {i:>6} {truth[i]:+10.5f} {shap[i]:+10.5f} {final[i]:+13.5f} {loo[i]:+10.5f}")
    print(f"  _shapley vs {truth_name}: max|diff|={np.abs(shap - truth).max():.2e} spearman={rho(shap, truth):.2f}"
          f"  | _contribution vs {truth_name}: max|diff|={np.abs(final - truth).max():.5f}"
          f" spearman={rho(final, truth):.2f} sign agree={int((np.sign(final) == np.sign(truth)).sum())}/{n}"
          f" zeroed={int((final == 0).sum())}/{n}")
    if "shapley_m_a" in extras and n <= 6:
        for col, name in ((1, "shapley_m_a"), (2, "shapley_m_b")):
            exact = exact_shapley(lambda S: float(game(S)[col]), n)
            got = np.array([extras[name][i] for i in everyone])
            print(f"  {name}: max|diff| vs exact {np.abs(got - exact).max():.2e}  values {np.round(got, 5).tolist()}")
    return dict(truth=truth, shap=shap, final=final, loo=loo, game=game, sign=sign, extras=extras)


def end_to_end(method, base, ctx):
    """evaluate_round, the function robustness.py calls, must reproduce the harness."""
    K = ctx["K"]
    for scorer in ("gtg", "l1o"):
        name = scorer + method[3:]
        R.set_seed(ctx["seed"] * 10_000 + K * 100 + ctx["round"])
        _, contribs, info = evaluate_round(ctx["model_dir"], K, ctx["x"], ctx["y"], ctx["s"],
                                           ctx["round"], ctx["dataset"], name)
        got = np.array([contribs[i] for i in range(K)])
        want = base["final"] if scorer == "gtg" else base["loo"]
        line = f"  e2e {name}: coalition_avg={info['coalition_avg']}  max|contribution - harness|={np.abs(got - want).max():.2e}"
        if scorer == "gtg":
            shap = np.array([info["shapley"][i] for i in range(K)])
            line += f"  max|_shapley - harness|={np.abs(shap - base['shap']).max():.2e}"
            for part in ("shapley_m_a", "shapley_m_b"):
                if part in info:
                    line += f"  max|{part} - harness|={np.abs(np.array([info[part][i] for i in range(K)]) - np.array([base['extras'][part][i] for i in range(K)])).max():.2e}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="adultnoniid")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--methods", nargs="+", default=["gtg_acc", "gtg_gap_acc_dp", "gtg_gap_loss_eop"])
    ap.add_argument("--ref_perms", type=int, default=200, help="reference permutations when K > 6")
    ap.add_argument("--root_dir", default=str(ROOT / "420"))
    args = ap.parse_args()

    task = importlib.import_module(f"{args.dataset}.{args.dataset}.task")
    x, y, s = R.load_test_data(args.dataset)
    model_dir = os.path.join(args.root_dir, args.dataset, str(args.K), str(args.seed))
    parts = [task.load_data(pid, args.K, return_sensitive=True) for pid in range(args.K)]
    sizes = [len(p[0]) for p in parts]  # the num_examples each client reports to FedAvg
    shares = lambda extra_rows: [n / (sum(sizes) + extra_rows) for n in sizes + [extra_rows]]

    for rnd in args.rounds:
        load = lambda r: tf.keras.models.load_model(os.path.join(model_dir, f"global_model_round_{r}.keras"), compile=False)
        global_model, prev_model = load(rnd), load(rnd - 1)
        clients = load_client_models(model_dir, args.K, rnd)
        weights = [clients[i].get_weights() for i in range(args.K)]
        ctx = dict(dataset=args.dataset, x=x, y=y, s=s, seed=args.seed, round=rnd, K=args.K, model_dir=model_dir,
                   ref_perms=args.ref_perms, global_model=global_model, prev_model=prev_model,
                   eval_model=tf.keras.models.clone_model(global_model))

        coeffs, mode = _coalition_coeffs(model_dir, rnd, weights, global_model)
        rebuilt = _average(weights, coeffs)
        maxdiff = max(float(np.max(np.abs(p - q))) for p, q in zip(global_model.get_weights(), rebuilt))
        print(f"\n=== round {rnd}: client training sizes {sizes} -> size shares {np.round(np.array(sizes) / sum(sizes), 5).tolist()}")
        print(f"  coalition averaging: {mode}; recovered coefficients "
              f"{None if coeffs is None else np.round(coeffs, 5).tolist()}; rebuilt global max|dw| = {maxdiff:.2e}")

        planted = {}
        if args.K <= 6:
            x0, y0, s0 = parts[0][0], parts[0][1], np.asarray(parts[0][4]).ravel()
            for grp in (1, 0):
                keep = np.flatnonzero(s0 == grp)
                m = task.load_model()
                m.set_weights(prev_model.get_weights())
                task.set_seed(task.SEED)  # client_app.fit: set_seed, then one plain model.fit
                m.fit(_take(x0, keep), _take(y0, keep), epochs=1, batch_size=32, verbose=0)
                planted[grp] = (m.get_weights(), len(keep))

        for method in args.methods:
            base = check("base", method, weights, coeffs, ctx, table=True)
            end_to_end(method, base, ctx)
            if args.K > 6:
                continue
            # Extra players aren't in the checkpoints: every player gets FedAvg's size share.
            dup = check("duplicate of client 0", method, weights + [weights[0]], shares(sizes[0]), ctx)
            print(f"  -> client 0 vs its copy: exact {dup['truth'][0]:+.5f} / {dup['truth'][-1]:+.5f}"
                  f" | _shapley {dup['shap'][0]:+.5f} / {dup['shap'][-1]:+.5f}"
                  f" | _contribution {dup['final'][0]:+.5f} / {dup['final'][-1]:+.5f}")
            for grp, (pw, rows) in planted.items():
                r = check(f"planted: group {'A' if grp else 'B'} only, {rows} rows of client 0", method,
                          weights + [pw], shares(rows), ctx)
                everyone = tuple(range(args.K))
                direct = r["sign"] * (float(r["game"](everyone + (args.K,))[0]) - float(r["game"](everyone)[0]))
                rank = int((r["truth"] > r["truth"][-1]).sum()) + 1
                print(f"  -> planted client: adding it to all {args.K} clients changes the utility by {direct:+.5f};"
                      f" exact {r['truth'][-1]:+.5f} (rank {rank}/{args.K + 1}), _shapley {r['shap'][-1]:+.5f},"
                      f" _contribution {r['final'][-1]:+.5f}")


if __name__ == "__main__":
    main()

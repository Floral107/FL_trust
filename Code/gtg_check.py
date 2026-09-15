"""gtg_check.py - does GTG-Shapley compute, and mean, what the scores claim?

Uses the production pieces (cont_evals metric functions, gtg_evals, l1o_evals,
gtg_shap.GTGShapley, robustness_metric.aggregate_models) on one dataset, K and seed,
and reports for each round and scoring method:

  base     exact Shapley over all 2^K coalitions vs GTG before ("raw") and after
           ("final") its best-coalition step, which zeroes every client outside the
           highest-utility coalition; plus LOO as robustness.py computes it
  agg      the saved global model vs the plain client average that coalitions use,
           and vs the sample-weighted average FedAvg actually computes
  dup      client 0 entered twice: exact Shapley must give both copies one value
  planted  an extra client trained like client_app.fit (one plain epoch from the
           previous global) on only one group of client 0's data; its score under
           the base metric vs under the gap

Deterministic metrics only (acc, loss, fair and their gaps): the comparison needs
GTG and the exact enumeration to see identical utilities.

  seed=42 python Code/gtg_check.py --dataset adultnoniid --K 4 --seed 42 \
      --rounds 2 10 --methods gtg_acc gtg_gap_acc_dp gtg_gap_loss_eop
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
from cont_evals import _gap_base_fn, _get_metric_fun, gtg_evals, l1o_evals
from fairness_metric import calculate_fairness_score
from gap_metrics import _take, make_gap_value, parse as parse_gap_method
from gtg_shap import GTGShapley
from robustness_metric import aggregate_models, load_client_models

ROOT = Path(__file__).resolve().parent.parent


def metric_fun_for(method, dataset, x, y, s):
    gap = parse_gap_method(method)
    gap_value = make_gap_value(_gap_base_fn(gap[0], dataset), x, y, s, gap[1]) if gap else None
    fair = lambda model, X, fm: calculate_fairness_score(model, X, s, dataset, y_test=y, fairness_metric=fm)
    return _get_metric_fun(method, fair, None, gap_value)


def exact_shapley(u, n):
    f = math.factorial
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            w = f(r) * f(n - r - 1) / f(n)
            phi[i] += w * sum(u(S + (i,)) - u(S) for S in itertools.combinations(others, r))
    return phi


def check(label, method, weights, ctx, table=False):
    """Exact Shapley vs GTG (raw, final) vs LOO for one list of client weights."""
    n = len(weights)
    loss = method.endswith("_loss")
    sign = -1.0 if loss else 1.0  # evaluate_round negates raw-loss contributions
    ds, x, y = ctx["dataset"], ctx["x"], ctx["y"]
    mf = metric_fun_for(method, ds, x, y, ctx["s"])
    cache = {}

    def u(subset):  # the gtg_evals utility: previous global for {}, else the plain weight average
        key = tuple(sorted(subset))
        if key not in cache:
            if key:
                ctx["eval_model"].set_weights(aggregate_models([weights[i] for i in key]))
                cache[key] = float(mf(ctx["eval_model"], x, ds, y))
            else:
                cache[key] = float(mf(ctx["prev_model"], x, ds, y))
        return cache[key]

    everyone = tuple(range(n))
    phi = sign * exact_shapley(u, n)
    rng = ctx["seed"] * 10_000 + n * 100 + ctx["round"]  # robustness.py's per-round seed for gap methods
    R.set_seed(rng)
    g = GTGShapley(num_players=n, loss_flag=loss)
    g.set_utility_function(u)
    raw = sign * np.array(g.compute(return_raw=True))
    best = sorted(g._find_best_subset())
    v_saved = float(mf(ctx["global_model"], x, ds, y))
    R.set_seed(rng)
    _, prod = gtg_evals(mf, ctx["model_dir"], n, x, y, ctx["round"], ds, dict(enumerate(weights)),
                        {i: i for i in range(n)}, v_saved, loss_flag=loss)
    final = sign * np.array([prod[i] for i in everyone])
    _, loo = l1o_evals(mf, v_saved, x, y, ds, dict(enumerate(weights)), ctx["global_model"])
    loo = sign * np.array([loo[i] for i in everyone])
    loo_plain = sign * np.array([u(everyone) - u(tuple(j for j in everyone if j != i)) for i in everyone])
    same_as_prod = np.allclose(final, np.where(np.isin(everyone, best), raw, 0.0), atol=1e-9)

    print(f"\n[{label}] round {ctx['round']} | {method} | {n} clients")
    print(f"  u(all, plain avg)={u(everyone):.4f}  u(saved global)={v_saved:.4f}  u(prev global)={u(()):.4f}"
          f"  best coalition={best}  coalitions evaluated={len(g.evaluated_subsets)}/{2 ** n - 1}"
          f"  production GTG == raw masked by best coalition: {same_as_prod}")
    if table:
        print("  client      exact    gtg_raw  gtg_final   loo(prod)  loo(plain)")
        for i in everyone:
            print(f"  {i:>6} {phi[i]:+10.5f} {raw[i]:+10.5f} {final[i]:+10.5f} {loo[i]:+11.5f} {loo_plain[i]:+11.5f}")
    rho = spearmanr(phi, raw).correlation if np.std(phi) > 0 and np.std(raw) > 0 else float("nan")
    print(f"  sum exact={phi.sum():+.5f} (= u(all)-u(empty) {sign * (u(everyone) - u(())):+.5f})  sum raw={raw.sum():+.5f}"
          f"  max|raw-exact|={np.abs(raw - phi).max():.5f}  spearman(exact,raw)={rho:.2f}"
          f"  sign agree={int((np.sign(raw) == np.sign(phi)).sum())}/{n}  zeroed by best coalition={int((final == 0).sum())}/{n}")
    return dict(phi=phi, raw=raw, final=final, loo=loo, u=u, sign=sign)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="adultnoniid")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--methods", nargs="+", default=["gtg_acc", "gtg_gap_acc_dp", "gtg_gap_loss_eop"])
    ap.add_argument("--root_dir", default=str(ROOT / "420"))
    args = ap.parse_args()
    assert args.K <= 6, "exact Shapley enumerates 2^(K+1) coalitions; keep K small"

    task = importlib.import_module(f"{args.dataset}.{args.dataset}.task")
    x, y, s = R.load_test_data(args.dataset)
    model_dir = os.path.join(args.root_dir, args.dataset, str(args.K), str(args.seed))
    parts = [task.load_data(pid, args.K, return_sensitive=True) for pid in range(args.K)]
    sizes = [len(p[0]) for p in parts]  # the num_examples each client reports to FedAvg

    for rnd in args.rounds:
        load = lambda r: tf.keras.models.load_model(os.path.join(model_dir, f"global_model_round_{r}.keras"), compile=False)
        global_model, prev_model = load(rnd), load(rnd - 1)
        clients = load_client_models(model_dir, args.K, rnd)
        weights = [clients[i].get_weights() for i in range(args.K)]
        ctx = dict(dataset=args.dataset, x=x, y=y, s=s, seed=args.seed, round=rnd, model_dir=model_dir,
                   global_model=global_model, prev_model=prev_model,
                   eval_model=tf.keras.models.clone_model(global_model))

        saved = global_model.get_weights()
        plain = aggregate_models(weights)
        sized = [np.average(np.stack([w[l] for w in weights]), axis=0, weights=sizes) for l in range(len(saved))]
        maxdiff = lambda a, b: max(float(np.max(np.abs(p - q))) for p, q in zip(a, b))
        print(f"\n=== round {rnd}: client training sizes {sizes}")
        print(f"  saved global vs plain average:        max|dw| = {maxdiff(saved, plain):.2e}")
        print(f"  saved global vs size-weighted average: max|dw| = {maxdiff(saved, sized):.2e}")

        planted = {}
        x0, y0, s0 = parts[0][0], parts[0][1], np.asarray(parts[0][4]).ravel()
        for grp in (1, 0):
            keep = np.flatnonzero(s0 == grp)
            m = task.load_model()
            m.set_weights(prev_model.get_weights())
            task.set_seed(task.SEED)  # client_app.fit: set_seed, then one plain model.fit
            m.fit(_take(x0, keep), _take(y0, keep), epochs=1, batch_size=32, verbose=0)
            planted[grp] = (m.get_weights(), len(keep))

        for method in args.methods:
            base = check("base", method, weights, ctx, table=True)
            dup = check("duplicate of client 0", method, weights + [weights[0]], ctx)
            print(f"  -> client 0 vs its copy: exact {dup['phi'][0]:+.5f} / {dup['phi'][-1]:+.5f}"
                  f" | gtg_raw {dup['raw'][0]:+.5f} / {dup['raw'][-1]:+.5f}"
                  f" | gtg_final {dup['final'][0]:+.5f} / {dup['final'][-1]:+.5f}")
            for grp, (pw, rows) in planted.items():
                r = check(f"planted: group {'A' if grp else 'B'} only, {rows} rows of client 0", method, weights + [pw], ctx)
                everyone = tuple(range(args.K))
                direct = r["sign"] * (r["u"](everyone + (args.K,)) - r["u"](everyone))
                rank = int((r["phi"] > r["phi"][-1]).sum()) + 1
                print(f"  -> planted client: adding it to all {args.K} clients changes the utility by {direct:+.5f};"
                      f" exact {r['phi'][-1]:+.5f} (rank {rank}/{args.K + 1}), gtg_raw {r['raw'][-1]:+.5f},"
                      f" gtg_final {r['final'][-1]:+.5f}, loo {r['loo'][-1]:+.5f}")
            del base


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""smoke_lossflag.py — direction/loss_flag smoke test on ALL datasets.

Verifies, with synthetic clients of KNOWN quality ordering (good > mid > bad):

TEST A (in-loop ST/DY convention: build_metric negates loss, loss_flag=False):
    GTG raw scores must order good > mid > bad for BOTH acc and loss.
    => proves neither acc nor loss is inverted in the weighting path.

TEST B (post-hoc BL convention: RAW loss utility + loss_flag):
    loss_flag=True  -> best subset MINIMIZES loss -> bad client excluded (0.0).
    loss_flag=False -> selection maximizes -> wrong subset (demonstrates the
    flag is selection-only, and that BL used it correctly via
    FLR_GTG_LOSSFLAG default ON in cont_evals).

Clients: c0=healthy model, c1=tiny-noise copy, c2=50/50 mix with random init,
c3=random init. Run: CUDA_VISIBLE_DEVICES= seed=42 python3 smoke_lossflag.py
"""
import os, sys
os.environ.setdefault("seed", "42")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("HF_HUB_OFFLINE", "0"); os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf

from reweight_eval import load_test_data, load_global_model, EVAL_ROUND
from score_metrics import build_metric
from inloop_scoring import build_gtg_score_provider
from weighted_utils import shift_and_normalize, fedavg_aggregate
from gtg_shap import GTGShapley

# healthy donor models on this box (verified non-collapsed in the CSVs)
DONOR = {
    "adultnoniid":  "420_st_acc/adultnoniid/4/42",
    "celebanoniid": "420_dy_rel/celebanoniid/20/42",
    "imdbnoniid":   "420_st_acc/imdbnoniid/4/42",
}

def rank_str(scores):
    order = np.argsort(scores)[::-1]
    return ">".join(f"c{i}" for i in order)

def main():
    results = []
    for ds, donor in DONOR.items():
        if not os.path.exists(os.path.join(donor, f"global_model_round_{EVAL_ROUND}.keras")):
            print(f"\n{'='*70}\n{ds}: donor {donor} missing (purged/not yet retrained) — SKIP", flush=True)
            results.append((ds, None, None, None))
            continue
        print(f"\n{'='*70}\n{ds}  (donor: {donor})", flush=True)
        x, y, s = load_test_data(ds)
        if "imdb" in ds:
            from imdbnoniid.imdbnoniid.task import process_text
            x = process_text(np.asarray(x))
        good = load_global_model(donor, EVAL_ROUND, ds)
        gw = good.get_weights()
        rng = np.random.default_rng(0)
        fresh = tf.keras.models.clone_model(good)      # random init
        bw = fresh.get_weights()
        tiny = [w + rng.normal(0, 0.01 * (np.std(w) + 1e-8), w.shape) for w in gw]
        mid  = [0.5 * a + 0.5 * b for a, b in zip(gw, bw)]
        params = [gw, tiny, mid, bw]                   # c0>c1>c2>c3 by construction
        ns = [1000, 1000, 1000, 1000]

        # standalone quality (sanity)
        probe = tf.keras.models.clone_model(good)
        accm = build_metric("acc", x, y, s_test=s, dataset=ds)
        lossm = build_metric("loss", x, y, s_test=s, dataset=ds)  # returns NEGATED loss
        for i, p in enumerate(params):
            probe.set_weights(p)
            print(f"  c{i}: acc={accm(probe):.3f} negloss={lossm(probe):.3f}", flush=True)

        # TEST A — in-loop provider, acc and loss
        ok_A = True
        for mname in ("acc", "loss"):
            mfn = build_metric(mname, x, y, s_test=s, dataset=ds)
            prov = build_gtg_score_provider(metric_fn=mfn,
                                            model_factory=lambda: tf.keras.models.clone_model(good),
                                            max_permutations=30)
            sc = prov(2, list(zip(params, ns)))
            w = shift_and_normalize(sc, beta=1.0, cap=2.0)
            good_top = sc[3] == min(sc) and (max(sc) in (sc[0], sc[1]))
            ok_A &= good_top
            print(f"  [A in-loop {mname:4}] scores={[f'{v:+.4f}' for v in sc]} "
                  f"rank={rank_str(sc)} weights={[f'{v:.2f}' for v in w]} "
                  f"{'PASS' if good_top else 'FAIL'}", flush=True)

        # TEST B — BL convention: RAW loss utility, loss_flag both ways
        model_b = tf.keras.models.clone_model(good)
        init_w = model_b.get_weights()
        def util_rawloss(subset):
            if not subset:
                model_b.set_weights(init_w)
                return -float(lossm(model_b))          # un-negate -> raw loss
            agg = fedavg_aggregate([params[i] for i in subset], [ns[i] for i in subset])
            model_b.set_weights(agg)
            return -float(lossm(model_b))
        picks = {}
        for flag in (True, False):
            g = GTGShapley(num_players=4, loss_flag=flag, max_permutations=30)
            g.set_utility_function(util_rawloss)
            vals = g.compute(round_num=2)              # return_raw=False -> best-subset
            best = g._find_best_subset()
            picks[flag] = best
            print(f"  [B BL rawloss loss_flag={flag}] best_subset={best} "
                  f"contributions={[f'{v:+.4f}' for v in vals]}", flush=True)
        ok_B = (3 not in picks[True])                  # bad client excluded when flag ON
        flag_matters = (set(picks[True]) != set(picks[False]))
        print(f"  [B verdict] bad-excluded-with-flag={ok_B} flag-changes-selection={flag_matters} "
              f"{'PASS' if ok_B else 'FAIL'}", flush=True)
        results.append((ds, ok_A, ok_B, flag_matters))

    print(f"\n{'='*70}\nSUMMARY")
    for ds, a, b, fm in results:
        if a is None:
            print(f"  {ds:14} SKIPPED (donor missing)")
        else:
            print(f"  {ds:14} inloop-direction={'PASS' if a else 'FAIL'}  "
                  f"BL-lossflag={'PASS' if b else 'FAIL'}  flag-matters={fm}")

if __name__ == "__main__":
    main()

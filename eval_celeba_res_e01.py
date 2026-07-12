"""eval_celeba_res_e01.py — re-eval celeba res at the NEW eps=0.01 (alpha=0.0025,
40 iters) for the Tab.4 cells: BL, ET-adv, ST(res-weighted), DY(res-weighted),
both K, 5 seeds. Final round-10 models only (cheap; not the GTG-contribution
gtg_adv which is the ~9h/job one). Writes results_celeba_res_e01.csv. GPU0.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/root/ulrich")
import tensorflow as tf, csv
from reweight_eval import load_test_data
from attack_metric import calculate_pgd_score

d = load_test_data("celebanoniid"); x, y = d[0], d[1]
print(f"celeba test: {len(x)}", flush=True)
SRC = {"BL": "420/celebanoniid", "ET-adv": "420_et_adv/celebanoniid",
       "ST": "420_st_res/celebanoniid", "DY": "420_dy_res/celebanoniid"}
SEEDS = [42, 107, 123, 2025, 9928]
rows = []
for name, base in SRC.items():
    for K in (4, 20):
        vals = []
        for s in SEEDS:
            p = f"/root/ulrich/{base}/{K}/{s}/global_model_round_10.keras"
            if not os.path.exists(p):
                print(f"{name} K{K} s{s}: MISSING", flush=True); continue
            m = tf.keras.models.load_model(p, compile=False)
            r = calculate_pgd_score(m, x, "celebanoniid", y, epsilon=0.01, alpha=0.0025, num_iter=40)
            vals.append(float(r)); rows.append({"setting": name, "K": K, "seed": s, "res": round(float(r), 4)})
        if vals:
            import numpy as np
            print(f">>> {name} celeba K{K} res: {np.mean(vals):.3f} +- {np.std(vals):.3f} (n={len(vals)})", flush=True)
with open("/root/ulrich/results_celeba_res_e01.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["setting", "K", "seed", "res"]); w.writeheader(); w.writerows(rows)
print("wrote results_celeba_res_e01.csv", flush=True)

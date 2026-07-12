"""Generate the bundled sample dataset for the live-scoring tab.

Five synthetic "clients" over an Adult-like binary task. The point is that two clients
look unremarkable on headline accuracy, yet contribution scoring exposes their hidden costs:

  * client_0..2 — clean, informative, IID data.
  * client_3   — label noise (labels drawn at random). Barely dents accuracy (an L2 model
                 shrugs off 20% noise) but it's the single largest swing in model loss /
                 calibration — the "looks fine, quietly destabilizes" client.
  * client_4   — group-biased labels (y forced to track `sex`). The fairness villain:
                 by far the most negative demographic-parity contribution.

The takeaway for the demo: accuracy alone hides both problems; the trust axes surface them.

Deterministic (seed 42). Run:  python make_sample_data.py
"""

import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_PER_CLIENT = 400
rows = []

for cid in range(5):
    age = rng.normal(40, 12, N_PER_CLIENT).clip(18, 90)
    hours = rng.normal(40, 10, N_PER_CLIENT).clip(5, 80)
    edu = rng.integers(6, 17, N_PER_CLIENT)
    sex = rng.choice(["F", "M"], N_PER_CLIENT)
    # Ground-truth signal: income depends strongly on education + hours (not on sex).
    # Coefficients are deliberately large so the clean task is highly learnable
    # (~0.9 accuracy); that headroom is what lets L1O expose a bad actor.
    logit = 0.9 * (edu - 11) + 0.12 * (hours - 40) + 0.04 * (age - 40)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(N_PER_CLIENT) < p).astype(int)

    if cid == 3:  # label noise: labels drawn at random, independent of features.
        # Actively conflicting training signal (hurts accuracy + loss), but because
        # the labels are random rather than systematically flipped, this client's own
        # test rows sit at ~50% for ANY model, so the L1O sign stays uninverted.
        y = rng.integers(0, 2, N_PER_CLIENT)
    if cid == 4:  # biased client: forces income=1 for M, 0 for F on half the rows
        biased = rng.random(N_PER_CLIENT) < 0.5
        y = np.where(biased, (sex == "M").astype(int), y)

    rows.append(pd.DataFrame({
        "client": f"client_{cid}",
        "age": age.round(1),
        "education_years": edu,
        "hours_per_week": hours.round(1),
        "sex": sex,
        "income": y,
    }))

out = pd.concat(rows, ignore_index=True)
dest = os.path.join(os.path.dirname(__file__), "sample_data", "clients_demo.csv")
os.makedirs(os.path.dirname(dest), exist_ok=True)
out.to_csv(dest, index=False)
print(f"wrote {dest} ({len(out)} rows)")

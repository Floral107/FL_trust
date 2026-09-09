# FL_trust — Trustworthiness of Client Contribution Scoring in Federated Learning

Code and result artifacts for the thesis experiments: federated contribution
scoring (leave-one-out and GTG-Shapley) evaluated across accuracy, loss,
fairness (DP/EO), robustness (PGD/C&W) and privacy (MIA) on three datasets
(ADULT, CelebA, IMDB) under IID and non-IID (Dirichlet α=0.5) partitions,
aggregated with FedAvg over 5 seeds. CIFAR-10 was also run and its CSVs ship
under `data/combo/`, but it is not reported in the thesis. MIT licensed.

## Layout

| Path | What it is |
|------|-----------|
| `adult/ celeba/ cifar/ imdb/ *noniid/` | Flower client/server apps per dataset (`client_app.py`, `server_app.py`, `task.py`, `pyproject.toml`) |
| `score_metrics.py`, `gtg_shap.py`, `inloop_scoring.py`, `fast_infer.py` | Contribution scoring (L1O, GTG-Shapley) |
| `fairness_metric.py`, `robustness_metric.py`, `privacy_metric.py`, `attack_metric.py` | Trustworthiness metrics |
| `pgd_attack.py`, `cw_attack.py` | Adversarial attacks |
| `et_training.py`, `weighted_strategy.py`, `weighted_utils.py`, `reweight_eval.py` | Explicit-trust training (ET) and score-based reweighting (ST/DY) |
| `robustness.py` | The post-hoc scoring driver (`--method {gtg,l1o}_<metric>`) |
| `cont_evals.py` | Canonical per-round evaluation — the single place the loss-sign convention lives |
| `eval_{bl,et,stdy}_models.py`, `eval.ps1` | Batch model evaluation |
| `run_experiments.ps1`, `run_et_experiments.{ps1,sh}`, `set_environment.ps1` | Experiment launchers |
| `paper/` | Gate, table/figure builders, thesis splicers. Run from the repo root. |
| `data/combo/` | Per-client contribution scores (L1O + GTG), 16 files × 1200 rows |
| `data/{bl,st,dy,et}/` | Final-model 7-metric evaluations for the utilization table |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`numpy<2.1` and `tensorflow<2.18` are pinned together: TF 2.17 against numpy 2.x
fails at import with `initialization of _pywrap_checkpoint_reader raised
unreported exception`.

## Running

**Seed.** Every app reads the seed from the lowercase `seed` environment
variable *at import time*. Set it before any `flwr run` or `robustness.py`
invocation (`$env:seed = "42"` / `export seed=42`); the launchers do this for
you. `set_environment.ps1` additionally pins `PYTHONHASHSEED` and the TensorFlow
determinism flags. Seeds used: `{42, 107, 123, 2025, 9928}`.

1. **Training + in-loop scoring** — `run_experiments.ps1 <ds> <iid|noniid> <K>`
   runs the Flower app and writes per-round checkpoints to `420/<ds>/<K>/<seed>/`.
2. **Post-hoc scoring** — `robustness.py --dataset <ds> --partition <p>
   --num_rounds 10 --seed <s> --method {gtg,l1o}_<metric>` writes
   `results_<ds>_<seed>_<method>_<metric>_fedavg.csv`, merged into `data/combo/`.
   Set `FLR_FORCE_CPU=1`: GPU evaluation after PGD raises an XLA device error.
   This step reads the `420/` checkpoints from step 1, which are **not** shipped
   (tens of GB); without them the driver exits non-zero with an empty CSV.
3. **Tables and figures** — `python paper/sanity_gate.py` first (it must pass),
   then the builders in `paper/`. Run them from the repo root; they use
   cwd-relative `data/` paths.

The merged CSVs under `data/` are included, so the reported numbers can be
inspected without retraining.

## The sign convention

**Every `*_contribution` column in `data/combo/` is higher = better, for every
metric, including loss.** The negation for loss happens in exactly one place —
`cont_evals.py :: evaluate_round`. `global_*_loss` stays raw cross-entropy. No
merge, collect, plot or table script may flip signs; doing so previously caused
a double-inversion that took two rounds of debugging to unpick.
`paper/sanity_gate.py` enforces this: for every dataset the acc~loss Spearman of
contributions must be positive under both L1O and GTG.

## Known limitations and concerns

Recorded during a review of this repository before upload. The first group is
established; the second is a list of concerns, not settled results.

**Reproducibility.** GTG contributions depend on the order rounds were scored in,
not only on `(dataset, seed, method)` — `set_seed` runs once per seed while
GTG-Shapley draws from the global numpy RNG each round, so resuming a run or
changing `FLR_ROUND_ORDER` shifts them (measured: 21 of 40 values moved by up to
3.9e-3). L1O is unaffected. ADULT is re-derived from OpenML on every evaluation
with `scikit-learn`/`pandas` unpinned, so it reproduces to ~7.6e-3 rather than
exactly. `paper/sanity_gate.py --strict` reports per-cell seed coverage; 17 cells
are currently backed by fewer than 5 seeds.

**Concerns.**

- *Possible non-convergence in some ET arms.* The CelebA ET-fair and ET-dp runs
  sit near chance accuracy (~0.5 on a binary task), ET-fair at loss ≈ ln 2.
  Where a model is near-constant, fairness and privacy saturate for trivial
  reasons, so those utilization-table cells may not be comparable to the rest.
  The cause is not established — two candidate mechanisms were tested and neither
  reproduced it. ADULT/IMDB ET arms and CelebA ET-adv appear unaffected, but the
  check was not exhaustive and other cells may be involved.
- *Utilization-table rows are per-metric diagonals* — each row comes from the run
  weighted for that metric, so the accuracy row is not the accuracy of the model
  in the resilience row. At CelebA K=4 the resilience-targeted run costs 0.25
  accuracy; the claim that accuracy stays within ~0.02 of the baseline may need
  rephrasing. The full weight-metric × eval-metric matrix is in `data/{st,dy}/`.

## License

MIT — see [LICENSE](LICENSE).

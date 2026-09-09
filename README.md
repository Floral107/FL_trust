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

## Known limitations

- **GTG is not reproducible from the command line alone.** `set_seed` runs once
  per seed, outside the round loop, while GTG-Shapley draws 30 shuffles per round
  from the global numpy RNG. Resuming an interrupted run, `FLR_ROUND_ORDER`, and
  `--num_clients` all change the stream. Measured: reversing round order moved
  21 of 40 contributions by up to 3.9e-3. L1O is unaffected.
- **ADULT is re-derived from OpenML on every evaluation.** No cached matrix
  (CelebA caches `.npz`, IMDB caches embeddings), and `scikit-learn`/`pandas` are
  unpinned, so a clean-room re-run reproduces ADULT to ~7.6e-3, not exactly.
- **Coverage.** `paper/sanity_gate.py` reports per-cell seed coverage; `--strict`
  makes a thin cell fail the gate. 17 cells are currently backed by fewer than
  5 seeds.

## Concerns found during the check before upload

Found while reviewing this repository prior to upload, recorded here so they are
not mistaken for accepted results. Some may require rephrasing in the thesis.

- **The CelebA ET-fair and ET-dp arms did not converge.** Both sit at ~0.5
  accuracy on a binary task; ET-fair's loss is ≈ ln 2 (uniform output), ET-dp's
  is 7.4–8.6 (diverged). Their fairness and privacy scores are therefore
  artifacts — a constant predictor has perfect demographic parity by
  construction — and those utilization-table cells should be disregarded pending
  a re-run. **The cause has not been identified.** Two candidate mechanisms were
  tested in isolation and neither reproduces the collapse: the fairness penalty
  at λ=2.0 on a comparable architecture trains to 0.98 accuracy, and DP-SGD at
  the shipped (C=1.0, σ=1.0, batch=32) on a 1.1M-parameter model trains to 0.92.
  Since the same CelebA ET-fair cells also show the duplicated-seed fault below,
  a launch/configuration problem in those specific runs is the more likely
  explanation than anything in the training math — but that is untested.
  ET-adv and all ADULT/IMDB ET arms are unaffected.
- **Three ET-fair seeds are duplicated** between K=4 and K=20 (123, 2025, 9928):
  identical `acc`, `loss` and `res` to the last digit, with only the stochastic
  metrics differing. Those two cells share a model. A candidate mechanism, not
  confirmed: `set_num_cl.py` rewrites a shared `pyproject.toml` in place, so
  concurrent cells on one host can read each other's client count.
- **Utilization-table rows are per-metric diagonals.** The `acc` row is the
  accuracy-weighted run; the `res` row is the resilience-weighted run. They are
  different models, so the accuracy figure is not the accuracy of the model in
  the `res` row. The claim that accuracy stays within ~0.02 of the baseline holds
  at K=20 but not at CelebA K=4, where targeting resilience costs 0.25 accuracy
  (0.913 → 0.660) for a 0.31 resilience gain. The full weight-metric ×
  eval-metric matrix is already collected in `data/{st,dy}/`; may need rephrasing.
- **Weight clipping.** The thesis describes clipping each weight to [0.5, 2];
  every committed config has `weight-cap = 0.0`, which disables it, with
  `weight-beta = 1.0` (raw transform). Needs resolving in one direction or the
  other.
- **Metric protocol drift.** The MIA configuration used for `data/combo/` (5
  fine-tune epochs, lr 5e-4, 0.2 member ratio) differs from the one used for
  `data/{bl,st,dy,et}/` and from the thesis (2 epochs, lr 1e-3, 0.5). The CelebA
  in-loop resilience proxy attacks at ε=0.03 while evaluation uses ε=0.01.

## License

MIT — see [LICENSE](LICENSE).

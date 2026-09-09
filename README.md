# FL_trust — Trustworthiness of Client Contribution Scoring in Federated Learning

Code and result artifacts for the paper's experiments: federated contribution
scoring (leave-one-out and GTG-Shapley) evaluated across accuracy, loss,
fairness (DP/EO), robustness (PGD/C&W), and privacy (MIA) on three datasets
(Adult, CelebA, IMDB) under IID and non-IID (Dirichlet α=0.5) partitions,
aggregated with FedAvg, 5 seeds. CIFAR-10 was also run and its CSVs ship under
`data/combo/`, but it is not reported in the paper. Released under the MIT
License.

## Layout

| Path | What it is |
|------|-----------|
| `adult/ celeba/ cifar/ imdb/ *noniid/` | Flower client/server apps per dataset (`client_app.py`, `server_app.py`, `task.py`, `pyproject.toml`) |
| `score_metrics.py`, `gtg_shap.py`, `inloop_scoring.py`, `fast_infer.py` | Contribution scoring (L1O, GTG-Shapley) |
| `fairness_metric.py`, `robustness_metric.py`, `privacy_metric.py`, `attack_metric.py` | Trustworthiness metrics |
| `pgd_attack.py`, `cw_attack.py` | Adversarial robustness attacks |
| `et_training.py`, `weighted_strategy.py`, `weighted_utils.py`, `reweight_eval.py` | Explicit-trust training and score-based reweighting |
| `robustness.py` | The post-hoc evaluation driver (`--method {gtg,l1o}_<metric>`) |
| `cont_evals.py` | Canonical per-round evaluation (single place for the loss-sign convention) |
| `eval_{bl,et,stdy}_models.py`, `eval.ps1` | Batch model evaluation |
| `run_experiments.ps1`, `run_et_experiments.{ps1,sh}` | Experiment launchers |
| `data/combo/`, `data/{bl,st,dy,et}/` | The merged result CSVs the paper's tables and figures are built from |
| `paper/` | Paper assembly: `sanity_gate.py` (run first), table/figure builders, thesis splicers. Run from the repo root: `python paper/sanity_gate.py`. The thesis source itself is not in this repo. See `PIPELINE.md`. |
| `archive/` | Retired material — compute-box snapshots, per-host scripts, one-off analyses, superseded builders, data backups. Excluded from the release: paths under `archive/` mentioned in `PIPELINE.md` are provenance notes, not runnable. |

## Reproducing the result CSVs

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Training + in-loop scoring**: `run_experiments.ps1` /
   `run_et_experiments.sh` run the Flower apps per dataset/seed and write
   per-round checkpoints.
2. **Post-hoc scoring**: `robustness.py --method {gtg,l1o}_<metric>` produces
   the raw `results_<ds>_<seed>_<method>_<metric>_fedavg.csv` files, merged
   into `data/combo/`. Post-hoc evals must run on CPU — set `FLR_FORCE_CPU=1`
   (GPU evaluation after PGD raises an XLA device error and writes no rows).

   This step reads the round checkpoints written by step 1 from `420/`, which
   is **not** shipped (tens of GB). Without it the driver finds no models and
   exits non-zero with an empty CSV.

The final merged CSVs are included under `data/`, so the paper's numbers can
be checked without retraining.

Determinism: seeds `{42, 107, 123, 2025, 9928}`. Every app reads the seed from
the lowercase `seed` environment variable at import time, so set it before any
`flwr run` or `robustness.py` invocation (`$env:seed = "42"`); the launchers in
`run_experiments.ps1` / `run_et_experiments.sh` do this for you.
`set_environment.ps1` additionally pins `PYTHONHASHSEED` and the TensorFlow
determinism flags. Known limitations of the reproduction — GTG's dependence on
round-scoring order, and the unpinned ADULT feature pipeline — are documented in
`PIPELINE.md`.

## License

MIT — see [LICENSE](LICENSE).

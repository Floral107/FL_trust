# FL_trust — Trustworthiness of Client Contribution Scoring in Federated Learning

Code and result artifacts for the paper's experiments: federated contribution
scoring (leave-one-out and GTG-Shapley) evaluated across accuracy, loss,
fairness (DP/EO), robustness (PGD/C&W), and privacy (MIA) on four datasets
(Adult, CelebA, CIFAR-10, IMDB) under IID and non-IID (Dirichlet α=0.5)
partitions, aggregated with FedAvg, 5 seeds. Released under the MIT License.

## Layout

| Path | What it is |
|------|-----------|
| `adult/ celeba/ cifar/ imdb/ *noniid/` | Flower client/server apps per dataset (`client_app.py`, `server_app.py`, `task.py`, `pyproject.toml`) |
| `score_metrics.py`, `gtg_shap.py`, `inloop_scoring.py` | Contribution scoring (L1O, GTG-Shapley) |
| `fairness_metric.py`, `robustness_metric.py`, `privacy_metric.py`, `attack_metric.py` | Trustworthiness metrics |
| `pgd_attack.py`, `cw_attack.py`, `eps_calibrate.py` | Adversarial robustness attacks |
| `et_training.py`, `weighted_strategy.py`, `reweight_eval.py` | Explicit-trust training and score-based reweighting |
| `robustness.py` | The post-hoc evaluation driver (`--method {gtg,l1o}_<metric>`) |
| `cont_evals.py` | Canonical per-round evaluation (single place for the loss-sign convention) |
| `data/combo/`, `data/{bl,st,dy,et}/` | Durable merged result CSVs the tables/figures are built from |
| `compute_score_*.py`, `build_tab5.py`, `_run_figs.py`, `short_figs.py` | Table and figure generation |
| `figs/` | Generated figures as used in the paper |
| `run_experiments.ps1`, `run_et_experiments.{ps1,sh}`, `full_stdy_rerun.sh` | Experiment launchers |
| `demo/` | Self-contained Streamlit demo (Docker + CI) for exploring results and live L1O scoring |
| `PIPELINE.md` | The data → tables/figures pipeline, sign conventions, and gotchas |

## Reproducing

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Training + scoring**: `run_experiments.ps1` / `run_et_experiments.sh` run the
   Flower apps per dataset/seed and write per-round checkpoints; `robustness.py
   --method {gtg,l1o}_<metric>` produces the raw
   `results_<ds>_<seed>_<method>_<metric>_fedavg.csv` files. Post-hoc evals must
   run on CPU (`CUDA_VISIBLE_DEVICES=`), see `PIPELINE.md`.
2. **Tables & figures**: the merged results are included in `data/combo/` and
   `data/{bl,st,dy,et}/`, so step 1 can be skipped — run
   `compute_score_diff.py`, `compute_score_fluct.py`, `build_tab5.py`,
   `_run_figs.py` — see the ordered table in `PIPELINE.md`.

Determinism: seeds `{42, 107, 123, 2025, 9928}`; `set_environment.ps1` pins
`GLOBAL_SEED`, `PYTHONHASHSEED`, and TensorFlow determinism flags.

## Demo

`demo/README.md` — Streamlit app (also Dockerized) to explore the precomputed
results and score uploaded client CSVs via live leave-one-out.

## License

MIT — see [LICENSE](LICENSE).

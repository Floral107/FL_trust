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
| `score_metrics.py`, `gtg_shap.py`, `inloop_scoring.py`, `fast_infer.py` | Contribution scoring (L1O, GTG-Shapley) |
| `fairness_metric.py`, `robustness_metric.py`, `privacy_metric.py`, `attack_metric.py` | Trustworthiness metrics |
| `pgd_attack.py`, `cw_attack.py` | Adversarial robustness attacks |
| `et_training.py`, `weighted_strategy.py`, `weighted_utils.py`, `reweight_eval.py` | Explicit-trust training and score-based reweighting |
| `robustness.py` | The post-hoc evaluation driver (`--method {gtg,l1o}_<metric>`) |
| `cont_evals.py` | Canonical per-round evaluation (single place for the loss-sign convention) |
| `eval_{bl,et,stdy}_models.py`, `eval.ps1` | Batch model evaluation |
| `run_experiments.ps1`, `run_et_experiments.{ps1,sh}` | Experiment launchers |
| `data/combo/`, `data/{bl,st,dy,et}/` | The merged result CSVs the paper's tables and figures are built from |

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
   into `data/combo/`. Post-hoc evals must run on CPU
   (`CUDA_VISIBLE_DEVICES=`).

The final merged CSVs are included under `data/`, so the paper's numbers can
be checked without retraining.

Determinism: seeds `{42, 107, 123, 2025, 9928}`; `set_environment.ps1` pins
`GLOBAL_SEED`, `PYTHONHASHSEED`, and TensorFlow determinism flags.

## License

MIT — see [LICENSE](LICENSE).

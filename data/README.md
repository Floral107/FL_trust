# Data layout — FLRobustness results

All Tab.4 / figure data lives here, organized by **strategy family**. Every file is
a tidy CSV (one row per experiment cell) so you can load with `pandas` and pivot
into any table/graph.

## Common dimensions (all families)
- **datasets**: `adultnoniid`, `imdbnoniid`, `celebanoniid` (all non-IID; Tab.4 is NIID-only)
- **num_clients (K)**: `4`, `20`
- **seed**: `42, 107, 123, 2025, 9928`
- **eval metrics** (7): `acc, loss, rel, res, fairDP, fairEO, priv`
  - `imdbnoniid` has no sensitive attribute → `fairDP`/`fairEO` are `NaN`
- `partition` is always `noniid`; `round` is always `10` (final global model)

## The four families

### `combo/` — BL (baseline, equal-weight FedAvg)
Per-client **contribution scores** (GTG-Shapley + leave-one-out) of the plain
baseline. This is the "no weighting / no special training" reference.
- Files: `results_<ds>_gtg.csv`, `results_<ds>_l1o.csv`
- Row = (dataset, partition, num_clients, seed, round, client_id)
- Columns: `global_<method>_<metric>`, `<method>_<metric>_contribution` for each metric
- Target size: **1200 rows** (5 seeds × 10 rounds × 24 client-slots[K4+K20])
- Built by: `robustness.py` (per seed/method/metric) → `merge_celeba.py` (celeba) / collector

### `et/` — ET (Explicit Training)
Models trained with an in-loss intervention, scored on all 7 metrics.
- Files: `results_<ds>_et.csv`
- Row = (dataset, partition, num_clients, seed, **mode**, round)
  - `mode` ∈ `fair` (DP penalty), `adv` (PGD adversarial training), `dp` (DP-SGD)
- Columns: `acc, loss, rel, res, fairDP, fairEO, priv`
- Built by: `eval_et_models.py` (reads `420_et_<mode>/<ds>/<K>/<seed>/`)

### `st/` — Static weighing
FedAvg where client aggregation weights are frozen at round 2 from one metric's
contribution scores, then applied rounds 3-10. Scored on all 7 metrics.
- Files: `results_<ds>_st.csv`
- Row = (dataset, partition, num_clients, seed, **weight_metric**, round)
  - `weight_metric` = the metric whose contribution drove the weighting
    (`acc, loss, fairdp, faireo, rel, res, priv`; imdb: no fair)
- Columns: `acc, loss, rel, res, fairDP, fairEO, priv`
- The **diagonal** (eval metric == weight_metric) is the Tab.4 ST cell; off-diagonal
  shows how weighting-by-X affects metric-Y.
- Built by: `eval_stdy_models.py --strats st` (reads `420_st_<weight_metric>/...`)

### `dy/` — Dynamic weighing
Like `st`, but weights are recomputed fresh each round from that round's scores
(not frozen). Same schema as `st`.
- Files: `results_<ds>_dy.csv`
- Built by: `eval_stdy_models.py --strats dy` (reads `420_dy_<weight_metric>/...`)

### `weighting_effect.csv` — did ST/DY weighting actually do anything?
A sanity flag so you can tell a *real* weighting result from one that silently
collapsed to baseline. For each (dataset, strat, K, seed, weight_metric) it gives
`reldiff_vs_bl` = ‖W_stdy − W_bl‖ / ‖W_bl‖ (weight-space distance of the ST/DY
round-10 model from the matching BL model) and `equiv_bl` = `reldiff_vs_bl < 1e-3`.
- `equiv_bl == True` → the metric had **no signal** to weight by, so the run is
  effectively identical to BL (e.g. **imdb `res`**: USE-embedding classifiers are
  ~0-resilient, so every client looks the same → uniform weights). Treat that
  st/dy cell as "= BL / null result", not a distinct number.
- Built by `weighting_effect.py` (only needs the model files). Distribution is
  bimodal: real weighting ≈ 0.2–0.8, no-effect ≈ 2e-4.

## Regenerating / collecting
- `python eval_et_models.py   --datasets <ds>`  → `et/`
- `python eval_stdy_models.py --datasets <ds>`  → `st/`, `dy/`  (idempotent, resumable)
- `bash collect_data.sh`  → pulls all result CSVs from the compute hosts into here
- Models live in `420/` (BL), `420_et_<mode>/` (ET), `420_{st,dy}_<weight_metric>/` (ST/DY)

## Completeness (update as runs finish)
- combo: adult/imdb/cifar = 1200 ✓ ; celeba/celebanoniid = filling (heavy metrics on GPU)
- et: adult/imdb ✓ ; celeba = pending eval on A40
- st/dy: being evaluated (CrySyS = adult/imdb-K4 ; A40 = celeba/K20)

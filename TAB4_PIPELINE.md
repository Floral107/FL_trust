# Completing MAIN.tex Table 4 (`tab:util`) — what does what

**Tab.4 shape:** 4 strategies (**BL / ST / DY / ET**) scored on **7 metrics**
(loss, acc, fairDP, fairEO, rel, res, priv) for **3 datasets** (adultnoniid,
imdbnoniid, celebanoniid) × **K ∈ {4, 20}**, averaged over **5 seeds**
{42,107,123,2025,9928}. imdb has no sensitive attr → fairDP/fairEO = N/A.

Every cell = (mean ± std over seeds) of one strategy's round-10 model evaluated
on one metric. The whole pipeline exists to produce those cells.

## The 4 rows — train → evaluate → data file

| Row | Trained by | Models live in | Evaluated by | Data file |
|-----|-----------|----------------|--------------|-----------|
| **BL** (plain FedAvg) | `bl_guarded_run.sh` (`weight-mode=none`) | `420/<ds>/<K>/<seed>/` | `robustness.py`→`merge_celeba.py` (contrib) **+** utility eval | `data/combo/` (contrib) **+ BL utility, see gap below** |
| **ST** (static weighting) | `st_dy_guarded_run.sh st <metric>` | `420_st_<metric>/...` | `eval_stdy_models.py --strats st` | `data/st/results_<ds>_st.csv` |
| **DY** (dynamic weighting) | `st_dy_guarded_run.sh dy <metric>` | `420_dy_<metric>/...` | `eval_stdy_models.py --strats dy` | `data/dy/results_<ds>_dy.csv` |
| **ET** (explicit training) | `et_guarded_run.sh <fair/adv/dp>` | `420_et_<mode>/...` | `eval_et_models.py` | `data/et/results_<ds>_et.csv` |

ST/DY/ET CSV schema: `dataset,partition,num_clients,seed,{mode|weight_metric},round,acc,loss,rel,res,fairDP,fairEO,priv`.

## How training works (the FL machinery)
- `flwr run` (Flower + TF) with a custom strategy in **`weighted_strategy.py`** (`WeightedFedAvg`: BL/ST/DY aggregation).
- ST/DY weighting signal: **`inloop_scoring.py`** (GTG score provider) → **`gtg_shap.py`** (GTG-Shapley) using cheap in-loop metric proxies in **`score_metrics.py`**; weights derived by **`weighted_utils.py`** (`shift_and_normalize`, `weighted_aggregate`).
- ET modifies the *local* loss (fair=DP penalty, adv=PGD batches, dp=DP-SGD); no server weighting.

## How evaluation works (the metric values)
All three evaluators call **`reweight_eval.eval_all_metrics(model, x, y, s, ds)`**, which computes the 7 metrics via:
- `cont_evals._accuracy_score / _loss_score` → acc, loss
- `robustness_metric.calculate_robustness_score` → rel (noise reliability)
- `attack_metric.calculate_pgd_score` → res (PGD resilience)
- `fairness_metric.calculate_fairness_score` (dp/eo) → fairDP, fairEO
- `privacy_metric` + `cont_evals._build_privacy_value` (MIA) → priv

This guarantees BL/ST/DY/ET are scored identically.

## QA layer (so the table is trustworthy)
- **`weighting_effect.py`** → `data/weighting_effect.csv`: per cell, `reldiff_vs_bl` + `equiv_bl`. If `equiv_bl=True`, that ST/DY weighting had no signal (e.g. imdb-res ≈ 0) → the cell ≡ BL; report as null, not a distinct number.
- **`round0_audit.py`** → `data/init_match.csv`: confirms BL/ST/DY share the same init. `purge_mismatched.py` quarantines + retrains init-mismatched cells. (Resolved: ST/DY consistent; BL ~0.06 init offset for 4 seeds accepted as negligible.)

## Collection
- **`collect_data.sh`** (+ `data/merge_collected.py`): pulls all result CSVs from A40 + CrySyS into local `data/{et,st,dy}` (dedup-merge) + `weighting_effect.csv`; celeba combo via `merge_celeba.py` on host.
- `data/README.md` documents the convention.

## ⚠️ The missing piece (what actually writes the table)
There is **no Tab.4 extraction script yet**. Final step:
`build_tab4.py` (TODO) — read `data/{combo,et,st,dy}` + `weighting_effect.csv`,
group by (dataset, K, strategy, metric), compute mean±std over seeds, mark
≡BL cells, and emit the LaTeX rows for `tab:util`.

## Gap: BL utility consistency
ET/ST/DY utility comes from `eval_all_metrics`. The **BL row** utility currently
lives implicitly in combo's `global_<method>_<metric>` (round-10 global). For an
apples-to-apples table, run `eval_all_metrics` on the `420/` BL models too
(a 5-line clone of `eval_stdy_models.py` → `data/bl/results_<ds>_bl.csv`), or
verify combo's global_* equals it. **Recommended: add the BL utility eval.**

## Other MAIN.tex items (beyond Tab.4)
- **§4.2 ET hyperparameters**: filled (λ=0.1, adv-ratio=0.5, ε/α, 7 steps, C=σ=1.0).
- **DP-SGD ε table**: computed (δ=1e-5, σ=C=1: imdb/celeba ε≈2.2–2.4, adult-K4≈5.6, adult-K20≈15.5) — **not yet placed** in MAIN.tex.
- **Contribution figures** (gtg/l1o): from `data/combo/` via `short_figs.py` / `article_figs.ipynb`.

## Status snapshot (update as runs finish)
- BL models 420/: ✓ (celeba combo contrib filling on GPU)
- ET data: adult/imdb ✓ ; celeba pending eval on A40
- ST/DY models: re-training for init-consistency (CrySyS done; A40 celeba ~day)
- ST/DY data: pending (eval_stdy resumes after re-train)
- weighting_effect: built, partial
- **build_tab4.py: not built (final step)**
- BL utility eval: not built (gap above)

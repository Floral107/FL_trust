# REPO_MAP.md — FLRobustness

Read-only survey of the repo as of 2026-09-01 (branch `main`, 9 commits).
Nothing has been moved or deleted to produce this document.

---

## 0. The one thing to understand first: two repos live in one directory

`.gitignore` is doing heavy curation. There are **two different "repos"** here:

| View | Size | What it is |
|---|---|---|
| **Tracked** (`git ls-files`) | **111 files** | The public code release (github.com/Floral107/FL_trust): 8 Flower app dirs, 15 root modules, `data/{combo,bl,st,dy,et}` CSVs, 3 launchers, `requirements.txt`, `LICENSE`, `README.md`. |
| **Working tree** | ~22 000 files | The above **plus** 12 100 model checkpoints (`420/`), a full compute-box snapshot (`a40_snapshot/`, 255 files), `_archive/` (174), `_pull/` (11 tarballs), 30 one-off `.sh` box scripts, 11 `jobs_*.txt`, the paper (`MAIN.tex`, `FedKDD.tex`), table/figure builders, and 5 internal `.md` status docs. |

Almost everything the cleanup is about is in the **untracked-but-present** layer.
So "archive" here mostly means *move on disk*, not *remove from git* — the git
release is already clean.

**Consequence for step 2:** moving a gitignored file into `archive/` produces a
zero-line `git diff`. That is expected, not a bug (this bit the last cleanup —
see `advisor-round3-verification` note about "git diff on tex is vacuous").

---

## 1. Entry points

### 1a. The main pipeline (the one that produces `data/combo/`)

```
[1] TRAIN     run_experiments.ps1 <ds> <iid|noniid> <K> [seed|all] [strategy]
                └─ set_num_cl.py  (rewrites <ds>/pyproject.toml num-supernodes)
                └─ cd <ds> && flwr run . --run-config "strategy='fedavg' proximal-mu=0"
                     server_app.py  → saves 420/<ds>/<K>/<seed>/global_model_round_<r>.keras
                     client_app.py  → saves 420/<ds>/<K>/<seed>/client_<i>_round_<r>.keras

[2] SCORE     python robustness.py --dataset <ds> --partition <p>
                     --num_rounds 10 --seed <s> --method {gtg,l1o}_<metric>
                └─ cont_evals.evaluate_round  (THE canonical eval + loss-sign flip)
                     ├─ l1o_evals   (leave-one-out)
                     └─ gtg_evals   (gtg_shap.GTGShapley)
                └─ metric modules: robustness_metric / attack_metric /
                                   fairness_metric / privacy_metric
                → ./results_<ds>_<seed>_<method>_<metric>_fedavg.csv     (raw, one metric)

[3] MERGE     merge_celeba.py (celeba) / column merges (adult, imdb)
                → data/combo/results_<ds>_{gtg,l1o}.csv                  (durable artifact)
```

This is the pipeline I will use for the step-4 definition of done. It is the only
flow that is fully represented in the tracked release and that has local inputs
(`420/adultnoniid/**` checkpoints are present on this machine).

### 1b. Secondary pipelines (Tab. 5 — the ST/DY/ET utilization table)

```
run_et_experiments.{sh,ps1}          → 420_et_{fair,adv,dp}/…   → eval_et_models.py   → data/et/
flwr run with weight-mode=st|dy      → 420_{st,dy}_<metric>/…   → eval_stdy_models.py → data/{st,dy}/
(plain fedavg, reuses 420/)                                     → eval_bl_models.py   → data/bl/
```
All three eval scripts share `reweight_eval.eval_all_metrics` (the 7-metric scorer).

### 1c. Paper assembly (all gitignored; documented in `PIPELINE.md`)

`sanity_gate.py` → `compute_score_diff.py` → `_splice_scorediff.py` →
`compute_score_fluct.py` → `_splice_scorefluct.py` → `_run_figs.py` →
`build_tab5.py` → `_splice_tab5.py`.

### 1d. Other executables

| Entry point | Status |
|---|---|
| `eval.ps1` | **Broken** — dispatches `gtg_acc`/`*_loss` to `accuracy.py`/`loss.py`, which were archived. Also hardcodes an absolute Windows path. See audit A2. |
| `set_environment.ps1` | Sourced manually to pin seed/determinism env. |
| `dashboard.py` | `streamlit run dashboard.py` — exploratory only. |
| `demo/app.py` | Streamlit demo; `.gitignore` says it "lives in its own repo". |
| `test_inloop_scoring.py`, `test_weighted_strategy.py`, `smoke_lossflag.py` | The only tests. All gitignored. |

---

## 2. Module map (what each root module does)

### Core library — the science (all tracked, all live)

| Module | Role |
|---|---|
| `cont_evals.py` | **The hub.** `evaluate_round()` loads a round's global + client ckpts, dispatches to a metric, and runs LOO or GTG. Owns the canonical loss-sign negation (line 226-227) — the single place sign logic is allowed to exist. |
| `gtg_shap.py` | `GTGShapley` — permutation-sampling Shapley with truncation + best-subset selection. |
| `score_metrics.py` | *Cheap* metric builders for **in-loop** ST/DY scoring (fewer PGD iters, no MIA). Distinct from the post-hoc metrics below. |
| `inloop_scoring.py` | Wraps `GTGShapley` into a `score_provider` closure for `WeightedFedAvg`. |
| `weighted_strategy.py` | `WeightedFedAvg` — the ST/DY Flower strategy (freeze-at-round-2 vs recompute-each-round). |
| `weighted_utils.py` | Pure-numpy `shift_and_normalize` (β shrinkage + cap) and weighted aggregation. No TF/flwr import. |
| `robustness_metric.py` | `rel` — certified reliability under Gaussian noise; plus `load_client_models` / `aggregate_models` used by `cont_evals`. |
| `attack_metric.py` | `res` — PGD and C&W attack success. Owns `PGD_EVAL_PARAMS` (the per-dataset eval ε/α/iters). |
| `fairness_metric.py` | `fairDP` / `fairEO`. |
| `privacy_metric.py` | `priv` — MIA via ART, with fine-tune-on-members protocol. Owns `MIAConfig`. |
| `pgd_attack.py`, `cw_attack.py` | The raw attack implementations. |
| `et_training.py` | `train_with_{fair,adv,dpsgd}` — the ET local-training losses, imported by every `*noniid` `task.py`. |
| `fast_infer.py` | Cached `@tf.function` forward pass; a pure speed helper for in-loop scoring. |

### Drivers / collectors (tracked)

| Module | Role |
|---|---|
| `robustness.py` | **The only post-hoc scoring driver.** CLI over (dataset, partition, seed, method). Resume-safe via `_load_done_keys`. |
| `reweight_eval.py` | Two jobs: (a) `eval_all_metrics` + `load_test_data`, the shared 7-metric scorer used by all three eval scripts; (b) a legacy post-hoc reweighting experiment (`run_experiment`) that `PIPELINE.md` explicitly says is **not** the Tab.5 source. |
| `eval_bl_models.py`, `eval_et_models.py`, `eval_stdy_models.py` | Idempotent collectors → `data/{bl,et,st,dy}/`. |
| `set_num_cl.py` | Regex-patches `num-supernodes` in a dataset's `pyproject.toml`. |

### Per-dataset Flower apps (tracked)

8 dirs: `{adult,cifar,imdb,celeba}` × `{iid, noniid}`. Each has
`client_app.py` / `server_app.py` / `task.py` / `pyproject.toml`.
`task.py` owns data loading + partitioning + model architecture + checkpoint
paths; `server_app.py` owns strategy selection (FedAvg / FedProx / WeightedFedAvg)
and the `_with_agg_save` wrapper.

`task.py` files are **near-duplicates** across the 8 apps (`set_seed`,
`_resolve_save_dir`, `save_client`, `save_global_model`, `get_num_clients`,
`train_with_proximal` are byte-or-near-identical); `robustness.py:79` even
relies on that (`"set_seed is byte-identical across every task.py"`).

### Paper builders (gitignored, live)

`compute_score_diff.py` (Tab.4), `compute_score_fluct.py` (Tab.3),
`build_tab5.py` (Tab.5), `_splice_*.py` (splice into `MAIN.tex`),
`_run_figs.py` (execs `article_figs.ipynb`'s cells), `short_figs.py` (FedKDD),
`sanity_gate.py` (loss-sign gate — the documented prerequisite of all of the above).

---

## 3. What's dead

| Item | Why it's dead |
|---|---|
| `score_metrics.py:333` `is_loss_metric()` | `return name.lower() == "loss" and False` — always `False`. Has no callers. |
| `gtg_shap.py:160-176` (the convergence-error branch) | Unreachable. `converge_min` defaults to `max(30, K)=30` and `max_number` is capped at `max_permutations=30`, so `_check_convergence` returns `False` for `index<30` and `True` at `index==30`. GTG always runs **exactly 30 permutations**; `convergence_criteria` / `last_k` / `convergence_denom_floor` never affect anything. |
| `<ds>/task.py::set_num_clients()` (e.g. `adultnoniid/task.py:278`) | Uses `toml.dump()`, which `set_num_cl.py:18-21` documents as *breaking* Flower's config reader. No callers. Live grenade. |
| `<ds>/task.py` module-level `config = toml.load(toml_file)` (line 276 in adult) | Assigned, never read. |
| `weighted_utils.fedavg_aggregate` (as used by `weighted_strategy`) | Imported at `weighted_strategy.py:45` but never called — the class has its own identical `_aggregate_fedavg` (line 208). Still genuinely used by `inloop_scoring.py:38`. |
| `WeightedFedAvg.adopt_reweighted_or_fallback` (line 223) | The old "do-no-harm guard", superseded by the β/cap shrinkage. Its docstring points at `build_tab4_final.py`, itself superseded by `build_tab5.py`. |
| `merge_celeba.py` | Reads `data/celeba_to_distribute/` — **the directory does not exist**. The documented celeba merge step cannot run as-is. |
| `data/README.md` references | `weighting_effect.csv` and `collect_data.sh` are both documented in `data/README.md` and both absent (`collect_data.sh` is in `_archive/`). |
| `build_tab4.py`, `build_tab4_final.py`, `tab4_final.tex` | Pre-2026-07-10 numbering; replaced by `build_tab5.py` / `tab5_new.tex`. `PIPELINE.md` says so explicitly. |
| `eval.ps1` `gtg_acc` / `*_loss` branches | Point at `accuracy.py` / `loss.py`, archived to `_archive/stale_drivers/`. |

## 4. What's duplicated

| Duplication | Detail |
|---|---|
| **`a40_snapshot/` ≈ repo root** | A whole copy of the compute box. 15 root `.py` files are **byte-identical** to root (`gtg_shap`, `inloop_scoring`, `pgd_attack`, `fairness_metric`, `fast_infer`, `robustness_metric`, `eval_{bl,et,stdy}_models`, `set_num_cl`, `dashboard`, `cw_attack`, `weighting_effect`, `eval_{acc,resacc}_one`, `eval_celeba_res_e01`); 9 are **older forks** (`robustness`, `cont_evals`, `score_metrics`, `attack_metric`, `privacy_metric`, `reweight_eval`, `weighted_strategy`, `weighted_utils`, `et_training`). Plus 13 snapshot-only scripts and ~90 raw per-seed CelebA result CSVs. |
| **`_archive/stale_drivers/{loss,accuracy}.py`** vs `a40_snapshot/{loss,accuracy}.py` | Same stale drivers, two copies. These are the ones with their own (wrong) sign flips. |
| **8 × `task.py`** | `set_seed`, `_resolve_save_dir`, `save_client`, `save_global_model`, `get_num_clients`, `set_num_clients`, `train_with_proximal` repeated verbatim in all 8 apps. |
| **6 × `server_app.py`** | `on_fit_config`, `on_evaluate`, `_with_agg_save`, `server_fn` strategy-selection block repeated per app with only the import path changing. |
| **FedAvg aggregation, 3 implementations** | `weighted_utils.fedavg_aggregate`, `WeightedFedAvg._aggregate_fedavg`, `robustness_metric.aggregate_models` (this one un-weighted). |
| **Two loss/accuracy scorers** | `cont_evals._accuracy_score` / `_loss_score` (post-hoc) vs `score_metrics.build_acc_metric` / `build_loss_metric` (in-loop). Deliberate (different cost budgets) but the loss-sign conventions differ, which is exactly the bug class this repo keeps hitting. |
| **`data/combo_bak_*` × 5 + `data/_*` × 6** | Point-in-time backups of `data/combo`, 16 CSVs each. |
| **`demo/data/*.csv`** | Copies of all 16 `data/combo` CSVs. |

## 5. What's experiment scratch

| Group | Count | Notes |
|---|---|---|
| Box launcher / watchdog `.sh` | 30 at root | `a40_*`, `crysys_*`, `celeba_k20_*`, `cap_*`, `ablate_beta`, `*_watchdog`, `*fill*`, `res_check`, `healthcheck`, `full_stdy_rerun`, `imdb_rerun`, `st_dy_guarded_run`. Per-host, per-incident. `.gitignore` already excludes `*.sh` except `run_et_experiments.sh`. |
| `jobs_*.txt` | 11 | Work queues consumed by those `.sh` files. |
| One-off analysis scripts | 8 | `eval_acc_one`, `eval_resacc_one`, `eval_cap_ab`, `eval_celeba_res_e01`, `eps_calibrate`, `prewarm_celeba`, `prewarm_imdb`, `merge_imdb_rerun` (a dated one-time fold-in). |
| Generated tex bodies | 4 | `tab4_final.tex`, `tab5_new.tex`, `tab_scorediff_new.tex`, `tab_scorefluct_new.tex` — outputs of the builders. |
| Status/handoff docs | 5 | `ASSESSMENT.md`, `MERGE_NOTES.md`, `STATUS_AUTONOMOUS.md`, `TAB4_PIPELINE.md`, `TAB4_SOLUTIONS.md`. |
| Tarballs | 11 in `_pull/`, 3 in `data/`, 2 in `rescue_20260706/`, 2 in `a40_snapshot/` | Box pulls / rescue archives. |
| `logs/` | 26 | Box run logs. |
| `vpn.txt` | 20 bytes | Local credential-ish note. Already gitignored. |

## 6. Data artifacts (what must survive at all costs)

| Path | Rows / files | Status |
|---|---|---|
| `data/combo/results_<ds>_{gtg,l1o}.csv` | 16 files × 1200 rows | **The paper's numbers.** Tracked. adult/cifar/imdb complete; **celeba has real holes** (see audit A9). |
| `data/{bl,st,dy,et}/results_<ds>_<fam>.csv` | 12 files | Tab.5 source. Tracked. |
| `data/tab4_summary.csv` | 1 | Tracked. |
| `420/`, `420_et_{fair,adv,dp}/` | 12 100 + 30 files | The model checkpoints. `420/adultnoniid` and `420/adult` are complete (5 seeds × K∈{4,20}); `420/celebanoniid/20/42` has 92 files vs the expected 211, and `420/celebanoniid/20/107` is missing. |
| `figs/` | 90 PNG | Regenerable via `_run_figs.py`; the finals live in `MAIN.tex`. |

---

# Proposed target structure

Nothing is deleted. `archive/` becomes the single quarantine, replacing today's
four overlapping ones (`_archive/`, `a40_snapshot/`, `_pull/`, `rescue_20260706/`).

```
FLRobustness/
├── README.md  LICENSE  requirements.txt  PIPELINE.md  REPO_MAP.md
├── adult/ adultnoniid/ celeba/ celebanoniid/          # 8 Flower apps, unchanged
│   cifar/ cifarnoniid/ imdb/ imdbnoniid/
├── <root modules>.py                                  # the 15 tracked core modules
├── robustness.py                                      # the post-hoc driver
├── eval_{bl,et,stdy}_models.py  reweight_eval.py      # collectors
├── run_experiments.ps1  run_et_experiments.{sh,ps1}   # launchers
│   set_environment.ps1  eval.ps1
├── paper/                    # NEW — the gitignored assembly layer, currently
│   ├── sanity_gate.py        #       loose at root
│   ├── compute_score_{diff,fluct}.py  build_tab5.py
│   ├── _splice_*.py  _run_figs.py  short_figs.py
│   ├── article_figs.ipynb
│   └── tab*.tex              # generated bodies
├── tests/                    # NEW — test_*.py + smoke_lossflag.py
├── data/                     # combo/ bl/ st/ dy/ et/ + README + merge helpers
├── 420*/                     # checkpoints, untouched
├── figs/  logs/  demo/
└── archive/                  # NEW single quarantine
    ├── boxes/a40_snapshot/          # was ./a40_snapshot
    ├── boxes/scripts/               # 30 root *.sh + 11 jobs_*.txt
    ├── oneoffs/                     # 8 one-off analysis scripts
    ├── superseded/                  # build_tab4*.py, tab4_final.tex, merge_imdb_rerun.py
    ├── pulls/                       # was ./_pull, ./rescue_20260706, data/*.tgz
    ├── data_backups/                # data/combo_bak_*, data/_*
    ├── notes/                       # 5 status .md files
    └── old_archive/                 # was ./_archive, verbatim
```

`.gitignore` shrinks to roughly: envs/caches, `420*/`, `archive/`, `figs/`,
`logs/`, `demo/`, `paper/`, `tests/`, `data/_*`, `*.tex`, `vpn.txt`.

---

# File classification

Legend: **K** = keep in place · **A** = archive (move, never delete) · **U** = uncertain, needs your call.

## Root Python — core library

| File | | Reason |
|---|---|---|
| `cont_evals.py` | K | The canonical evaluator; owns the loss-sign convention. |
| `robustness.py` | K | The only post-hoc scoring driver. |
| `gtg_shap.py` | K | GTG-Shapley implementation. |
| `score_metrics.py` | K | In-loop metric builders for ST/DY. |
| `inloop_scoring.py` | K | GTG score provider for `WeightedFedAvg`. |
| `weighted_strategy.py` | K | The ST/DY Flower strategy. |
| `weighted_utils.py` | K | Pure-math weight transform; unit-tested. |
| `robustness_metric.py` | K | `rel` metric + client-model loading. |
| `attack_metric.py` | K | `res` metric (PGD/CW) + `PGD_EVAL_PARAMS`. |
| `fairness_metric.py` | K | `fairDP`/`fairEO`. |
| `privacy_metric.py` | K | `priv` (MIA) + `MIAConfig`. |
| `pgd_attack.py` | K | PGD kernel used by eval and by ET-adv training. |
| `cw_attack.py` | K | C&W kernel; `*_adv_cw` methods depend on it. |
| `et_training.py` | K | ET local losses, imported by all `*noniid/task.py`. |
| `fast_infer.py` | K | Compiled forward pass for in-loop scoring. |
| `set_num_cl.py` | K | Required by both launchers. |
| `reweight_eval.py` | K | Hosts `eval_all_metrics`, on which 3 collectors depend. |
| `eval_bl_models.py` | K | Produces `data/bl/`. |
| `eval_et_models.py` | K | Produces `data/et/`. |
| `eval_stdy_models.py` | K | Produces `data/{st,dy}/`. |

## Root Python — paper assembly → `paper/`

| File | | Reason |
|---|---|---|
| `sanity_gate.py` | K→`paper/` | The mandatory pre-regeneration gate. |
| `compute_score_diff.py` | K→`paper/` | Tab.4 body. |
| `compute_score_fluct.py` | K→`paper/` | Tab.3 body. |
| `build_tab5.py` | K→`paper/` | Tab.5 body (current). |
| `_splice_scorediff.py`, `_splice_scorefluct.py`, `_splice_tab5.py` | K→`paper/` | Splice into `MAIN.tex`. |
| `_run_figs.py`, `article_figs.ipynb` | K→`paper/` | Figure regeneration. |
| `short_figs.py` | U | Only feeds `FedKDD.tex` (the short paper). Keep or archive with FedKDD? |
| `tab_scorediff_new.tex`, `tab_scorefluct_new.tex`, `tab5_new.tex` | K→`paper/` | Current generated bodies. |
| `build_tab4.py`, `build_tab4_final.py`, `tab4_final.tex` | A | Superseded by `build_tab5.py` per `PIPELINE.md`. |

## Root Python — tests → `tests/`

| File | | Reason |
|---|---|---|
| `test_weighted_strategy.py` | K→`tests/` | The only real unit tests (β/cap transform, ST/DY state machine). |
| `test_inloop_scoring.py` | K→`tests/` | TF-free GTG plumbing smoke test. |
| `smoke_lossflag.py` | K→`tests/` | Guards the loss-direction convention — the repo's most-repeated bug. |

## Root Python — one-offs

| File | | Reason |
|---|---|---|
| `eval_acc_one.py`, `eval_resacc_one.py` | A | Single-model print helpers for the β/cap ablation. |
| `eval_cap_ab.py` | A | Weight-cap A/B, completed. |
| `eval_celeba_res_e01.py` | A | One-time celeba re-eval at ε=0.01; the result is already in the data. |
| `eps_calibrate.py` | A | One-time IMDB ε calibration; the answer is baked into `PGD_EVAL_PARAMS`. |
| `merge_imdb_rerun.py` | A | Dated one-time fold-in (2026-07-10). |
| `prewarm_celeba.py`, `prewarm_imdb.py` | U | Cache prewarmers. Commit `ab0b35f` already "dropped standalone cache-prewarm helpers" — these two survived. Archive, or were they deliberately kept? |
| `merge_celeba.py` | U | `PIPELINE.md` calls it a pipeline step, but its input dir `data/celeba_to_distribute/` no longer exists. Keep as documentation-of-record, or archive? |
| `dashboard.py` | U | Streamlit explorer. Superseded by `demo/`? |
| `weighting_effect.py` | U | Produces `data/weighting_effect.csv`, which `data/README.md` documents but which is absent. |

## Flower apps

| Path | | Reason |
|---|---|---|
| `adult/`, `adultnoniid/`, `celeba/`, `celebanoniid/`, `imdb/`, `imdbnoniid/` | K | The 3 paper datasets × 2 partitions. |
| `cifar/`, `cifarnoniid/` | U | Not in the paper (`sanity_gate.py:28` — "reported but not gating"; `PIPELINE.md` — "cifar is not in the paper"). But `data/combo/results_cifar*.csv` is tracked and the demo ships cifar data. Keep for completeness, or archive? |

## Scripts / ops

| Path | | Reason |
|---|---|---|
| `run_experiments.ps1`, `run_et_experiments.{sh,ps1}`, `set_environment.ps1` | K | Documented launchers. |
| `eval.ps1` | U | Tracked and README-documented, but broken (audit A2). Fix in step 3, or archive it and document `robustness.py` directly? |
| 30 root `*.sh` (a40_*, crysys_*, celeba_k20_*, cap_*, watchdogs, fills) | A | Per-host, per-incident box scripts. |
| `jobs_*.txt` (11) | A | Work queues for the above. |

## Directories

| Path | | Reason |
|---|---|---|
| `data/combo/`, `data/{bl,st,dy,et}/`, `data/tab4_summary.csv`, `data/README.md` | K | The durable results. |
| `data/merge_collected.py`, `data/combine_celeba_sources.py` | K | Collectors referenced by `data/README.md`. |
| `data/combo_bak_*` (5), `data/_*` (6), `data/*.tgz` (3) | A | Point-in-time backups and staging. |
| `420/`, `420_et_*/` | K | Checkpoints — the only way to re-derive anything. |
| `_archive/` | A→`archive/old_archive/` | Already quarantined; just consolidate. |
| `a40_snapshot/` | A→`archive/boxes/` | Whole-box copy; 9 files are *older forks* of live modules, so leaving it at root invites importing the wrong one. |
| `_pull/`, `rescue_20260706/` | A→`archive/pulls/` | Tarball pulls. |
| `logs/` | A | Box run logs. |
| `figs/` | K | Regenerable, but cheap to keep and referenced by `MAIN.tex`. |
| `demo/` | U | `.gitignore` says "lives in its own repo" and `.github/workflows/demo-ci.yml` builds it. Is the copy here still authoritative, or archive it? |
| `.github/` | U | Only builds `demo/`. Follows whatever `demo/` does. |

## Docs

| File | | Reason |
|---|---|---|
| `README.md`, `LICENSE`, `PIPELINE.md`, `requirements.txt` | K | |
| `REPO_MAP.md` | K | This file. |
| `MAIN.tex` | K | The thesis/paper source. |
| `FedKDD.tex` | U | The short conference paper. Still live, or archive with `short_figs.py`? |
| `ASSESSMENT.md`, `MERGE_NOTES.md`, `STATUS_AUTONOMOUS.md`, `TAB4_PIPELINE.md`, `TAB4_SOLUTIONS.md` | A→`archive/notes/` | Dated internal status docs; `PIPELINE.md` supersedes the TAB4 pair. |
| `vpn.txt` | U | 20 bytes, credential-ish. Leave in place (already gitignored) or archive? |

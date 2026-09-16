# FLRobustness — data → paper pipeline

One page. If a step isn't listed here, it's not part of the pipeline (one-off
patch scripts live in `archive/` and must not be re-run).

## Sign convention (THE rule)

**Every `*_contribution` column in `data/combo/` is higher = better, for every
metric — including loss.** The negation for loss happens in exactly one place:
`cont_evals.py :: evaluate_round` (since 2026-07-08). `global_*_loss` stays raw
CE. No merge, collect, plot, or table script may flip signs — that caused the
June/July 2026 double-flip mess (raw files from different code generations had
opposite conventions; uniform data-level flips just moved the bug between
datasets).

`paper/sanity_gate.py` enforces the convention: for every dataset the acc~loss
Spearman of contributions must be positive under both LOO and GTG (the
advisor's baseline-inversion check). **Run it before regenerating anything.**

Caveat: raw per-seed files produced by pre-2026-07-08 code carry the inverse
loss sign (celeba/cifar new-generation files did). If you merge from an old
archive and the gate fails, negate that file's loss contributions once and
document it — do not add flips to pipeline scripts.

## Flow

```
GPU/CPU boxes: robustness.py --method {gtg,l1o}_<metric>   (THE only post-hoc driver; CPU eval only!)
               (loss.py / accuracy.py were stale duplicates with their own sign flips -> archive/old_archive/stale_drivers)
        └─> results_<ds>_<seed>_<method>_<metric>_fedavg.csv        (raw, canonical sign)
                └─> _fills_<date>/fold_in.py on branch fills-eps-fix-2026-09-15  (NaN-only fold-in: writes a combo cell ONLY where it is
                    NaN, never overwrites -> cannot move a published number; idempotent, backs up data/combo first)
                    (superseded: archive/oneoffs/merge_celeba.py - its data/celeba_to_distribute/ was the
                     multi-host staging map from when celeba arrived from several boxes; dead)
                        └─> data/combo/results_<ds>_{gtg,l1o}.csv   (THE durable artifact)

boxes: run_et_experiments / st-dy sweeps
        └─> data/_collect/<host>/  ─ merge_collected.py ─> data/{et,st,dy}/

reweight_eval.py ─> results_reweight_<ds>_<K>.csv   (post-hoc reweighting analysis ONLY —
                                                     Tab 5 ST/DY come from the in-loop
                                                     retrained models in data/{st,dy})
```

## Regeneration (in order)

| # | Command | Output |
|---|---------|--------|
| 1 | `python paper/sanity_gate.py` | gate — must PASS before 2-7 |
| 2 | `python paper/compute_score_diff.py` | `tab_scorediff_new.tex` (Tab 4 body) |
| 3 | `python paper/_splice_scorediff.py` | Tab 4 into MAIN.tex (dated .bak) |
| 4 | `python paper/compute_score_fluct.py` + `python paper/_splice_scorefluct.py` | Tab 3 into MAIN.tex |
| 5 | `python paper/_run_figs.py` | `data/combo/plots/*.png`, then copy MAIN.tex-referenced names to `figs/` |
| 6 | `python paper/build_tab5.py` | `tab5_new.tex` (Tab 5 / tab:util body, from `data/{bl,st,dy,et}`) |
| 7 | `python paper/_splice_tab5.py` | Tab 5 into MAIN.tex (dated .bak) |

(`archive/superseded/build_tab4*.py` are the pre-2026-07-10 Tab 5 builders — "tab4" in those
filenames is the OLD numbering; `build_tab5.py` replaced them after the box-era
`generate_thesis_table.py` was lost with the box. `archive/oneoffs/merge_imdb_rerun.py` was the
one-time fold-in of the 2026-07-10 IMDB rerun + celeba gtg_loss recompute.)

## Known limitation: GTG is not reproducible from the CLI alone

`gtg_*` numbers depend on the **order rounds were scored in**, not just on
`(dataset, seed, method)`.

`robustness.py` calls `set_seed(seed)` once per seed, outside the round loop,
while `GTGShapley` draws from the global numpy RNG (once at construction, then
30 shuffles per round). So round *r*'s permutations depend on how many rounds ran
before it in the same process. Three things change the stream:

- **resuming** an interrupted run (`_load_done_keys` skips finished rounds),
- **`FLR_ROUND_ORDER`**,
- **`--num_clients`** (whether K=20 consumes RNG before K=4).

Measured 2026-09-01 on `adultnoniid` / K=4 / seed 42 / `gtg_acc`, forward vs.
reversed round order: `global_gtg_acc` identical to the bit, but **21 of 40**
`gtg_acc_contribution` values moved, by up to 3.9e-3 on contributions spanning
0.000-0.066.

**Consequence:** the `gtg_*` columns in `data/combo/` cannot be re-derived
exactly without knowing how the original sweep was interrupted. LOO is
unaffected (no RNG). Treat GTG contributions as estimates with ~4e-3 of sampling
noise, which is what they are.

**Not fixed on purpose.** Seeding per `(seed, K, round)` would make GTG
reproducible going forward, but every `gtg_*` column already in `data/combo/` was
produced under the current stream, so the fix would make the shipped data
un-reproducible from the fixed code. Deferred until a `gtg_*` re-run is happening
anyway. See `AUDIT.md` A6.

## Known limitation: ADULT is re-derived from OpenML at eval time

`adult*/task.py::load_data` re-fetches OpenML 1590 and re-fits the
encoder/scaler/split on **every** call, and `requirements.txt` pins neither
`scikit-learn` nor `pandas`. CelebA caches partitions to `.npz` and IMDB caches
USE embeddings; ADULT caches nothing. A clean-room re-run of
`adultnoniid`/K=4/seed 42/`l1o_acc` reproduced all 40 rows but with max|diff|
7.6e-3. Pin those two packages, or cache the processed matrix, before claiming
bit-exact ADULT reproduction. See `AUDIT.md` A30.

## Gotchas that keep coming back

- Post-hoc evals run on CPU (`CUDA_VISIBLE_DEVICES=`); GPU eval after PGD hits
  an XLA device error and silently writes 0 rows (`res` columns of local
  reweight CSVs show 0.0 — that's this).
- `archive/data_backups/combo_bak_*` are point-in-time backups (preloss = pre 2026-06-29 flip,
  presign_20260706 = pre second flip, phase0_20260708 = pre celeba fix).
- cifar combo GTG files still carry the inverse loss sign (cifar is not in the
  paper; gate reports it as info). Negate once if cifar ever returns.

## Layout note (2026-09-01 cleanup)

All paper-assembly scripts moved from the repo root into `paper/`. They use
**cwd-relative** `data/` paths on purpose, so keep running them **from the repo
root** (`python paper/sanity_gate.py`), not from inside `paper/`. The generated
table bodies (`tab*.tex`) are still written to, and read from, the repo root.

Retired material — box snapshots, per-host `.sh` scripts, one-off analyses,
superseded builders, tarball pulls, data backups, old status notes — now lives
under a single `archive/`. Nothing there is part of the pipeline; nothing there
may be re-run. See `REPO_MAP.md`.

## Coverage gate

`python paper/sanity_gate.py` now also reports, per (dataset, metric, K), how
many of the 5 seeds actually back each final-round cell. Every builder
aggregates with a plain pandas mean, which drops NaN silently, so a thin cell
renders identically to a full one. Warn-only by default; `--strict` makes an
under-covered cell fail the gate.

As of 2026-09-01 there are **17 under-covered cells**, the worst being
`results_celeba_l1o.csv` K=4, where `acc` / `loss` / `fair` / `fair_eo` are each
backed by **1 of 5 seeds** yet reported as mean +/- std.

## Direction B: group-gap utilities (branch `direction-b`)

`u = 1 − |M(A) − M(B)|` for `M ∈ {acc, loss, rel, res}` (priv parked until reviewer
feedback), scored by the same LOO/GTG code. Methods:
`{l1o,gtg}_gap_{acc,loss,rel,res}_{dp,eop}`, where `dp` uses every test sample of a
group and `eop` only its true positives (y = 1). Logic lives in `Code/gap_metrics.py`
(check: `python Code/test_gap_metrics.py`), routed through `cont_evals.evaluate_round`.

- **Names end in `_dp` / `_eop` on purpose.** `evaluate_round` negates contributions,
  and GTG minimises, for any name ending in `_loss`; gap utilities are higher-is-better.
- **Loss** enters per group as the paper's loss utility `e^(−mean CE)`; a gap of raw CE
  is unbounded.
- **Group A** is `--sensitive_attr` as `column=value`, by default female on both datasets
  (ADULT `sex=Female`, CelebA `Male=0`). The paper's CelebA runs coded s = Male; every
  fairness and gap score is symmetric in A and B, so no number depends on the coding. A
  non-default attribute writes `results_<ds>_<seed>_<method>_s-<attr>_fedavg.csv`, never
  the default file, and `fold_in.py` skips those names.
- **CelebA attributes other than `Male`** come from a sidecar next to the partition cache
  (`*.s-<attr>.npz`), rebuilt from the HF label columns and checked against the cached
  labels before it is written. The cache itself only stores `Male` (before 2026-09-15 a
  cache hit returned `Male` for any requested attribute). Building a sidecar needs the HF
  dataset: point `HF_HOME` / `HF_DATASETS_CACHE` at the repo-root `.hf_cache`; offline works.
- **Coalitions are averaged the way the server averaged that round** (AUDIT A36): FedAvg's
  per-client coefficients are recovered from the checkpoints
  (`weighted_utils.recover_fedavg_coeffs`) and renormalised per coalition. Every row records
  `coalition_avg`: `size` (weighted), `equal` (equal partitions, plain mean, as in the
  shipped runs) or `fallback` (coefficients not recoverable, plain mean, and LOO then scores
  V(All) as that plain mean too).
- **Two GTG columns** (AUDIT A37): `<method>_contribution` is GTG exactly as the paper ran it
  (truncation, then only the best coalition keeps a value), while `<method>_shapley` is the
  Shapley value of the same coalition game without those steps — exact for K ≤ 6, otherwise
  permutation sampling with no truncation and every client first equally often
  (`gtg_shap.shapley_values`). Direction B reports `_shapley`; `_contribution` stays for
  comparison with the paper.
- **Gap rows** also carry `group_a`, `n_group_a`, `n_group_b`, each group's value on the
  global model (`global_<method>_m_a` / `_m_b`) and each client's attribution of those two
  values (`<method>_shapley_m_a/_m_b` for GTG, `<method>_contribution_m_a/_m_b` for LOO).
  A narrower gap means little until you know whether the worse group rose or the better one
  fell, and these cost no extra model evaluations.
- **Resilience on a group with no clean-correct sample is NaN**, not the 0.0 that PGD
  returns: inside a gap that would read as a perfectly fair coalition.
- **Reproducibility.** Gap methods reseed per (seed, K, round) (AUDIT A6). Other methods are
  not reseeded, and the cached empty coalition (A33) changes how many random draws `rel`
  consumes, so `gtg_rob` and in-loop `rel` scores no longer reproduce the shipped ones bit
  for bit — reproduce those from `master`.
- **Run:** job lines `<ds> <partition> <K> <seed> <method> [column=value]` in
  `_direction_b/jobs_*.txt`, then from the repo root
  `OUT_DIR=../_direction_b bash _fills_20260914/run_fills.sh _direction_b/jobs_X.txt`.
  Fold the fixed scorer's output in with
  `cd _direction_b && FLR_COMBO=../Code/data/combo_v2 python ../_fills_20260914/fold_in.py --apply`:
  `FLR_COMBO` must be explicit, files/rows/columns are created as runs arrive, cells are only
  ever filled, and a kept cell whose incoming value differs is reported. Without `FLR_COMBO`
  the target is the shipped `data/combo`, which refuses fixed-scorer output (and vice versa).
- **GTG sanity check:** `seed=42 python Code/gtg_check.py --dataset adultnoniid --K 4 --seed 42`
  compares an independent exact Shapley enumeration with `_shapley` and `_contribution`,
  checks the recovered coefficients against the clients' real training sizes, runs
  `evaluate_round` end to end, and scores a duplicated and a planted group-skewed client.
  At K > 6 pass `--ref_perms N` to compare against a long sampling reference instead.

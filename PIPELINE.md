# FLRobustness — data → paper pipeline

One page. If a step isn't listed here, it's not part of the pipeline (one-off
patch scripts live in `_archive/` and must not be re-run).

## Sign convention (THE rule)

**Every `*_contribution` column in `data/combo/` is higher = better, for every
metric — including loss.** The negation for loss happens in exactly one place:
`cont_evals.py :: evaluate_round` (since 2026-07-08). `global_*_loss` stays raw
CE. No merge, collect, plot, or table script may flip signs — that caused the
June/July 2026 double-flip mess (raw files from different code generations had
opposite conventions; uniform data-level flips just moved the bug between
datasets).

`sanity_gate.py` enforces the convention: for every dataset the acc~loss
Spearman of contributions must be positive under both LOO and GTG (the
advisor's baseline-inversion check). **Run it before regenerating anything.**

Caveat: raw per-seed files produced by pre-2026-07-08 code carry the inverse
loss sign (celeba/cifar new-generation files did). If you merge from an old
archive and the gate fails, negate that file's loss contributions once and
document it — do not add flips to pipeline scripts.

## Flow

```
GPU/CPU boxes: robustness.py --method {gtg,l1o}_<metric>   (THE only post-hoc driver; CPU eval only!)
               (loss.py / accuracy.py were stale duplicates with their own sign flips -> _archive/stale_drivers)
        └─> results_<ds>_<seed>_<method>_<metric>_fedavg.csv        (raw, canonical sign)
                └─> merge_celeba.py (celeba)  /  column merges (adult, imdb)
                        └─> data/combo/results_<ds>_{gtg,l1o}.csv   (THE durable artifact)

boxes: run_et_experiments / st-dy sweeps
        └─> data/_collect/<host>/  ─ merge_collected.py ─> data/{et,st,dy}/

reweight_eval.py ─> results_reweight_<ds>_<K>.csv                   (Tab 5 inputs)
```

## Regeneration (in order)

| # | Command | Output |
|---|---------|--------|
| 1 | `python sanity_gate.py` | gate — must PASS before 2-7 |
| 2 | `python compute_score_diff.py` | `tab_scorediff_new.tex` (Tab 4 body) |
| 3 | `python _splice_scorediff.py` | Tab 4 into MAIN.tex (dated .bak) |
| 4 | `python compute_score_fluct.py` + `python _splice_scorefluct.py` | Tab 3 into MAIN.tex |
| 5 | `python _run_figs.py` | `data/combo/plots/*.png`, then copy MAIN.tex-referenced names to `figs/` |
| 6 | `python build_tab5.py` | `tab5_new.tex` (Tab 5 / tab:util body, from `data/{bl,st,dy,et}`) |
| 7 | `python _splice_tab5.py` | Tab 5 into MAIN.tex (dated .bak) |

(`build_tab4*.py` are the pre-2026-07-10 Tab 5 builders — "tab4" in those
filenames is the OLD numbering; `build_tab5.py` replaced them after the box-era
`generate_thesis_table.py` was lost with the box. `merge_imdb_rerun.py` was the
one-time fold-in of the 2026-07-10 IMDB rerun + celeba gtg_loss recompute.)

## Gotchas that keep coming back

- Post-hoc evals run on CPU (`CUDA_VISIBLE_DEVICES=`); GPU eval after PGD hits
  an XLA device error and silently writes 0 rows (`res` columns of local
  reweight CSVs show 0.0 — that's this).
- `data/combo_bak_*` are point-in-time backups (preloss = pre 2026-06-29 flip,
  presign_20260706 = pre second flip, phase0_20260708 = pre celeba fix).
- cifar combo GTG files still carry the inverse loss sign (cifar is not in the
  paper; gate reports it as info). Negate once if cifar ever returns.

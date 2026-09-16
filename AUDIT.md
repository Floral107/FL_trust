# AUDIT.md — correctness review of what remains after the cleanup

Scope: everything still at the repo root + `paper/` + the 8 Flower apps, after
the `cleanup/repo-structure` reorganisation. `archive/` is not audited.

Ordered by **how much damage the failure does before you notice it**. Every
entry names the concrete failure, not the smell.

## Status as of 2026-09-14

| | Findings | State |
|---|---|---|
| Fixed | A1, A2, A4, A5, A7, A10, A13, A15, A21, A24, A26, A27, A28, A29 | Commits `d3e8d2d`, `f3d95dd`. Verified number-neutral: an `l1o_acc` re-run is bit-identical to the pre-fix output. |
| Fixed (dead code removed) | A25 | `set_num_clients()` deleted from all 8 `task.py`; recoverable from git history. |
| Documented, deliberately not fixed | A6, A30 | Both would change published numbers. Written into `PIPELINE.md` as known limitations. |
| Fixed 2026-09-14 on branch `fills-eps-fix-2026-09-15` only (master unchanged) | A12, A31 | A12: ET params written back to all 3 `pyproject.toml` (lambda 2.0; celeba ratio 1.0 / steps 10). A31: adult PGD eval eps 0.3 -> 0.03 in both param tables. **The `.tex` files are handed in and stay as submitted**, so the paper's stated 0.3 vs the measured 0.03 is a documented discrepancy, not a fix. Backups `*.bak_a12`, `*.bak_eps`. |
| Open — needs your decision | A9, A11, A32 | A9: combo `priv` protocol (5, 5e-4, 0.2/0.7) vs the documented (2, 1e-3, 0.5/0.5). A11: now quantified — the celeba in-loop proxy scores in the saturated regime yet ranks effectively; disclose, or align to 0.01 and re-run all celeba ST/DY. A32: presentational — print `\textemdash` for Phi built on a degenerate loss reference (K=4 adult IID currently prints six `+-0.00` values off one seed). |
| Open — hygiene | A3, A8, A14, A16, A17, A18, A19, A20, A22, A23, A33 | No decision needed; none is urgent. |
| Fixed 2026-09-15 on branch `direction-b` only (not merged) | A34 | CelebA attribute sidecar. Latent: no shipped number was affected. |
| Documented 2026-09-15 | A35, A30 (second data point) | A35: ADULT codes s = female, CelebA s = male; every score is symmetric, and direction-b names group A explicitly. |
| Fixed 2026-09-16 on branch `direction-b` only (not merged) | A33, A36, A37 | Coalitions use FedAvg's own coefficients, recovered from the checkpoints. `<method>_shapley` carries a clean Shapley value (exact for K ≤ 6, balanced sampling above) beside the paper's `<method>_contribution`. The empty coalition is cached. Shipped columns stay reproducible from `master`. |

**Branch note (2026-09-15):** every fix, fill and tool dated 2026-09-14/15 in this file is committed on
branch `fills-eps-fix-2026-09-15` (`cfdbbe9`, pushed to `github` and `origin`, not merged). A `master`
checkout has none of them: adult eps is still 0.3 there, and `data/combo` is the as-submitted state.

**Coverage, 2026-09-14:** the A21 check was combo-only. Extended to the Tab. 5
families (`coverage_families()` in `paper/sanity_gate.py`), which surfaced **24**
further under-covered cells in `data/{st,dy}` that nothing had ever counted — adult
`res` at K=4 missing seed 2025 across 4 weight-metrics, and 4 whole celeba rows.
Unlike the combo holes these are not re-scorable: no `420_st_*` / `420_dy_*`
checkpoints exist locally and the boxes that held them are gone.

**Fills, 2026-09-15:** combo coverage holes **17 -> 4**. Re-scored from on-disk checkpoints
(`_fills_20260914/`) and folded with `fold_in.py`: NaN-only, 2200 cells written, never overwrote an
existing value, `data/combo` backed up before each apply.
- celeba K4 IID `l1o` acc/fair/fair_eo/loss: 1 -> 5 seeds. The Tab. 4 CelebA-4-I LOO row flips sign on
  fairDP, fairEO and res, and priv goes +0.80+-0.00 -> 0.00+-0.63 (the handed-in article stays as submitted).
- celebanoniid K4 seed 42 `l1o` + `gtg` acc/fair/fair_eo/priv: 4 -> 5 seeds.
- adult K20 seed 9928 `gtg_adv` round 10: 4 -> 5 seeds, re-run at the corrected eps=0.03 (A31); rounds
  1-9 reproduce the shipped values to <=2.2e-3. Its round-10 contributions are all exactly 0 (A32), so
  it adds coverage and L2 but no Phi.

The remaining 4 are all `rob` (randomized smoothing): celeba K20 `l1o` s42/s123 and `gtg` s42/s9928,
celebanoniid K4 `l1o` + `gtg` s42. Memory-bound on the 15.7 GB laptop: `calculate_image_robustness`
builds a ~2.5 GB `(chunk_size=10, 5000, 64, 64, 3)` tensor with several live copies and segfaulted
(rc=139) with other jobs resident. Lowering `chunk_size` would change the Monte Carlo draw pattern vs
the shipped seeds, so it was not done. **Not run: skipped by decision on 2026-09-15** (cost below); these 4 cells stay documented at 3/5 and 4/5 seeds.

**Measured cost of those 4 holes (2026-09-15, clean run, zero standby entries):** one CelebA `rob` eval takes
**924 s** at `OMP_NUM_THREADS=4` and peaks at **12.8 GB** working set (rel 0.8985, inside the shipped seeds'
0.883-0.921). With the A33 eval counts, serial and alone: celebanoniid K4 `l1o_rob` ~13 h; celebanoniid K4
`gtg_rob` 3.4-4.9 days; celeba K20 `l1o_rob` 2.2 days per seed (x2); celeba K20 `gtg_rob` 3.4-64 days per seed
(x2). All four: ~15 days at best, months at worst, for cells that do not change the handed-in article. More
threads would shorten it by an unmeasured factor.

`paper/sanity_gate.py --strict` now fails on missing seeds only; zero-variance cells (A32) are reported
but cannot be fixed by re-running, so they no longer block it.

**Submitted state is preserved in git.** `git show master:Code/data/combo/<file>` (commit `0e8075c`) is the data the handed-in
article was built from: all 16 combo files on master equal the pre-fill backup
`Code/data/combo_bak_20260914_191605`, and regenerating Tab. 3/4 from it matches MAIN.tex byte for byte
(36/36 table rows). The fills are committed on branch `fills-eps-fix-2026-09-15` (commit `cfdbbe9`, pushed to `github` and `origin`, not merged), in exactly 4 files (`results_adult_gtg`,
`results_celeba_l1o`, `results_celebanoniid_gtg`, `results_celebanoniid_l1o`), so `git diff master fills-eps-fix-2026-09-15 -- Code/data/combo`
is the complete record of what changed. Keep master as the as-submitted state.

**A21's first catch:** the new coverage check found 17 under-covered cells. The
worst is `results_celeba_l1o.csv` at K=4, where `acc`, `loss`, `fair` and
`fair_eo` are each backed by **1 of 5 seeds** while being reported as mean ± std.
See the coverage section below.

---

## P0 — silently corrupts numbers that end up in the paper

### A1. A failed metric becomes the value `0.0`, indistinguishable from a real score
`robustness_metric.py:66-68` · `attack_metric.py:62-64` · `attack_metric.py:372-374`

```python
except Exception as e:
    print(f"[ERROR] PGD robustness failed for {dataset_name}: {e}")
    return 0.0
```

`rel`, `res` and `cw` all return `0.0` on *any* internal failure — OOM, an XLA
device error, a missing app dir, a dtype mismatch. That `0.0` is returned to
`cont_evals.l1o_evals` / `gtg_evals` as a legitimate utility.

**Failure:** in LOO, `contribution = global_metric − 0.0 = global_metric`, i.e. the
failing client gets the single largest positive contribution in the round — a
large, plausible, completely fabricated number. In GTG the poisoned coalition
utility shifts every player's marginal contribution. Nothing in the row marks it
invalid, and `robustness.py` exits 0.

**Fix shape:** return `float("nan")`, or re-raise and let A4 handle it. `nan`
propagates visibly and the existing `pd.to_numeric(..., errors="coerce")` guards
downstream already tolerate it.

### A2. One failing coalition silently rewrites every client's Shapley value
`cont_evals.py:316-318`

```python
except Exception as e:
    print(f"Error in subset {subset} utility ({dataset}): {e}")
    return 0.0
```

Same mechanism as A1, but inside the Shapley permutation loop, where a single
bad `u(S)` enters ~30 marginal-contribution differences.

### A3. A failing client is dropped from the round instead of failing the round
`cont_evals.py:286-287`

The per-client `try` in `l1o_evals` prints and continues, so `contributions` comes
back with K−1 entries and `robustness.py:281-292` writes K−1 rows for that round.

**Failure:** the round is silently narrower than every other round. `paper/sanity_gate.py:48`
(`if not Ms or len({len(x) for x in Ms}) != 1`) then throws away the *entire*
(dataset, K) cell as "degenerate" rather than reporting the gap — so a partial
failure is laundered into a missing table cell.

### A4. A totally failed sweep exits 0 with a 0-row CSV
`robustness.py:299-300`

The whole per-round body — model load, metric, LOO/GTG, CSV flush — is inside one
`except Exception as e: print(...)`. A systematic failure (wrong `--partition`,
GPU-after-PGD XLA error, missing app dir) prints one line per round and the
process still exits 0.

**Failure:** this is the documented CPU-only-eval trap. A multi-hour job "finishes"
having written nothing; the only way to notice is to count rows afterwards. Add a
non-zero exit when zero rows were written, and let unexpected exception types
propagate.

### A5. Adult reliability ignores its own input and re-loads the *IID* dataset
`robustness_metric.py:60-62`

```python
from adult.adult.task import load_data as load_adult_data
_, _, x_test, _, meta = load_adult_data(partition_id=0, num_partitions=1, ...)
```

The `x_test` argument passed in by `cont_evals` is discarded and replaced.

**Failure, three ways:**
1. `rel` for **`adultnoniid`** is computed on data loaded through the **IID** app.
2. `robustness.py --eval_subset N` (`robustness.py:249-268`) is silently ignored for
   adult `rel`, so adult rel rows are computed on a different test set size than
   the adult rows for every other metric in the same combo file.
3. A host that deploys only `adultnoniid/` raises `ModuleNotFoundError` here —
   which A1 converts into `rel = 0.0`.

---

### A31. ADULT PGD eval eps is 10x the value that produced every shipped number
`attack_metric.py:19` · `score_metrics.py:199` · MAIN.tex:345

Both param tables said `"adult": (0.3, 0.007, 40)`. Every shipped ADULT `res`/`adv`
number — `data/combo`, `bl`, `st`, `dy`, `et` — was in fact produced at **eps=0.03**,
the generic fallback in `calculate_pgd_score:40`. The original runs predate the table
and fell through to that default; the table was added later "to document what the
article reports" and carried a decimal typo. Git cannot show this: history begins at
the release snapshot, after the experiments ran.

**Measured (2026-09-14)**, adult K20 global model, shipped vs recomputed:

```
  seed  round | shipped    eps=0.03     diff
  9928      1 | 0.801138   0.801235   9.7e-05
    42      5 | 0.890872   0.891030   1.6e-04
   123     10 | 0.915212   0.916736   1.5e-03
   107      2 | 0.891408   0.894052   2.6e-03
```

All inside the A30 sklearn-drift band. At eps=0.3 the same model gives res 0.0113
(alpha=0.007) or 0.000121 (alpha=eps/4) — the dimension saturates to zero.

**Failure:** shipped numbers are fine; the hazard is forward-looking. Any adult re-run
(combo, or reweight_eval for bl/st/dy/et, which resolves eps from the same table at
`reweight_eval.py:220`) silently produces ~0.000 and disagrees with published data by
two orders of magnitude. The paper also stated a parameter that was never used.

**CODE fixed 2026-09-14:** both tables -> `(0.03, 0.007, 40)` / `(0.03, 0.007, 7)`, so a
re-run now reproduces the shipped numbers instead of ~0.000.

**PAPER deliberately NOT changed.** `MAIN.tex` / `LaTeX/TIST/MAIN.tex` / `LaTeX/UAI/main.tex`
are handed-in articles and stay as submitted (user decision, 2026-09-14). They therefore
state an ADULT eval eps of 0.3 while every number in them was measured at 0.03. Known,
documented discrepancy — same category as A6/A30. If a reviewer asks, the reproduction
table above is the evidence. Any FUTURE paper version should say 0.03.

MAIN.tex:866 was correct as written regardless — the ET *training* eps for adult genuinely
is 0.3 (`adultnoniid/pyproject.toml: et-adv-eps = 0.3`), i.e. it trains at 10x the eval
radius. Confirmed the June ST/DY sweeps also ran the in-loop proxy at 0.03: adult
res-weighted ST/DY beat BL in every seed (K20 BL 0.55-0.75 -> ST 0.79-0.92, DY 0.79-0.91),
impossible with a degenerate eps=0.3 proxy.

---

### A32. GTG truncation makes a fully-covered cell carry no rank information
`gtg_shap.py` (truncation eps1=0.001/eps2=0.05/eps3=0.002) · `paper/compute_score_diff.py:57`

GTG-Shapley returns **exact 0.0** for every client when coalition utilities move less
than the truncation thresholds. The column is then present and non-NaN — it passes the
A21 coverage check at 5/5 — but `compute_score_diff` skips it, because Spearman needs
`np.std(m) > 1e-12`. Coverage is not usability.

Phi is Spearman(metric, `loss`), so a degenerate **loss** column kills every Phi in the
row, whatever the metric columns do. Measured on `results_adult_gtg.csv` (2026-09-14):

```
  adult IID GTG | loss seeds with variance | printed
  K=4           | 1 of 5                   | six Phi values as "X +- 0.00"
  K=20          | 0 of 5                   | six "\textemdash"
```

**Failure:** the same underlying condition renders two different ways. K=20 is honest
(undefined). K=4 prints `-0.20+-0.00`, `0.40+-0.00`, `1.00+-0.00` etc. — six confident
numbers with zero error bars, each from the one seed that escaped truncation. The
`priv` cell reads as perfect, perfectly stable rank agreement; it is a single draw.

**Not fixable by re-running.** Unlike the A21 holes (missing runs -> fills repair them),
this is GTG behaving as designed on a dataset whose utilities barely move. A re-run
reproduces the same zeros. The fix is presentational: emit `\textemdash` when the loss
reference is degenerate in most seeds, i.e. treat K=4 the way K=20 is already treated.

**Detection:** `paper/sanity_gate.py` now prints `[N seed(s) zero-variance -> only M
enter Phi]` next to every affected cell. Also hits celeba K4 GTG acc/fair_eo (3 usable)
and several adult K20 metrics.

### A34. The CelebA partition cache returned `Male` for any requested sensitive attribute
`celeba/celeba/task.py`, `celebanoniid/celebanoniid/task.py` (`load_data`) · found and fixed 2026-09-15 on branch `direction-b`

The cache key (`n{K}_p{pid}_tr30000_te5000.npz`) does not include the attribute. A cache
**hit** returned the stored `s_train` / `s_test` whatever `sensitive_attr` asked for, and a
cache **miss** stored the requested attribute under that same key, so the first caller
decided what every later caller got.

**Latent, not realized:** every shipped CelebA run (scoring and ET-fair training) asked for
`Male`, the attribute every cache on this machine was built with. No number is affected.

**Fix (direction-b):** the cache only ever stores `Male`. Any other attribute is rebuilt
from the HF label columns into a sidecar `*.s-<attr>.npz`, repeating `load_data`'s row
selection and split, and written only if the rebuilt `Smiling` and `Male` columns match
the cached ones. Verified offline for `Young` on `celeba` n1_p0 (28 s; the `Male` path is unchanged).

### A36. LOO and GTG score non-IID clients against a model FedAvg never built
`robustness_metric.aggregate_models` (plain mean) · `cont_evals.l1o_evals`, `gtg_evals` · `client_app.fit` returns `len(x_train)`

FedAvg weights each client by its training-set size. Every coalition model in LOO and GTG is the
**plain** mean of the client weights. The IID apps partition into equal parts (`IidPartitioner`,
ADULT's equal slices), so the two coincide there. The non-IID apps of all three datasets use a
Dirichlet(0.5) split, so they don't: ADULT non-IID K=4 clients hold 596 / 16 921 / 9 305 / 12 250
training rows.

Measured with `gtg_check.py` (adultnoniid, K=4, seed 42, 2026-09-15):

```
round | saved global vs plain mean | vs size-weighted mean | acc: saved global | plain mean of all 4
  2   | max|dw| 3.9e-02           | 1.2e-07               | 0.8471            | 0.8227
 10   | max|dw| 1.6e-02           | 9.8e-08               | 0.8469            | 0.8245
```

- **LOO** takes v(All) from the saved, size-weighted global model but v(All \ i) from a plain mean,
  so every client in a round carries the same offset (+0.022 for acc at round 10). Ranks within a
  round survive; signs and L2 distances do not. Client 0 scores +0.003 as shipped and -0.020 when
  both sides use the same aggregation.
- **GTG** attributes u(plain mean of all clients) - u(previous global): the change to a model that
  was never deployed (0.8245 against the real 0.8469 above).

**Scope:** every non-IID LOO and GTG column in `data/combo`, and the direction-B gap columns if they
are run unchanged.

**Fixed 2026-09-16 on `direction-b`.** `weighted_utils.recover_fedavg_coeffs` recovers each client's
FedAvg coefficient for a round by least squares on the client updates against the previous global;
`cont_evals._coalition_coeffs` labels every row `size`, `equal` (equal partitions: plain mean, as
shipped) or `fallback` (not recoverable: plain mean, and LOO then scores V(All) as that same plain
mean). Verified with `gtg_check.py`: on adultnoniid at K=4 and K=20 the recovered coefficients equal
the clients' training-size shares to five decimals (K=20 shares span 0.0003-0.106) and rebuild the
saved global to 3e-7, and u(all clients) now equals u(saved global), so LOO's round-constant offset
is gone. ADULT IID `l1o_acc` stays bit-identical to the pre-fix code.

### A37. GTG's best-coalition step zeroes most clients and breaks symmetry
`gtg_shap.py` (`compute` -> `_find_best_subset`, `_calculate_best_subset_values`)

After estimating the Shapley values, `compute()` keeps them only for the clients in the single
highest-utility coalition it evaluated and sets every other client to exactly 0. This is a different
mechanism from A32, where coalition utilities barely move and every client is 0.

Measured with `gtg_check.py` (adultnoniid, K=4, seed 42, rounds 2 and 10; `acc`, `gap_acc_dp`,
`gap_loss_eop`; the real clients, plus a duplicated client, plus two planted group-skewed clients):

- **The estimate before this step is sound.** Against exact Shapley over all 2^K coalitions: max error
  0.001-0.007 on values of 0.02-0.06, Spearman 0.80-1.00, signs agree for 113 of 114 client values
  (the exception: +0.0002 exact vs -0.0001). Production output equals it masked by the best
  coalition in every case, so the check reproduces what `robustness.py` runs.
- **The step discards 25-80% of it.** For gap utilities the best coalition is often a single client:
  `gap_loss_eop` at round 10 keeps only client 3 (+0.024) and zeroes client 0, whose exact value is
  the largest (+0.040), and client 1 (-0.034).
- **Symmetry fails.** A client entered twice has equal exact values by construction; after the step
  one copy keeps -0.027 and the other gets 0 (`acc`, round 2), or +0.054 against 0 (`gap_loss_eop`, round 2).
- **Planted harm disappears.** A client trained on one group only, whose addition lowers the utility
  (6 cases across the three metrics), gets a negative exact and pre-step value in all 6 and exactly 0
  from production GTG in all 6.

**Two further defects in the same estimate, found in review on 2026-09-16.** They matter because the
fix below promotes GTG's own estimate to a reported column:

- The first player of each permutation cycles by index over a fixed 30 permutations, so at K=20
  clients 0-9 start twice and clients 10-19 once. The first position carries the largest marginal,
  and on a symmetric game (10 seeds) that inflates clients 0-9 by about 1.4x.
- The within-permutation eps truncation compares u(all) with u(empty) at position 0, so a round whose
  clients cancel out (u(all) ~= u(empty)) returns zeros for everyone even when individual clients move
  the utility a lot. Gap utilities are non-monotone, so this is their typical case: on ADULT IID round
  10, `_contribution` zeroed all five clients while exact Shapley was non-zero.

**Fixed 2026-09-16 on `direction-b`.** `<method>_contribution` is still GTG exactly as the paper ran
it. `<method>_shapley` is `gtg_shap.shapley_values`: exact over all 2^K coalitions for K <= 6 (at K=4
that is the same 15 coalitions GTG already scores, so it costs nothing) and, above that, permutation
sampling with no truncation and every client first equally often. Verified with `gtg_check.py`:
`_shapley` equals an independent exact enumeration to <= 1e-17 in all 24 K=4 scenarios on ADULT iid
and noniid, a duplicated client gets the same value as the original, and `evaluate_round` reproduces
the harness exactly. At K=20 against a 200-permutation reference, `_shapley` (40 permutations) gives
max|diff| 1.3e-2 / Spearman 0.87 on `acc` and 4.6e-3 / 0.94 on the accuracy gap, while
`_contribution` gives Spearman 0.69 and 0.07 with 7/20 and 17/20 clients zeroed. Read K=20 gap columns
from `_shapley`; raising the permutation count is the knob if its ~1e-2 noise matters.

---

## P1 — results are not reproducible from the command line

### A6. GTG results depend on the run's interruption history
`robustness.py:243` (`set_seed(seed)`) · `gtg_shap.py:58` (`np.random.randint`) · `gtg_shap.py:118` (`np.random.shuffle`)

`set_seed(seed)` is called **once per seed**, outside the `num_clients` and `round`
loops. `GTGShapley` then draws from the *global* numpy RNG — once at construction
and 30 times per round in `compute()`. So round *r*'s permutations depend on how
many rounds were scored before it **in the same process**.

Three ways the same command produces different numbers:
- **Resume.** `_load_done_keys` (`robustness.py:184-197`) skips already-written rounds,
  so a resumed run scores the remaining rounds against a different RNG stream than
  a fresh run would.
- **`FLR_ROUND_ORDER`** (`robustness.py:216-232`) reorders rounds.
- **`--num_clients`** decides whether K=20 consumes RNG before K=4.

**Confirmed empirically** (2026-09-01, `archive/verification/`). Same command,
same checkpoints, only `FLR_ROUND_ORDER` reversed:

```
global_gtg_acc        max|diff| = 0.000e+00    <- identical models + identical data
gtg_acc_contribution  max|diff| = 3.880e-03
rows differing:       21 / 40
```

`global_gtg_acc` being bit-identical proves the models and test set are the same;
the whole difference is the RNG stream. On contributions that range 0.000-0.066,
3.9e-3 is 6-20 % of a typical value.

**Failure:** `gtg_*` columns in `data/combo/` cannot be regenerated from
`(dataset, seed, method)` alone — you also need to know how the original run was
interrupted. LOO is unaffected (no RNG). **Fix shape:** move `set_seed` inside the
round loop, or seed a `np.random.Generator` per `(seed, K, round)` in `GTGShapley`.

### A7. Missing `seed` env var crashes inside a Ray worker with an unreadable trace
`adult/adult/task.py:6` · and the same line in all 7 other apps

```python
SEED = int(os.environ.get("seed"))     # TypeError if unset
```

Import-time, inside Flower/Ray workers. **Failure:** forgetting `$env:seed` produces
`TypeError: int() argument must be a str... not 'NoneType'` buried in worker logs
rather than a one-line "set `seed`". A default + explicit error message costs one line.

Related: `reweight_eval.py:32` works around this with `os.environ.setdefault("seed","42")`,
which means **anything that imports `reweight_eval`** — `eval_bl_models.py:21`,
`eval_et_models.py`, `eval_stdy_models.py:26` — silently pins seed 42 for data
loading regardless of what the caller intended.

### A8. GTG's convergence criterion is dead code; it always runs exactly 30 permutations
`gtg_shap.py:54-60` and `gtg_shap.py:155-176`

`converge_min = max(30, num_players) = 30` and
`max_number = min(2**K, ..., max_permutations=30) = 30`.
So `_check_convergence` returns `False` while `index < 30` and `True` at `index == 30`
— lines 160-176 are unreachable.

**Failure:** not a wrong number, but `convergence_criteria`, `last_k` and
`convergence_denom_floor` are inert. MAIN.tex:347 reports ε₂ = 0.05 as a GTG
parameter that governs sampling; in this code it governs nothing. Tuning it will
waste a day, and it is exactly the kind of thing a reviewer asks about.

---

## P2 — config / code / paper drift

### A9. `priv` in `data/combo/` and `priv` in `data/{bl,st,dy,et}/` are different measurements
`robustness.py:157-158,263` vs `privacy_metric.py:53-54` vs `reweight_eval.py:257-261` vs MAIN.tex:347

| | fine-tune epochs | fine-tune lr | member_ratio | attack_train_ratio |
|---|---|---|---|---|
| `robustness.py` → `data/combo/` (Tab. 3/4) | **5** | **5e-4** | **0.2** | **0.7** |
| `MIAConfig` defaults → `data/{bl,st,dy,et}/` (Tab. 5) | 2 | 1e-3 | 0.5 | 0.5 |
| MAIN.tex:347 | 2 | 1e-3 | 0.5 | 0.5 |

**Failure:** a `priv` value in Tab. 3/4 is not comparable to a `priv` value in
Tab. 5, and the combo one does not match the protocol the paper describes. Pick
one config (the `MIAConfig` defaults match the paper) and make `robustness.py`
use it, or state the difference explicitly in MAIN.tex.

### A10. `attack_metric.py`'s own docstring contradicts its parameter table
`attack_metric.py:8-14` (says CelebA ε=0.03, α=0.007) vs `attack_metric.py:19` (`(0.01, 0.0025, 40)`)

MAIN.tex:347 confirms the **code** is correct. **Failure:** stale comment; the next
person tuning CelebA `res` trusts the docstring.

### A11. The in-loop CelebA `res` proxy attacks a 3× larger ε than the metric it's scored against
`score_metrics.py:195` (claims "same eps/alpha as the post-hoc eval") vs `score_metrics.py:200` (`celeba: (0.03, 0.007, 7)`) vs `attack_metric.py:19` (`celeba: (0.01, 0.0025, 40)`)

adult and imdb do match on ε/α (only the iteration count differs, which is the
documented affordability trade-off). CelebA does not.

**Failure:** ST/DY `res`-weighting on CelebA optimises client weights for a
perturbation budget that is not the one Tab. 5 evaluates them at.

**Measured 2026-09-14** (celebanoniid K4 seed 42, round-10 global):

```
  in-loop proxy (0.03, 0.007,  7) : res = 0.0026
  eps 0.03 at eval depth    (40)  : res = 0.0009
  post-hoc eval (0.01, 0.0025, 40): res = 0.5080
```

So the proxy scores **in the saturated regime** — the shallower 7-step attack does not
rescue eps=0.03. The weighting works anyway: coalition scores land around 0.001-0.004
rather than exactly 0, so the client *ordering* survives, and it transfers to the eval
radius (celeba res BL 0.520 -> ST 0.748, DY 0.835 at K=4; 0.532 -> 0.609/0.628 at K=20).
An empirically sound result resting on a fragile mechanism: disclose it, or align the
proxy to 0.01 and accept that every shipped celeba ST/DY number needs re-running.

### A12. Default ET config does not reproduce the paper's ET rows
`adultnoniid/pyproject.toml:31`, `celebanoniid/pyproject.toml:33`, `imdbnoniid/pyproject.toml:32` vs MAIN.tex:868

- `et-lambda-fair = 0.5` in all three; MAIN.tex says **λ = 2.0**. (The 0.5→2.0 fix
  was applied at run time and never written back to the config.)
- `celebanoniid/pyproject.toml:34,37`: `et-adv-ratio = 0.5`, `et-adv-steps = 7`;
  MAIN.tex says **100 % of batches and 10 steps** for CelebA. `imdbnoniid` already
  has `1.0` / `10`.

**Failure:** `./run_et_experiments.sh celebanoniid 4 fair 42` with the committed
config reproduces the *superseded* result, not the published one.

### A13. `requirements.txt` cannot run the first step of the pipeline
`requirements.txt`

Missing: `tqdm` (`robustness.py:33` — the main driver), `requests`
(`imdbnoniid/imdbnoniid/task.py:126`), `scipy` (`paper/sanity_gate.py:25`,
`paper/compute_score_diff.py:8`), `matplotlib` (`paper/_run_figs.py:6`).

Also `tensorflow`, `keras` and `numpy` are unpinned here, while the app
`pyproject.toml`s pin `tensorflow>=2.11.1,<2.18.0` and the `.keras` checkpoints in
`420/` were written by TF 2.17 / Keras 3.2. Two dependency declarations, disagreeing.

**Failure:** `pip install -r requirements.txt && python robustness.py ...` on a
clean box dies with `ModuleNotFoundError: No module named 'tqdm'`, and a
sufficiently new TF may not read the saved models at all.

### A30. The ADULT feature pipeline runs at eval time against unpinned scikit-learn/pandas
`adultnoniid/adultnoniid/task.py:171-265` · `requirements.txt`

`load_data` calls `fetch_openml(data_id=1590)` and re-fits `OneHotEncoder` +
`StandardScaler` + `train_test_split` **on every evaluation**. Unlike CelebA
(`.npz` partition cache) and IMDB (content-hashed embedding cache), ADULT has no
cached snapshot — the exact test matrix is a function of the installed
scikit-learn and pandas versions, neither of which `requirements.txt` pins.

**Measured** (2026-09-01): a clean-room re-run of `adultnoniid` / K=4 / seed 42 /
`l1o_acc` on pinned TF 2.17 + numpy 1.26 but sklearn 1.9.0 / pandas 3.0.5
reproduced all 40 shipped rows with the correct schema, but not bit-exactly:

```
global_l1o_acc        max|diff| = 7.166e-03    exact: 4/40
l1o_acc_contribution  max|diff| = 7.575e-03    exact: 5/40
```

~7e-3 on 9 769 test rows is ~70 samples landing on the other side of the split —
consistent with a slightly different train/test partition, not a scrambled
feature order (feature count is still 108, so the saved models load correctly).
I did not isolate which library is responsible.

**Failure:** the ADULT columns of `data/combo/` cannot be reproduced exactly on a
fresh install, and the discrepancy is the same order of magnitude as the smaller
contributions themselves. **Fix shape:** pin `scikit-learn` and `pandas`, or cache
the processed ADULT matrix to disk the way CelebA and IMDB already do — which
would also stop every eval re-downloading and re-encoding the dataset.

**Second data point (2026-09-15, this laptop):** `adultnoniid` / K=4 / seed 42 / round 10 /
`l1o_fair` differs from `data/combo` by up to **1.36e-2** (global) and **5.2e-3**
(contribution), while the same run from an unmodified `fills-eps-fix-2026-09-15` checkout
and from `direction-b` is bit-identical. The drift is the environment, not the code, and
fairDP drifts more than the 7.6e-3 measured for `acc` above.

### A35. The sensitive group is coded differently in ADULT and CelebA
`adult*/task.py` (`s = sex == "Female"`) · `celeba*/task.py` (`SENSITIVE_ATTR = "Male"`)

The paper uses the `sex` attribute for both datasets, but s = 1 means **female** in ADULT
and **male** in CelebA. Every fairness score (fairDP, fairEO) and every direction-B gap
score is symmetric in the two groups (`|A - B|`), so no number depends on the coding:
CelebA `l1o_fair` (K=4, seed 42, round 10) is identical with either group as s = 1
(checked 2026-09-15). It matters only where a group is *named*: signed per-group
analyses, group-size columns, prose.

**Direction B (branch `direction-b`):** `robustness.py --sensitive_attr` takes `column=value`
and defaults to female as group A on both datasets (ADULT `sex=Female`, CelebA `Male=0`);
every gap row records `group_a`. Shipped columns and the handed-in text are unchanged.

### A14. The interpreter on this machine is in exactly that broken state
Not a repo bug, but it blocks any local verification.

`numpy 2.4.6` is installed against `tensorflow 2.17.1`, which requires `numpy<2.1`
→ `import tensorflow` raises
`SystemError: initialization of _pywrap_checkpoint_reader raised unreported exception`.
`flwr 1.16.0` also violates the `<1.16.0` pin in every `pyproject.toml`.
This is the "this laptop has a TF DLL issue" noted at `test_weighted_strategy.py:4`.

**Fix applied:** a pinned `.venv` (`numpy<2.1`, `tensorflow==2.17.1`, `flwr<1.16`,
plus the four missing packages) — see the end-to-end run below.

---

## P3 — entry points and paths

### A15. `eval.ps1` is broken for the three cheapest methods, on every machine
`eval.ps1:32-34` and `eval.ps1:45-50`

```powershell
$script_base = "C:\BME\7. felev\szakdoga\FLRobustness"   # absolute, machine-specific
$accuracy    = Join-Path $script_base "accuracy.py"       # archived
$loss        = Join-Path $script_base "loss.py"           # archived
```

`accuracy.py` / `loss.py` were retired as stale duplicates *with their own sign
flips* and now live in `archive/old_archive/stale_drivers/`.

**Failure:** `.\eval.ps1 adultnoniid noniid 10 gtg_acc 42` → "The term ... is not
recognized". README.md advertises this script. `robustness.py` already handles
`*_acc` and `*_loss`, so all three branches should route there.

### A16. `robustness.py` finds no checkpoints when run from anywhere but the repo root
`robustness.py:87` (`BASE_DIR = "./420"`, cwd-relative) vs `adultnoniid/adultnoniid/task.py:16` (`BASE_DIR.parent / "420"`, file-relative)

**Failure:** run from a subdirectory it prints `Directory not found: ./420/...` per
cell and exits 0 with an empty CSV — A4 again, with no error. Resolve `BASE_DIR`
from `__file__` like the apps do.

### A17. Running an experiment dirties a git-tracked file
`set_num_cl.py:22-67`

`set_num_cl.py` regex-rewrites `<ds>/pyproject.toml`, which is tracked in git.

**Failure:** every training run leaves `git status` dirty and can commit a stray
`num-supernodes`, so `git status` stops being a usable signal mid-experiment.

### A18. `reweight_eval.load_test_data` maps every IID dataset name to the non-IID loader
`reweight_eval.py:149-166`

`"adult" in d` → `adultnoniid.adultnoniid.task`; same for celeba and imdb.

**Failure:** latent. All current callers pass non-IID names, so nothing is wrong
today — but `eval_bl_models.py --datasets adult` would silently score IID models
against non-IID test data and report it as valid.

### A19. Metric modules import the IID `imdb` app even when scoring `imdbnoniid`
`cont_evals.py:13` · `robustness_metric.py:204` · `attack_metric.py:172` · `attack_metric.py:427`

All four do `from imdb.imdb.task import process_text`, while `reweight_eval.py:226`
does the same import from `imdbnoniid`. The lazy-import comments say the point was
to survive hosts that deploy only the `*noniid` dirs — but the module they import
is the IID one.

**Failure:** on an `imdbnoniid`-only host these raise `ModuleNotFoundError`, which
A1/A2 convert to `0.0` metric values. Half-fixed instance of the same bug class
that already cost a re-run.

---

## P4 — results-CSV schema and completeness

### A20. Combo column names are not uniform, and one consumer doesn't compensate
`data/combo/*.csv` headers · `robustness.py:289-291` · `reweight_eval.py:71-79`

- adult / celeba / imdb: `<method>_loss_contribution`. **cifar: bare `loss_contribution`.**
- adult / cifar carry `<method>_priv_abs_contribution` and `<method>_fair_eo_abs_contribution`; celeba / imdb do not.
- `robustness.py:291` writes a generic `abs_contribution` column that no combo file preserves.

`paper/compute_score_diff.py:29-40` and `paper/sanity_gate.py:34-35` both defend with
a `pick(*candidates)` fallback. `reweight_eval.py:72-73` does **not**:

```python
"loss": "loss_contribution",   # exists only in cifar
"acc":  "acc_contribution",
```

**Failure:** `reweight_eval.run_experiment` would hit `score_col not in df_s.columns`
(`reweight_eval.py:120-122`) and silently `continue` past any loss-targeted group.
Masked today only because `TARGET_DIMS` uses fairDP/res/priv.

### A21. Silent holes in the shipped results
Measured from the files as they stand:

| File | Hole |
|---|---|
| `results_adult_gtg.csv` | 20 NaN in `gtg_adv_contribution` |
| `results_celeba_gtg.csv` | 400 NaN in `gtg_rob_contribution` |
| `results_celeba_l1o.csv` | 400 NaN `l1o_rob_contribution`; 160 NaN each in acc / fair / fair_eo / loss |
| `results_celebanoniid_{gtg,l1o}.csv` | 40 NaN across 5-6 contribution columns |
| all 4 adult + cifar files | `*_priv_abs_contribution` is 1200/1200 NaN — column exists, never populated |
| `420/celebanoniid/20/42` | 92 of the expected 211 checkpoints |
| `420/celebanoniid/20/107` | missing entirely |

**Failure:** every downstream aggregation is a `pandas` mean, which skips NaN
silently. A CelebA `rel` cell is averaged over the seeds that happen to have data,
and the table reports it as a 5-seed mean. There is no coverage assertion anywhere
in `paper/`.

### A22. The documented CelebA merge step cannot run
`archive/oneoffs/merge_celeba.py:35`

`SRC_DIR = data/celeba_to_distribute` — the directory does not exist. `PIPELINE.md`
lists this as the merge step for CelebA.

Also `merge_celeba.py:105`: `groupby(KEY, as_index=False).mean(numeric_only=True)`
silently averages duplicate keys instead of erroring, so a double-merge produces a
plausible wrong number.

### A23. `data/README.md` documents two artifacts that don't exist
`data/README.md` — "Regenerating / collecting" cites `data/weighting_effect.csv`
(absent) and `bash collect_data.sh` (now `archive/old_archive/collect_data.sh`).
`weighting_effect.py` itself is now in `archive/oneoffs/` per your call.

---

## P5 — dead / misleading code

| # | Location | Issue |
|---|---|---|
| A24 | `score_metrics.py:334` | `return name.lower() == "loss" and False` — always `False`, no callers. |
| A25 | `<ds>/task.py:278` (all 8 apps) | `set_num_clients()` uses `toml.dump()`, which `set_num_cl.py:18-21` documents as *breaking* Flower's config reader. Unused but callable. |
| A26 | `<ds>/task.py:276` | module-level `config = toml.load(toml_file)` — assigned, never read. |
| A27 | `weighted_strategy.py:45` | imports `fedavg_aggregate`, never calls it; the class has an identical `_aggregate_fedavg` at line 208. Third copy: `robustness_metric.aggregate_models`. |
| A28 | `weighted_strategy.py:223` | `adopt_reweighted_or_fallback` — the retired do-no-harm guard, superseded by β/cap shrinkage. Its docstring points at `build_tab4_final.py`, now in `archive/superseded/`. |
| A29 | `gtg_shap.py:186,189` | mojibake in Hungarian comments (`legr\ufffdvidebb`). Lines 236 and 253 `print()` raw Shapley vectors to stdout from `_normalize_values`. |

---

### A33. GTG always runs exactly 30 permutations, and re-draws the empty-coalition utility in each
`gtg_shap.py:26,41,56-60,74-76,120,155-159`

Two mechanics that set the cost of every GTG cell and, for stochastic metrics, part of its noise:

1. **The adaptive stop never fires.** `_check_convergence` returns `False` while `index < converge_min`
   *before* it checks `max_number`. With the defaults `converge_min=30` and `max_permutations=30`, K=4 gets
   `max_number = min(16, 30, 30) = 16` yet still runs to 30, and K=20 runs to exactly 30. The relative-error
   convergence criterion and the `2**K` / `max_percentage` cap are unreachable for K in {4, 20}. Not a
   correctness bug (30 is the intended cap), but those branches are dead code for every shipped cell.

2. **The empty coalition is not cached.** `evaluate_subset([])` bypasses `evaluated_subsets` and calls the
   utility on the previous-round model once per permutation: 30 extra metric evals per round at any K. For
   deterministic metrics (acc, loss, fair) that is only cost. For `rob` (randomized smoothing, fresh
   `tf.random.normal` draws on every call) each permutation also gets a different empty-coalition baseline,
   so the first player's marginal and the j=0 truncation decision carry extra Monte Carlo noise.

**Cost model, metric evals per round:** LOO = K+1. GTG = grand + global + 30 empty + the cached subsets that
survive eps-truncation (at most 14 for K=4, 570 for K=20): ~32-46 at K=4, ~32-601 at K=20. For K=4 that
predicts GTG at 6.4-9.2x LOO wall time; the 2026-09-14 celebanoniid fills measured 6.4x (fair_eo) and
7.0x (fair).

**Fixed 2026-09-16 on `direction-b`** (the user chose to include it): the empty coalition is cached like
any other subset, which removes 29 evaluations per round at K=4. Deterministic metrics (acc, loss, fair,
res and the gaps) are unchanged; `rel` and the in-loop `rel` proxy now consume a different random stream,
so those no longer reproduce the shipped numbers bit for bit — reproduce them from `master`.

---

### A38. `--noise_samples` was parsed and then ignored; reliability always ran at L=100
`robustness.py:138-141` vs `robustness_metric.py:119,122,182,188`

Three layers that each looked correct and together did nothing:

1. `robustness.py` defines `--noise_samples`, and `args.noise_samples` is then read **nowhere** in the file.
2. `calculate_image_robustness` reads `FLR_NOISE_SAMPLES` from the environment — which nothing ever set.
3. `calculate_tabular_robustness` did not read it at all, so even exporting the variable by hand changed
   nothing on ADULT.

So `L` was pinned at the default 100 for every dataset, on every run, always. Both docstrings say
"Honors FLR_NOISE_SAMPLES env var (set by robustness.py --noise_samples)", describing a hand-off that did
not exist, and the flag's own help text advertises "30 gives ~3× speedup with modest CI widening" — a
speedup that was unreachable through the documented interface.

**Failure:** not a wrong number. Every reliability value ever produced used L=100, which is what the paper
describes, so nothing shipped is affected. The damage is an inert tuning knob: a sweep launched with
`--noise_samples 30` silently costs 3× what the operator budgeted and yields L=100 numbers that the
operator will label L=30. This is the same class as A8 — a parameter that reads as tunable and governs
nothing — and exactly what a reviewer asks about.

**Fixed 2026-09-16 on `direction-b`** (`68bde74` makes the tabular path honour the env var, `b72ba9a` makes
`main()` set it from the flag). Default behaviour is unchanged at L=100; only an explicit flag or env var
now takes effect. Verified end to end on adultnoniid K=4 seed 42 round 10, `l1o_rob`: `--noise_samples 5`
finishes in 12 s against a ~4 min reference at L=100, rc=0, sane values.

**Why the knob is worth having, measured** (`_direction_b/rel_L_sweep.py`, round-10 global model,
8-32 repetitions per cell): the gap-utility estimator noise at L=100 is 0.00017 (`dp`) and 0.00060 (`eop`),
against a per-client contribution spread of 0.003-0.007 — so reliability is resolvable with room to spare,
and its weak client discrimination is a real property, not a measurement artefact. Noise follows
~1/sqrt(L·N). At L=20 the `dp` gap-noise is 0.00034, still ~10× below the signal, at 3.9 s per evaluation
against 20.0 s — about 5× less compute. `eop` groups are small (A = 359 samples) and should stay at
L=50-100.

---

## Suggested order of attack

1. **A1 + A2 + A4** together — one change of policy (`nan` instead of `0.0`, non-zero
   exit on zero rows). This is the difference between "a failure is loud" and
   "a failure is a number in your thesis".
2. **A6** — seed per round. Without it no GTG number is reproducible on demand.
3. **A5** — adult `rel` on the wrong partition is a correctness bug in shipped data.
4. **A13 + A15** — makes the release installable and its documented entry point work.
5. **A9 / A11 / A12** — decide which number is canonical, then make config and
   MAIN.tex agree. Cheap, and exactly what a reviewer probes.
6. **A21** — add a coverage assertion to `paper/sanity_gate.py` so holes fail the gate.
7. The rest is hygiene.

---

## Appendix — forensics: which findings actually touched the shipped numbers?

Run 2026-09-01 against `data/combo/` as it stands. This is the question that
decides whether a fix needs a re-run or not.

### A1 / A2 / A4 — LATENT, not realized. No re-run needed.

Scanned all 16 combo files for the failure signature: a LOO row where
`<method>_<metric>_contribution == global_<method>_<metric>` exactly (which is
what `contribution = global - 0.0` produces), and any `global_*` recorded as
exactly `0.0`.

```
(no hits in any combo file)
```

So no shipped number was ever produced by a swallowed exception. The `0.0`-on-
failure path is a live hazard for the *next* run, not damage already done —
which means **A1/A2/A4 can be fixed without invalidating anything**.

### A5 — REALIZED, but numerically negligible. Fix the code; no re-run.

At `num_partitions=1` both ADULT apps return the whole dataset, so I checked
whether the 80/20 split actually differs. It does — the IID app slices
positionally while the non-IID app reorders by Dirichlet class-grouping before
`train_test_split(random_state=42)`:

```
adult/      (IID app):   x_test (9769, 108)   y mean 0.231139
adultnoniid/(NIID app):  x_test (9769, 108)   y mean 0.240864
identical: False       row overlap: 1943/9769 = 19.9%
```

So in every `results_adultnoniid_*` row, `rel` is measured on a test set sharing
only ~20 % of its rows with the one used for acc / loss / fair / res / priv.
Same defect reaches `data/{bl,st,dy,et}` via `reweight_eval.eval_all_metrics`.

Measured impact on round-10 global `rel`, shipped path vs. correct path:

```
  K  seed | rel AS SHIPPED  rel ON CORRECT SET      diff
  4    42 |       0.990026            0.989528  +0.00050
  4   107 |       0.986355            0.987136  -0.00078
  4   123 |       0.988750            0.989014  -0.00026
 20    42 |       0.988561            0.989189  -0.00063
 20   107 |       0.987367            0.987347  +0.00002
 20   123 |       0.988617            0.988035  +0.00058
  max |diff| = 0.00078    mean |diff| = 0.00046
```

ADULT `rel` is saturated near 0.988, so any reasonable test split lands in the
same place. Against the shipped rel contributions
(`l1o_rob` median 4.0e-3, p90 8.4e-3; `gtg_rob` median 2.7e-3, p90 6.6e-3) a
7.8e-4 perturbation is ~20 % of a typical value at worst, and only 6-8 % of
non-zero rows are smaller than it. The true contribution error is smaller still,
since contribution is a *difference* of two values computed on the same wrong
set, so most of the offset cancels.

**Verdict:** fix `robustness_metric.py:60-62` so it stops discarding `x_test`,
but the published ADULT rel numbers stand.

### A6 — REALIZED and material. Fixing it changes published numbers.

See A6 above: 21/40 contributions move by up to 3.9e-3 on values ranging
0.000-0.066. Any fix re-derives every `gtg_*` column in `data/combo/`.

### A30 — REALIZED. Environment pinning, not a code fix.

max|diff| 7.6e-3 on a clean-room ADULT re-run; see A30.

### A38 — LATENT, not realized. No re-run needed.

The flag never took effect, so every shipped reliability number was produced at the
default L=100 — the value the paper describes. The defect is an inert knob, not a wrong
`L`. Nothing in `data/combo/` or `data/{bl,st,dy,et}/` needs recomputing.

It becomes live the moment someone uses the now-working flag, so the measured noise
budget belongs with it: at L=100 the gap-utility noise is 0.00017 (`dp`) / 0.00060 (`eop`)
against a 0.003-0.007 per-client signal; L=20 is safe for `dp` groups, L=50-100 for `eop`.

---

## Appendix — end-to-end verification run (2026-09-01)

Environment: fresh `.venv` built from the corrected `requirements.txt`
(TF 2.17.1, numpy 1.26.4, flwr 1.15.x); the global interpreter is unusable, see A14.

| Check | Result |
|---|---|
| `python paper/sanity_gate.py` (from repo root) | **GATE PASSED** — all 6 paper family x method cells positive; cifar reported as info only |
| `robustness.py --dataset adultnoniid --partition noniid --num_rounds 10 --seed 42 --method l1o_acc --num_clients 4` | exit 0, 40 rows, 0 NaN, schema exact |
| same, `--method gtg_acc` | exit 0, 40 rows, schema exact |
| same, `--method gtg_acc` with `FLR_ROUND_ORDER` reversed | exit 0 — and 21/40 contributions differ (see A6) |
| `test_weighted_strategy.py` | 13/13 pass (the 3 strategy tests un-skip once TF/flwr import) |
| `test_inloop_scoring.py` | 5/5 pass |
| `smoke_lossflag.py` | not run — needs the IMDB USE embedder (network + ~1 GB download) |

Emitted schema, matching `robustness.py:281-292` exactly:

```
dataset, partition, num_clients, seed, round, client_id,
global_<method>, <method>_contribution, abs_contribution
```

Artifacts kept under `archive/verification/`.

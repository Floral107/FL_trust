# AUDIT.md — correctness review of what remains after the cleanup

Scope: everything still at the repo root + `paper/` + the 8 Flower apps, after
the `cleanup/repo-structure` reorganisation. `archive/` is not audited.

Ordered by **how much damage the failure does before you notice it**. Every
entry names the concrete failure, not the smell.

## Status as of 2026-09-01

| | Findings | State |
|---|---|---|
| Fixed | A1, A2, A4, A5, A7, A10, A13, A15, A21, A24, A26, A27, A28, A29 | Commits `d3e8d2d`, `f3d95dd`. Verified number-neutral: an `l1o_acc` re-run is bit-identical to the pre-fix output. |
| Fixed (dead code removed) | A25 | `set_num_clients()` deleted from all 8 `task.py`; recoverable from git history. |
| Documented, deliberately not fixed | A6, A30 | Both would change published numbers. Written into `PIPELINE.md` as known limitations. |
| Open — needs your decision | A9, A11, A12 | Config/paper drift. Fixing means deciding which number is canonical, then re-running or amending MAIN.tex. |
| Open — hygiene | A3, A8, A14, A16, A17, A18, A19, A20, A22, A23 | No decision needed; none is urgent. |

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

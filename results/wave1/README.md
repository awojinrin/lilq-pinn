# Wave 1 results (Computational Package 1, Addendum v2.2 Section 4.2)

These are the results of wave 1, run on TAMU Grace on 2026-09-29 at commit
`8a3f5f7`, the code this branch starts from. Its commits add only
`results/`. `COMMIT` records the commit and the source-tree hash that every
job checked before running.

The folder holds every file the wave wrote except the trained models
(`*.pt`, 301 files), which are kept on Grace and offline and are available
on request. The 196 Component C collocation point sets
(`C_oversampling/runs/*/collocation.npz`) were added in a second commit, at
the advisor's request (reply of 29 Sept, item 2.6). The folder
contains all 458 files of `wave1_report.tar.gz`, byte for byte, plus each
run's own logs (`iterations.csv`, `run.json`, `summary.json`,
`environment.txt`).

## Jobs and SUs

All nine jobs completed with exit code 0 (`sacct.txt`; the report job lists
itself as RUNNING because it ran `sacct` before it finished, at 3:18). The
output of each job is in `slurm_logs/`.

| Job | Script | Allocation | Elapsed | SU (computed) |
|---|---|---|---|---|
| 19898410 | 00 preflight | 8 cores + 1 A100 | 0:11:07 | 14.8 |
| 19898411 | 29 A1 gate | 8 cores + 1 A100 | 1:06:04 | 88.1 |
| 19898412 | 10a timed LiL-Q, CPU | 48 cores (exclusive) | 1:18:37 | 62.9 |
| 19898413 | 10b timed LiL-Q, GPU | 48 cores + 2 A100 (exclusive) | 0:06:45 | 21.6 |
| 19898414 | 42 Component C | 24 cores | 0:12:41 | 5.1 |
| 19898415 | 43 B6 basis study | 24 cores | 0:05:18 | 2.1 |
| 19898416_0 | 20 B4 GPU, Bratu | 48 cores + 2 A100 (exclusive) | 0:37:20 | 119.5 |
| 19898417_0 | 21 B4 CPU, Bratu | 48 cores (exclusive) | 0:15:18 | 12.2 |
| 19898418 | 91 wave report | 24 cores | 0:03:18 | 1.3 |

The SU column is computed as (cores + 72 per A100) x elapsed hours and adds
up to about 328. `myproject` shows 332 SU charged for the wave. An exclusive
A100 node is allocated both of its GPUs (`gres/gpu=2` in `sacct.txt`), so it
is charged 192 SU/h, not 120. Grace printed no "Requested SUs" line at
submission.

## Where each check's result is

| Check | File | Result |
|---|---|---|
| Preflight | `slurm_logs/lilq-preflight.19898410.out` | 415 passed, 4 skipped; smoke runs of B and C ok |
| A1 (plain PINN, 1 h) | `A_calibration/checks/a1.json`, `A1_s0/` | eps_u = 5.80e-5 <= 1e-3: passed |
| A2 (GPU determinism) | `A_calibration/checks/a2.json` | bit-identical: passed |
| F2 Jacobian | `A_calibration/checks/f2_jacobian.json` | max relative FD error 3.8e-9: passed |
| Component A search | `A_calibration/search/`, `A_calibration/tuning_log.md` | configurations for wave 2's screen |
| Section 3.3 runs (both passes) | `B_instrumentation/<benchmark>_P<P>_<device>_<pass>/`, `B_instrumentation/runs_index.csv` | 63 runs, all `ok` |
| B1 reproduction | `B_instrumentation/reproduction_check.csv` | 15 violations, 10 round-off, 31 reported, 84 ok (below) |
| B2 | `run.json` of each run | recorded at k=1 |
| B3 GPU/CPU | `B_instrumentation/gpu_cpu_equivalence.csv` | equivalent at all 5 sizes (P=1875 under the amended round-off rule only) |
| Section 3.7 Beltrami, pressure pinned per level | `B_instrumentation/beltrami_pinned/report.json`, `beltrami_pinned_kmax/` | full column rank 7984/7984; t=1 pressure error 0.7515% (1.83% in the pin gauge) |
| B4 Bratu | `B_instrumentation/four_method_tables.csv`, `four_method_jobs/bratu_*/models/*/history.csv` | 28 rows (P=25/100/225 GPU, P=225 CPU), no failures |
| B6 basis study | `B_instrumentation/basis_study/table3_basis_study.csv`, `runs/` | 9 bases |
| Component C | `C_oversampling/results/oversampling.csv`, `C_oversampling/figures/`, `runs/` | 196 runs (4 configurations x 7 ratios x CGL, paper, 5 random seeds) |

## Check B1's violations

All 15 are the same as in the local validation of 2026-09-23 (`DECISIONS.md`,
"Check B1"), with the same values:

- Seven BL-gravity rows (iterations at every P, kappa at P=64 and 1024, rank
  at P=1024): the cos_fourier basis and retargeted losses are a deliberate
  deviation (`note` column).
- Iterations one off at Bratu P=225 (4 vs 3), BL P=64 (10 vs 11) and
  Kovasznay P=300 (9 vs 10).
- Kovasznay pressure error 2.6-7x smaller than Table 9 at P=675, 1200, 1875
  (see the Kovasznay entry in `DECISIONS.md`: reported as an improvement).
- Darcy S3 Darcy-x MSE 3.0e-3 vs 1.97e-1 (S1, S2 and SPE10 match).
- BL P=1024 numerical rank 936 vs 934.

## Four-method Bratu rows (B4)

Stopping reasons as classified by `lilq.four_method_log.stopping_fields`.
At P=25 every method reaches its target. At P=100 LiL-N reaches its target
(1.0e-4) and all six NiL runs stop on `optimizer_stall` with losses of
7.5e-4 to 6.1e-3. At P=225 no method reaches its target on either device:
runs end on `iteration_cap` (10,000) or `optimizer_stall`.

## Loading the trained models

`lilq.saved_models` reloads every model (`load_solution`, `load_network`,
`load_f1`, `load_f2`). All 301 were reloaded locally after download. The
reloaded A1 network gives eps_u = 5.798993885675724e-5 (recorded
5.798993885675638e-5), and every Kovasznay solution gives the velocity error
in `runs_index.csv`. The 9 B6 solutions (`basis_study/runs/*/solution.pt`)
store a `ComparisonConfig` defined in
`experiments/run_burgers_basis_comparison.py`, which ran as `__main__`, so
they load only with that class made visible first:

```python
import __main__, sys
sys.path.insert(0, 'experiments')
import run_burgers_basis_comparison as m
__main__.ComparisonConfig = m.ComparisonConfig
from lilq.saved_models import load_solution
solution = load_solution('results/wave1/B_instrumentation/basis_study/runs/sin_cheb')
```

No wave 2 or 3 script defines a class of its own, so their saved models do
not need this.

## Before wave 1 ran

The first submission, at `e69dbf7`, was refused by Grace for `--mem=0` (two
jobs cancelled while pending; nothing ran). `111dddc` fixed that. Its
preflight (job 19898134) failed because one test file imported another,
which Grace's Python resolved differently; `8a3f5f7` fixed that. Both are in
`DECISIONS.md`, and wave 1 ran in full at `8a3f5f7`.

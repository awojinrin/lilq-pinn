# Wave 4 results (Computational Package 1: Component A option 3, clean timing, B10, the follow-up items, and the package assembly)

Wave 4 ran on TAMU Grace on 2 October 2026 at commit `17b3539`, the commit
this branch starts from. The branch's own commit adds only `results/wave4/`.
`COMMIT` holds the commit and the source-tree hash that every job checked.

**What the folder holds:**
- **Everything the wave wrote** except collocation files (`*.npz`), with the
  trained models: 122 `*.pt`, 78 MB.
  - 91 of them are wave 2's Component A models, which `submit_wave4.sh`
    copied in before the wave (search, screening runs, F1's finalists).
  - They are byte for byte those on branch `wave2-results`.
- **`slurm_logs/`:** the jobs' Slurm logs, the finalize job's included.
- **All 406 files of `wave4_report.tar.gz`,** byte for byte.
- **`package1_tables/`:** what `90_finalize` derived when it assembled
  `package1` from waves 1-4.
  - `WAVES.json`: the wave commits, the 226 overrides, the 26 runs marked
    superseded, and `status`.
  - The merged B tables and the Section 4.6 results.
  - The residual-band figures.
  - B9's tables with their allocation filled in.

  The rest of `package1` is copies of the waves' own files, which are on
  the four results branches.
- **The package is provisional** (`WAVES.json`'s `status`) until the advisor
  has reviewed this report.

Waves 1-3 are on branches `wave1-results`, `wave2-results` and `wave3-results`.

## 1. Jobs and SUs

**Every job completed with exit code 0** (`sacct.txt`). The report job lists
itself as RUNNING, because it read `sacct` before finishing.

**Charged:** `myproject` shows 3,866.23 SU for wave 4 (13,744.99 used,
against 9,878.76 before). The balance is 6,255.01.

**Computed:** `su_per_job.csv`, at Grace's rates (`wave_report.su_rate`):
- one SU per core-hour, plus 72 per A100-hour;
- 192 per hour for a job holding a whole A100 node.

These rates reproduced wave 3's charge exactly. Here they give 3,812 with
the finalize job, 1.4% under the charge. Grace reports only the account
total, so the gap cannot be traced to a job.

| Job | What | Allocation | Elapsed | SU/h | SU |
|---|---|---|---|---|---|
| 19931246 | 00 preflight (tests, smoke runs) | 8 cores + 1 A100 | 00:09:25 | 80 | 12.6 |
| 19931249 | 29 gate: search, A1, A2, F2 Jacobian | 8 cores + 1 A100 | 01:06:12 | 80 | 88.3 |
| 19931252 | 34 F1 re-pick (no training) | 8 cores + 1 A100 | 00:02:01 | 80 | 2.7 |
| 19931255 | 35 F2 full stage: 3 x 5 seeds x 60 min | 48 cores + 1 A100 (whole node) | 15:29:31 | 192 | 2,974.5 |
| 19931259_0 | 32 CPU reruns, F1_18 | 48 cores, exclusive | 05:07:05 | 48 | 245.7 |
| 19931262_1 | 32 CPU reruns, F2_05 | 48 cores, exclusive | 05:08:47 | 48 | 247.0 |
| 19931265 | 33 float32 run, F1_18 | 48 cores + 1 A100 (whole node) | 01:01:41 | 192 | 197.4 |
| 19931268 | 13a clean timing, CPU | 48 cores, exclusive | 00:23:54 | 48 | 19.1 |
| 19931277 | 13b clean timing, GPU (after 13a) | 48 cores + 1 A100 (whole node) | 00:02:53 | 192 | 9.2 |
| 19931280 | 43 Table 3 | 24 cores | 00:04:11 | 24 | 1.7 |
| 19931288 | 44 B10 | 24 cores | 00:22:22 | 24 | 8.9 |
| 19931289 | 45 B8 reruns, K_max = 60 | 24 cores | 00:01:12 | 24 | 0.5 |
| 19931291 | 91 wave report | 24 cores | 00:02:49 | 24 | 1.1 |
| 19931293 | 90 finalize (package1) | 24 cores | 00:07:51 | 24 | 3.1 |
| | **Total** | | | | **3,812** |

## 2. Component A

**The gate passed on this commit** (`A_calibration/checks/`):
- **A1:** eps_u = 5.78e-5 against the target 1e-3.
- **A2:** bit-identical.
- **The F2 Jacobian check:** maximum relative error 3.8e-9 on the A100.

**Selection** (`screening/F1_selection.json`, `F2_selection.json`).
Validation residuals: `*/validation.json` for wave 2's runs; `run.json`'s
`val_residual` for wave 4's.
- **F1:** wave 2's finalists F1_04, F1_15 and F1_18 are kept, as the advisor
  decided. The validation top three of the screening runs would be F1_19,
  F1_01 and F1_08 (`top_by_validation`; `top_kept_matches_validation`:
  false).
- **F2:** F2_10, F2_05 and F2_14, pinned. The recomputed validation top three
  is the same (`top_kept_matches_validation`: true).

**Representatives** (`full/*_representative.json`): the lowest median
validation residual. No validation failed.

| Finalist | Median validation residual | eps_u, 5 seeds | Median eps_u | Median eps_p (mean-free) |
|---|---|---|---|---|
| F1_04 | 3.82 | 6.2e-2 5.7e-2 1.3e-1 8.5e-2 7.4e-2 | 7.4e-2 | 9.7e-1 |
| F1_15 | 2.26e-5 | 7.4e-5 1.2e-4 6.3e-5 8.7e-5 1.1e-4 | 8.7e-5 | 2.7e-4 |
| **F1_18** | **3.44e-6** | 4.5e-5 2.4e-5 4.4e-5 4.2e-5 3.1e-5 | **4.2e-5** | 2.2e-4 |
| F2_10 | 8.69e-15 | 1.5e-10 7.7e-10 2.3e-10 2.9e-10 9.0e-11 | 2.3e-10 | 3.2e-9 |
| **F2_05** | **1.03e-15** | 1.7e-10 3.1e-10 1.6e-9 1.4e-10 9.1e-11 | **1.7e-10** | 1.2e-9 |
| F2_14 | 7.36e-15 | 5.8e-10 5.4e-9 1.2e-10 1.6e-9 1.0e-10 | 5.8e-10 | 2.7e-9 |

**Comparison with wave 2's representatives:**
- F1_04 had a median eps_u of 7e-2.
- F2_16 had 8e-3; it has fewer rows than parameters, and ended on
  `mu_overflow` in screening.

**How the runs ended:**
- All 15 F2 full runs used their whole hour (`run_endings.csv`).
- F2_05 took about 4,600 LM steps per hour on the A100, F2_10 about 1,500
  and F2_14 about 3,400.

**CPU reruns** (`full_cpu/`, one exclusive CPU node, 60 min each):
- **F1_18:** eps_u 6.5e-5, 5.9e-5, 4.3e-5, 9.0e-5 and 3.3e-5; median 5.9e-5
  against 4.2e-5 on the GPU. About 10,600 steps per run.
- **F2_05:** eps_u 9.4e-10, 3.0e-9, 1.3e-8, 2.0e-9 and 1.8e-9; median 2.0e-9
  against 1.7e-10 on the GPU. About 310 LM steps per run against about
  4,600 on the A100.

**float32 run** (`float32/F1_18_s0`: Adam in float32, then L-BFGS in
float64): eps_u 4.54e-5, against 4.54e-5 for the float64 run of seed 0.

**Thresholds** (`representative_thresholds.csv`: first logged time at which
eps_u reaches 1e-4, 1e-6, 1e-8 and 1e-9):
- **F2_05, GPU:**
  - 1e-4: 8-12 s;
  - 1e-6: 16-22 s;
  - 1e-8: 74-502 s;
  - 1e-9: 92-547 s for four seeds. Seed 2 never reaches it.
- **F2_05, CPU:**
  - 1e-4: 122-196 s;
  - 1e-6: 243-351 s;
  - 1e-8: 1,209-1,895 s for four seeds;
  - 1e-9: 3,524 s for seed 0 only.
- **F1_18:** reaches 1e-4 at 306-517 s on the GPU and 2,119-2,977 s on the
  CPU, and never reaches 1e-6.

## 3. Section 4.6: LiL-Q against the representatives

`package1_tables/A_calibration/results/kovasznay_comparison.csv` and
`time_to_accuracy.csv`, with the figure in `package1_tables/A_calibration/figures/`.
- **LiL-Q:** one run per size. Its time is the clean time (`time_source`:
  clean), with the logged time beside it.
- **The baselines:** the medians over five seeds. "Time to accuracy" is the
  first logged time at which a baseline run reaches LiL-Q's eps_u at that
  P.

| LiL-Q P | LiL-Q eps_u | LiL-Q GPU (clean) | LiL-Q CPU (clean) | F2_05 GPU: reached, median time | F2_05 CPU | F1_18 GPU | F1_18 CPU |
|---|---|---|---|---|---|---|---|
| 75 | 3.8e-1 | 0.043 s | 0.056 s | 5/5, 0.7 s | 5/5, 11 s | 5/5, 11 s | 5/5, 67 s |
| 300 | 2.9e-2 | 0.059 s | 0.29 s | 5/5, 2.9 s | 5/5, 44 s | 5/5, 21 s | 5/5, 144 s |
| 675 | 1.2e-5 | 0.15 s | 0.64 s | 5/5, 12 s | 5/5, 188 s | 0/5 | 0/5 |
| 1,200 | 7.4e-9 | 0.39 s | 1.67 s | 5/5, 141 s (83-1,022) | 4/5, 1,957 s | 0/5 | 0/5 |
| 1,875 | 7.0e-13 | 0.86 s | 4.33 s | 0/5 | 0/5 | 0/5 | 0/5 |

## 4. Clean timing

`B_instrumentation/clean_timing/`:
- `clean_timing.csv`;
- `clean_vs_logged.csv`, from the report (the reply on wave 3, Section 4);
- each job's provenance, in `provenance_cpu/` and `provenance_cuda/`.

**The runs:**
- Every quoted LiL-Q time had a warm-up, then a timed run with diagnostics
  off; the median of five for runs under 1 s.
- K_max = 60 throughout.
- The CPU job ran first, then the GPU job.
- 42 rows, no failures.
- Every clean run took exactly as many iterations as its logged run.

**Clean against logged:**
- **Within 0.87-1.09** for every row but one: bratu, burgers, viscous and
  gravity BL, elasticity (both quantities), Kovasznay on the CPU, and Darcy.
- **Kovasznay on the GPU** is 0.75-0.98, lowest at P = 1,875 (0.75). That is
  the cost of the per-iteration diagnostics on the GPU.
- **Beltrami:**
  - unpinned: clean 282.5 s, warm-up 246.7 s, logged 302.7 s (wave 2);
  - pinned: clean 267.0 s, warm-up 286.3 s, logged 266.9 s (wave 1).

  The wave 2 gap between the logged run (303 s) and its warm-up (227 s) is
  within the run-to-run spread of these 4-5 minute solves on Grace, about
  250-300 s.
- **The exception: gravity BL P = 256,** `training_time`:
  - clean 3.169 s, from a single timed run (it was over 1 s, so no repeats);
  - its warm-up took 0.725 s, the logged run 0.739 s, all 12 iterations.

  The timed run was slowed by something outside the solve. This value
  enters `four_method_lilq.csv` as well. See the report note.

## 5. B8: the three LiL-Q reruns with K_max = 60

`B_instrumentation/b8_jobs/lilq_kmax60/` and `b8_kmax60_vs_wave3.csv`
(the merged table: `package1_tables/B_instrumentation/b8_initial_guess.csv`,
131 rows, K_max for every LiL-Q row).

| Case | Guess | P | Wave 3 (K_max 20) | Wave 4 (K_max 60) | Target |
|---|---|---|---|---|---|
| gravity | initial condition | 256 | 20 iterations, cap, loss 0.0988 | 23, target, loss 0.0749 | 0.075 |
| gravity | initial condition | 576 | 20, cap, 0.1255 | 23, target, 0.0434 | 0.045 |
| gravity | zero | 64 | 20, cap, 0.9507 | 43, target, 0.2367 | 0.24 |

## 6. B10: Kovasznay on CGL grids with Clenshaw-Curtis weights

`B_instrumentation/b10/`: `b10.csv`, `b10_vs_paper_grid.csv` and
`README.md`, with each run's `iterations.csv` and `run.json`.
- **Runs:** 16, all on CGL grids. Every paper pass reached its tolerance in
  6-9 iterations, and every system had full rank.
- **Boundary weight:** `bc_weight_total` gives the total weight of one
  velocity component's boundary rows: 1 lambda_bc with Clenshaw-Curtis
  weights, 4 with equal weights (as on the paper grid).

| P | Weights | N/P = 5: eps_u | N/P = 10: eps_u | kappa (retained) | Paper grid eps_u | Ratio |
|---|---|---|---|---|---|---|
| 300 | Clenshaw-Curtis | 2.341e-2 | 2.341e-2 | 6.3e2 | 2.90e-2 | 0.81 |
| 1,200 | Clenshaw-Curtis | 6.576e-9 | 6.576e-9 | 1.0e4 | 7.36e-9 | 0.89 |
| 1,875 | Clenshaw-Curtis | 5.80e-14 | 5.81e-14 | 3.1e4 | 7.00e-13 | 0.083 |
| 1,875 | equal | 5.87e-14 | 5.86e-14 | 4.4e4 / 4.1e4 | 7.00e-13 | 0.084 |

**N/P = 5 and 10 agree** to 4-5 significant digits. With Clenshaw-Curtis
weights on CGL points, the discrete norm is already an accurate quadrature
at N/P = 5.

**At P = 1,875, equal weights on the same CGL grid do as well**, so the gain
over the paper grid there comes from the points, not the weights.

The K_max passes (60 iterations, zero tolerance) end at the same errors.

## 7. Table 3 (B6), rerun whole with the default-initialization ELM row

`B_instrumentation/basis_study/table3_basis_study.csv`:
- **`eps_u`:** the Burgers residual mean square on the 201 x 201 test grid.
- **The two kappas:** `kappa_raw` and `kappa_retained`.
- **Rank:** `num_rank_svd`, and `num_rank_gelsy`, the rank of LAPACK's
  column-pivoted QR (the CPU solve uses `gelsy`).
- **`num_rank_qr` is empty in every row.** It is filled only on the GPU's QR
  path. The pivoted-QR rank is `num_rank_gelsy`.

| Basis | final ‖R‖²_h | eps_u | SVD rank (of 625) | kappa_raw | kappa_retained |
|---|---|---|---|---|---|
| cheb_cheb | 7.8e-5 | 9.0e-5 | 625 | 2.5e3 | 2.5e3 |
| sin_cheb | 4.9e-9 | 5.0e-9 | 625 | 3.4e2 | 3.4e2 |
| sin_sin | 4.98 | 0 (by construction) | 625 | 5.5e1 | 5.5e1 |
| cos_cheb | 1.7e-1 | 1.3e-1 | 625 | 3.4e2 | 3.4e2 |
| augsin_cheb | 3.1e-8 | 3.2e-8 | 625 | 5.2e2 | 5.2e2 |
| fourier_cheb | 1.3e-5 | 1.3e-5 | 625 | 4.3e9 | 4.3e9 |
| fourier_fourier | 1.3e-5 | 1.3e-5 | 610 | 2.2e16 | 6.0e11 |
| elm (Xavier) | 4.9e-2 | 4.4e-2 | 43 | 1.3e18 | 7.2e11 |
| **elm_default** | **4.4e-4** | **4.4e-4** | **176** | 7.3e17 | 6.6e11 |
| sin_fourier | 4.9e-9 | 5.0e-9 | 625 | 9.6e8 | 9.6e8 |

## 8. The package assembly (`90_finalize`)

`slurm_logs/lilq-finalize.19931293.out`:
- **Assembly:** `package1` was assembled from the four waves by copying
  (3,391 files), with each wave's commit in `WAVES.json`.
- **Overrides:** 226 files are taken from a later wave. They are listed in
  `WAVES.json`.
- **Superseded runs:** 26 run folders are marked superseded
  (`SUPERSEDED.json` in each, and listed in `WAVES.json`):
  - wave 2's F2_03, F2_16 and F2_17 full runs;
  - its F1_04 and F2_16 CPU reruns;
  - its F1_04 float32 run.

**The tables:**
- **Check B1:** 15 violations, the same as in every wave. 7 are gravity BL,
  by design, and 8 were recorded before.
- **Section 3.3:** 63 runs, 0 failed.
- **Four-method:** 140 rows and 54 stall controls, 0 failures.
- **B8:** 131 rows: wave 3's 128 and the three reruns.
- **B9:** 28 rows, with the allocation filled in for 36 table rows. It reads
  "8 cores + 1 x NVIDIA A100-PCIE-40GB, shared".
- **LiL-Q's four-method rows** (`four_method_lilq.csv`): 16 rows from the
  clean times. Every clean run's iterations equal its logged pass's; a
  mismatch would have failed the job.
- **Section 4.6:** on the clean LiL-Q times.

**Wave 4's own check B1** (`B_instrumentation/reproduction_check.csv`)
shows 2 violations and 137 missing. Wave 4 reran only gravity BL P = 64
among the Section 3.3 runs. Its 2 violations are that run's iterations and
kappa (43 and 103 against the paper's 9 and 7,800): the basis change
recorded in `DECISIONS.md`.

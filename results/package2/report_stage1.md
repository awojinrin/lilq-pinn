# Computational Package 2: Stage 1 report

**To:** R. Younis. **From:** G. Awojinrin. **Date:** 4 October 2026.

**Code:** `awojinrin/lilq-pinn`, release tag `v2.0.0` (commit 5339f60). `main`
and `v3-dev` are both at 5339f60.

**Results:** `package2_results/` (Section 12.1 layout), attached.

## Summary

Everything in Stage 1 is done: Part 1 (the release, gate G1) and the
gating jobs of Section 3. The cost was 5.3 Grace SU.

**For your reply:**
1. **The Kovasznay baseline decision** (Section 4.2). The P_N–P_{N−2}
   square system has full rank with the corner pin at p_d = 10 and 15.
   **I recommend releasing the Kovasznay part of item 1.**
2. **The release of Stage 2**, under `su_plan.csv`: about 700 Grace SU
   expected (1,548 requested) and about 660 FASTER SU.
3. **Three manuscript corrections** found by the release check (Section
   1.2): the B.5 four-method P range, four rounding slips in Table 7, and
   three stale entries in Table 14.
4. **One question:** a public archive for `package1` (Section 8).

## 1. Part 1: the release (gate G1)

### 1.1 Appendix B.5 check (`G1_release/b5_check.md`)

Every entry of `tab:run_settings` was compared with what the runs of
`package1` used. The evidence was read from each run's `run.json`, the
package tables and the saved models. The code was not changed. Result:
**one discrepancy and six clarifications**; every other entry matches.

- **Discrepancy: the four-method row, P "25–1,051".** The range combines
  LiL's smallest P (25) with NiL's largest parameter count. The NiL
  networks have 37–1,051 parameters: at P = 25, h = 4 and h² + 5h + 1 =
  37. Suggested entry: "LiL: 25–1,024; NiL: 37–1,051 parameters".
- **Clarifications** (none is an error in the table):
  - BL N/P is 9.75–9.81 ("9.8" is right as a rounded value).
  - Every random tensor grid uses collocation seed 42; the caption could
    say so.
  - The paper passes recorded K_max = 25/20/50. No result depends on it:
    every pass stops before its cap, except gravity BL P = 64, whose
    K_max = 60 rerun is the one in `package1`.
  - The Beltrami row is the eight-pin run (N = 21,816).
  - Darcy NiL also ran in float32 (B9); the table does not state the
    precision.
  - B10 also holds the four equal-weight runs used for the comparison of
    weights.

### 1.2 Tag, `RELEASE.md` and the regeneration check

**The tag.** `v3-dev` was fast-forwarded into `main` and tagged `v2.0.0`.
I used a version number rather than the suggested name: releases are not
named after a journal. The arXiv version's code keeps its tag,
`v1.0-manuscript`.

**`RELEASE.md`** maps every table and figure of Section 6 and Appendix B,
by its printed number (Tables 2–17 and B.1–B.6, Figures 3–23), to the
script that regenerates it, the files it reads, and the script that wrote
those files.

**The regeneration check.** The manuscript's tables were typed from our
CSVs, so two new scripts regenerate everything and compare it with your
`main.tex` and figure files.
- **`experiments/release_tables.py`** rebuilds 19 tables from `package1`
  (medians over seeds, stopping markers and ratios included). It compares
  every cell with `main.tex`, within half a unit of the last printed
  digit.
- **`experiments/manuscript_scripts.py`** runs your manuscript folder's
  scripts (`our_scripts/`), unchanged except for their input paths. It
  renders each figure beside the manuscript's file and compares the
  screening tables row by row.

| Check | Result |
|---|---|
| 19 tables, 2,578 cells | all agree except 4 cells of Table 7 (below) |
| 20 data figures, your scripts on the published wave branches | all pixel-identical to the manuscript's files |
| the same on `package1` | 18 of 20 identical. `fig_conv.py` asserts that gravity BL P = 64's paper pass is wave 2's K_max = 20 file; `package1` holds wave 4's K_max = 60 rerun, whose `norm_R_h` agrees with the curve the script builds to 1.1e-16. |
| screening tables B.4, B.5 (`gen_screening.py`) | 48 of 48 rows identical |
| Figure 20 (permeability) | pixel-identical; its script is now in the repository (`scripts/plot_permeability_grid.py`) |
| numbers in the text, traced by hand | Section 6.6 (δ_P and the ratios, `delta.py`), the Section 6.11 held-out evaluation, the B.2 caption counts and B.4: all match |

The regenerated CSVs and the comparison records are in
`G1_release/tables/` and `G1_release/manuscript_scripts*/`.

**Manuscript corrections found by the check:**

1. **Table 7 (`tab:bl_reference_errors`): four rounding slips.** The
   values equal those of your `A1_bl_lilq_errors.csv`; the printed
   roundings are wrong.

   | Case | P | Column | Value | Printed | Should be |
   |---|---|---|---|---|---|
   | N_g = 0 | 576 | E_S (stop) | 9.0458e-4 | 9.1e-4 | 9.0e-4 |
   | N_g = 0 | 576 | min E_S (k) | 9.0458e-4 (4) | 9.1e-4 (4) | 9.0e-4 (4) |
   | N_g = −5 | 1,024 | min E_S (k) | 1.1477e-2 (54) | 1.2e-2 (54) | 1.1e-2 (54) |
   | N_g = −5 | 1,024 | E_S (k = 60) | 1.1477e-2 | 1.2e-2 | 1.1e-2 |

2. **Table 14 (`tab:darcy_training`): three NiL entries are stale.** They
   describe the pre-GitHub notebook's per-field recipe, not the single
   formulation you approved before wave 3, which produced Table 15's NiL
   rows (`problems/darcy.py`).

   | Entry | Table 14 | The networks behind Table 15 |
   |---|---|---|
   | Activation | tanh / SiLU | SiLU |
   | Output (h*) | sigmoid | linear (`P_mid + P_half · net`) |
   | Optimizer | Adam, StepLR | Adam, cosine annealing to 1e-5, gradient clipping at norm 1 |

   - **Not in the table:** the loss weights (50 for the PDE, 20 for the
     boundaries).
   - **Correct as printed:** the size (2 × 32 per field, 3,555
     parameters), 150,000 iterations and float64.

### 1.3 `package1` reassembled at the tag (`G1_release/package1_v2.0.0_check.md`)

**The job.**
- **What ran:** Grace job 19963045, 7 min 26 s, about 3 SU. It rebuilt
  `package1` by copying the four wave folders with the tag's own
  `90_finalize` steps. Nothing was rerun.
- **Where:** `results/package1_v2.0.0/package1` (413 MB), backed up to
  Grace home.
- **Its `provenance.json`** records the release, commit 5339f60, the wave
  commits (8a3f5f7, 905e58c, 905e58c, 17b3539) and status final.

**The comparison** with the final package assembled on the laptop was file
by file (SHA-256, then by content where they differed). Every result is
identical or agrees to round-off:
- 3,282 files byte-identical, including every log, table and model;
- 28 identical apart from line endings;
- the two run indexes hold the same rows in another order;
- the 11 recomputed reference solutions agree to 2e-16 – 1.4e-13;
- the package-side figures differ only in font: Grace has no Times New
  Roman.

### 1.4 `kovasznay_comparison.csv` (`G1_release/kovasznay_comparison.csv`)

The F1 medians are now taken over the final errors in `run.json`: `eps_u`
is 4.19e-5 on the GPU and 5.88e-5 on the CPU (it was 4.11e-5 and 5.66e-5).
`eps_v` and `eps_p_meanfree` are also final values; the CPU `eps_p` median
moves from 3.42e-4 to 3.45e-4. Table 11 already carries the corrected
values. The note is in `RELEASE.md`.

### 1.5 Stall controls

Confirmed from `four_method_controls.csv`
(`G1_release/stall_control_departures.csv`). Four of the six late-departing
controls depart at the original's last logged row before the stall. Two
depart 13–14 iterations before it:
- `bl_P1024_NiL-Q_s2_cuda` departs at 8,140; the stall is at 8,153.
- `burgers_P400_NiL-Q_s2_cuda` departs at 5,800; the stall is at 5,814.

Both run ids are recorded in `RELEASE.md`.

### 1.6 BL reference errors of the network formulations (optional; `G1_release/bl_reference_errors_networks.csv`)

All 70 saved models were evaluated on the 201 × 201 finite-difference
reference grid. `eps_ref_min` and `k_min` are `NA`, as in your reply:
final weights only, no reruns.

The table below gives the GPU runs; NiL entries are the median over seeds
0–2, with the range in brackets. LiL-Q is Table 7's error at the stop.

| Case | P | LiL-Q | LiL-N | NiL-N | NiL-Q |
|---|---|---|---|---|---|
| N_g = 0 | 64 | 9.9e-2 | 1.3e-1 | 4.8e-2 [2.6e-2–5.0e-2] | 1.0e-1 |
| | 256 | 3.8e-2 | 3.8e-2 | 1.4e-2 | 1.4e-2 |
| | 576 | 9.0e-4 | 7.1e-3 | 4.8e-3 | 4.2e-3 |
| | 1,024 | 2.0e-4 | 1.1e-3 | 1.0e-3 | 2.1e-3 |
| N_g = −5 | 64 | 2.0e-1 | 1.9e-1 | 1.9e-1 | 1.9e-1 |
| | 256 | 3.2e-2 | 4.7e-2 | 3.4e-2 | 4.8e-2 |
| | 576 | 2.1e-2 | 3.0e-2 | 1.6e-2 | 1.9e-2 |
| | 1,024 | 1.2e-2 | 1.1e-2 | 7.8e-3 | 1.1e-2 |

**Reading:**
- **At the two smallest viscous sizes,** the NiL networks' final errors are
  below LiL-Q's error at its stop.
- **At viscous P = 576 and 1,024,** LiL-Q's error is about 5–10 times
  lower.
- **In the gravity case** the four formulations are within a factor of
  about 2 of each other, and NiL-N is below LiL-Q at P = 576 and 1,024
  (1.6e-2 against 2.1e-2, and 7.8e-3 against 1.2e-2).

## 2. Kovasznay square-system pilot (Section 4.2, step 1; `P2_3_classical/kovasznay/pilot.md`)

**The system.** The P_N–P_{N−2} pair:
- velocity in the tensor Chebyshev space of p_d modes per direction,
  pressure in p_d − 2;
- the momentum and continuity rows at the (p_d − 2)² interior CGL points;
- the velocity data at the 4p_d − 4 boundary CGL points;
- the continuity row nearest (−0.5, −0.5) replaced by the pressure pin.

**The solve.** Newton from zero, stopping at relative coefficient change
< 1e-9, K_max = 60. Errors on the 301 × 401 grid.

| p_d | unknowns = equations | pin | rank | κ₂ (final) | κ₂ at zero | Newton iterations | E_u | E_v | E_p (mean-free) |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 264 | corner | 264 (full) | 1.8e4 | 4.8e3 | 10 | 2.3e-2 | 8.8e-2 | 3.8e-2 |
| 15 | 619 | corner | 619 (full) | 3.4e5 | 1.7e5 | 6 | 1.85e-5 | 1.16e-4 | 9.1e-5 |

**Result: full rank with the corner pin at both sizes**, so the fallback
is not needed. The interior pin was run for comparison: it changes only
the pressure gauge, and the velocity and mean-free pressure errors are
identical.

**A test of the Newton matrix.** `tests/test_square_kovasznay.py` checks
the Newton matrix against a finite-difference Jacobian of the nonlinear
residual.

**For scale:** at p_d = 15 the velocity space has 450 coefficients, and
E_u = 1.85e-5 against LiL-Q's 1.23e-5 at P = 675.

**Recommendation:** release the Kovasznay part of item 1 (p_d = 10, 15, 20
and 25).

## 3. Bratu square baseline, untimed check (Section 3, item 2; `P2_3_classical/check_c1.*`)

**The port.** Your pilot is ported as `baselines/square_chebyshev.py`. One
assembly path is shared by the square and least-squares variants.

**Check C1 passed.** On the pilot's 101 × 101 grid against the p = 48
reference:

| Method | p | Error (port) | `rows.json` | Relative difference |
|---|---|---|---|---|
| LS-hard-CGL-1.5 | 12 | 1.0827e-5 | 1.0827e-5 | 4e-12 |
| LS-hard-CGL-1.5 | 20 | 6.4719e-7 | 6.4719e-7 | 7e-11 |
| SQ-CGL | 14 | 1.2742e-5 | 1.2742e-5 | 3e-12 |
| SQ-CGL | 22 | 8.0343e-7 | 8.0343e-7 | 2e-11 |

**The sanity run.** LS-hard with N = P on the interior CGL points of the
12-point grid reproduces SQ-CGL at p = 12 to 1.2e-15.

**Check C2 passed.** The reference error (p = 48 against p = 64, on the
201 × 201 grid) is 3.6e-9. The smallest Bratu error item 1 will report
(LS-hard at p = 24) is 2.4e-7, a factor of about 67 larger; 10 is
required.

## 4. Scalar reference errors (Section 6.2; `P2_12_reference_errors/`)

**References** (`reference/`):
- **Bratu:** SQ-CGL at p = 48, with a reference error of 3.6e-9.
- **Burgers:** Cole–Hopf by adaptive quadrature. Two checks:
  - the integrator passes the Basdevant check: u_x(0, 1.6037/π) =
    −152.005162 at ν = 0.01/π, against the published −152.00516;
  - at ν = 0.1 it agrees with an independent series solution to 3.8e-15.

**The reruns.** The 8 reported LiL-Q configurations were rerun on a Grace
CPU node with the `package1` thread setting (48), with `eps_ref` logged at
every iteration (Grace job 19956475, 2.3 SU).

**Check C4 passed.** Every rerun repeats `package1`'s iterations, and
`norm_R_h` agrees **exactly** (relative difference 0.0) at every k.

**Reference errors.** `scalar_reference_errors.csv` has 78 rows: the 8
LiL-Q reruns and the 70 saved network models. The networks' `eps_ref_min`
and `k_min` are `NA`. The table gives GPU medians over seeds for the
networks, with LiL-Q's final error; LiL-Q's smallest error is the same as
its final except where shown.

| | P | LiL-Q final (smallest, at k) | LiL-N | NiL-N | NiL-Q |
|---|---|---|---|---|---|
| Bratu | 25 | 1.5e-1 | 2.0e-1 | 8.5e-2 | 1.3e-1 |
| | 100 | 3.5e-4 | 1.8e-3 | 7.1e-3 | 4.2e-3 |
| | 225 | 5.8e-5 | 2.4e-3 | 3.6e-3 | 1.5e-3 |
| Burgers | 25 | 1.8e-1 | 2.5e-1 | 1.3e-1 | 2.7e-1 |
| | 100 | 5.7e-3 | 5.3e-3 | 7.6e-3 | 9.0e-3 |
| | 225 | 2.2e-3 | 2.2e-3 | 9.8e-4 | 6.8e-4 |
| | 400 | 1.9e-4 (1.7e-4, k = 3) | 3.1e-4 | 7.0e-4 | 6.7e-4 |
| | 625 | 4.6e-5 (3.7e-5, k = 3) | 7.0e-4 | 6.4e-4 | 4.8e-4 |

**Two things to note:**
- **Where the networks win.** At the smallest sizes, and at Burgers P =
  100 and 225, some networks match or beat LiL-Q against the reference.
  At Burgers P = 225 the network medians are 2–3 times lower. At the
  larger sizes LiL-Q is ahead:
  - 5–20 times at Bratu P = 100, and 25–60 times at P = 225;
  - 2–4 times at Burgers P = 400, and 10–15 times at P = 625.
- **Burgers P = 400 and 625.** LiL-Q's smallest error is one iteration
  before the last. The last step lowers the residual, but the iterate
  converges to the discrete collocation solution, whose own error is the
  final value.

## 5. Elasticity active-coefficient counts (Section 9.2; `P2_6_elasticity_counts.csv`)

| P | u_x: coefficients above 1e-12 · max\|β\| | u_y | `gelsy` rank | SVD rank |
|---|---|---|---|---|
| 50 | 1 of 25 | 5 of 25 | 50 | 50 |
| 200 | 1 of 100 | 5 of 100 | 200 | 200 |
| 450 | 1 of 225 | 5 of 225 | 450 | 450 |
| 800 | 1 of 400 | 5 of 400 | 800 | 800 |
| 1,250 | 1 of 625 | 5 of 625 | 1,250 | 1,250 |

The manufactured solution lies in the span of 1 and 5 basis functions at
every P, and every system has full rank. This is consistent with errors at
round-off (Table 8).

## 6. `su_plan.csv` for Stage 2

**Rates** (Section 12.5, question 3):

| Cluster | Class | Rate | How it was obtained |
|---|---|---|---|
| Grace | timed CPU (whole CPU node, 48 cores) | 48 SU/h | job 19956475: 2.3 SU in 172 s |
| Grace | CPU (24 cores, 32 GB) | 24 SU/h | HPRC's published rule (effective cores) |
| Grace | timed GPU (one A100 with 48 cores) | 192 SU/h | waves 2–3 |
| FASTER | shared A100 (8 cores, 32 GB) | 137 SU/h (9 effective cores + 128 per A100-hour) | job 3173939: 9.48 SU in 249 s |
| FASTER | CPU (24 cores) | 24 SU/h | job 3172108: 31.07 SU in 4,660 s |

Both clusters' rates match HPRC's published rules: effective cores, plus
72 per A100-hour on Grace or 128 per A100-hour on FASTER.

**The plan** (one row per job in `su_plan.csv`). The estimates are scaled
from timings measured in Package 1:
- Beltrami: 282 s at P = 7,984, scaled as P³;
- Kovasznay: 4.3 s on the CPU and 0.86 s on the GPU at P = 1,875;
- BL: 61 s for K_max = 60 at P = 1,024;
- the ELM basis runs: 12 s each.

"Requested" is the walltime times the rate; the charge is for elapsed
time.

| Item | Jobs | Requested | Expected | Cap |
|---|---|---|---|---|
| 1 | Bratu; Kovasznay if released; Burgers optional | 144 | 72 | 400 |
| 2 | Bratu, Burgers; BL optional. Walltime covers every run hitting its 15-min cap. | 504 | 120 | 600 |
| 3 | certified grids, one job | 72 | 24 | 400 |
| 4 | ELM sweep and Kovasznay ELM | 48 | 24 | 200 |
| 5 | FASTER: array of 12 Darcy runs | 1,233 FASTER | ≈ 660 FASTER (at most 822) | 1,500 FASTER |
| 6 | ν-refinement and basis reruns | 144 | 48 | 300 |
| 7 | scaling: CPU (largest Beltrami in a job of its own, about 1.7 h per run, under the 300-SU skip threshold) and GPU | 624 | 408 | 900 |
| 8 | manufactured elasticity | 12 | 6 | 100 |
| **Grace total** | | **1,548** (1,260 without the optional and conditional jobs) | **≈ 700** | 2,950 |

**Grace balance:** about 6,200 SU, so the reserve is untouched.

**The least certain estimates:**
- the finite-difference BL references at ν = 0.01 (item 6, given 6 h);
- whether the largest Beltrami system fits on a 40 GB A100, which item 7
  reports either way.

## 7. Answers to the questions of Section 12.5

1. **Saved models.** Yes. The NiL-N, NiL-Q and LiL-N models of the
   four-method runs are in `B_instrumentation/four_method_jobs/*/models/`,
   and those of B8 in `B_instrumentation/b8_jobs/*/models/` (both in
   `package1`). They are final weights only. Per your reply, Sections 1.6
   and 4 use them for `eps_ref_final`, with `NA` for the iteration
   columns.
2. **Thread setting of the paper's LiL-Q CPU timings.** 48 threads: a whole
   Grace CPU node (exclusive, two Xeon Gold 6248R) with
   `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 48` and 48
   PyTorch threads, as recorded in each timing's `hardware.json`.
3. **SU rates.** See Section 6: Grace CPU node 48 SU/h; FASTER 137 SU/h
   for a shared A100 and 24 SU/h for 24 CPU cores.
4. **Calendar estimate for Stage 2.** About three weeks from your reply:
   - **Week 1:** the code and tests of items 1–4 and 8, and their jobs
     submitted. Item 1's Bratu code already exists.
   - **Week 2:** items 5–7, with the FASTER array and the scaling jobs.
   - **Week 3:** the analysis, figures and `report.md`.

## 8. Question

**A public archive for `package1`.**
- **Where it is now:** `RELEASE.md` names the results, but `package1`
  (413 MB with the models) lives on Grace scratch, which is not backed up.
  Copies are in Grace home and on my laptop.
- **What's public already:** the wave results without the models are on
  the `wave1..4-results` branches.
- **The proposal:** deposit `package1_v2.0.0` on Zenodo, which issues a
  DOI, and cite it in `RELEASE.md` and the Data and Code Availability
  statement. Shall I?

## Files (`package2_results/`)

- **`G1_release/`:**
  - `RELEASE.md`, `b5_check.md`, `b5_actual.csv`;
  - the corrected `kovasznay_comparison.csv`;
  - `bl_reference_errors_networks.csv` and `stall_control_departures.csv`;
  - `tables/` (the regenerated tables and `release_check.*`);
  - `manuscript_scripts/` and `manuscript_scripts_package1/` (your
    scripts' outputs and the comparisons);
  - `figures/`;
  - `package1_v2.0.0_check.md` and `manifest.csv`.
- **`P2_3_classical/`:** `check_c1.*`, `kovasznay/pilot.*`.
- **`P2_12_reference_errors/`:** the 8 reruns (`<benchmark>_P<P>/`),
  `check_c4.*` and `scalar_reference_errors.csv`.
- **`reference/`:** the Bratu and Burgers references, and their checks.
- **`P2_6_elasticity_counts.csv`.**
- **`su_plan.csv`.**

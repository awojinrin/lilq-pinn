# Decisions affecting results

A running log of implementation decisions in this codebase that could change
*computed* results -- hyperparameters, tolerances, grid densities, precision,
stopping criteria -- as distinct from ordinary bug fixes with a single
unambiguous correct behavior. Each entry says what was decided, why, exactly
what it affects, and how to revisit it later if circumstances change.

Cross-references `Post-JCP/Codebase_v3_Proposal.md` for the fuller
investigation behind each entry, and Section 10 (Q1-Q10) of
`Post-JCP/reponse_Package1_v2.md` for the original forensic findings.

Format: newest first.

See also `Phase0_Empirical_Verification.md` for the full pre-v2 / GitHub /
v3-dev three-way comparison run.

---

## 2026-10-04 -- Package 2, Stage 1, batch 5 on Grace: check C4 passed; the LiL-Q rows of the scalar reference-error table

**The job** (`p2s1_scalar_reruns`, job 19956475, node c531, exclusive, 48
threads) completed in 2 min 52 s, about 2.3 SU, at commit `750f0b6`
(`results/package2_stage1/COMMIT`).
- **Check C4:** every one of the 8 reruns has the same iterations as
  `package1`, and `norm_R_h` agrees **exactly** (relative difference 0.0) at
  every k. `eps_ref` is logged on every row.
- **The references** written on Grace match the laptop's to 1.4e-14 (Bratu)
  and 7.8e-16 (Burgers). Grace's Bratu reference error is 3.6e-9, and the
  Burgers Basdevant and series checks pass as on the laptop.

**The table.** `experiments/network_reference_errors.py --lilq-runs` adds the
LiL-Q rows to `scalar_reference_errors.csv`, with the final and smallest
`eps_ref` and the k of the smallest, read from the reruns' `iterations.csv`.
The network rows, evaluated against Grace's references, are unchanged
(within 1e-12).

**One thing to report:** at Burgers P = 400 and P = 625, LiL-Q's smallest
`eps_ref` comes one iteration before the last. The last iteration lowers the
residual but raises the error against the reference:
- P = 400: 1.67e-4 to 1.93e-4;
- P = 625: 3.67e-5 to 4.56e-5.

The iterate converges to the discrete collocation solution, whose own error
is the final value. At every other configuration, the final iterate has the
smallest error.

Tests: `tests/test_package2_stage1_release.py::test_lilq_rows_read_eps_ref_from_the_reruns`.

---

## 2026-10-04 -- Package 2, Stage 1, batch 5: the Grace job (references, scalar reruns with eps_ref, check C4) and the stage names

**eps_ref in the scalar solvers.**
- **The option:** `problems.bratu.run_lil_q` and `problems.burgers.run_lil_q`
  take `reference_npz`. When given, the iteration log's test errors gain
  `eps_ref` against that reference (`lilq.references.make_eps_ref_fn`).
- **The solve is unchanged:** it is a diagnostic, off the clock, and the
  solver never sees it. A test checks that the coefficients and `norm_R_h`
  are bit for bit those of a run without it.
- **The driver:** `component_b.py --reference-dir` passes the references to
  the Bratu and Burgers runs. Each summary gains `test_eps_ref`.

**Check C4 for the reruns** (`experiments/p2_c4_check.py`): `norm_R_h` of
each rerun against `package1`'s at every k, to 1e-10 relative, with the
iteration counts and `eps_ref` on every row.
- **Rehearsal on this laptop,** against Grace's wave 2 logs: every run
  passes, with the largest differences 8.2e-11 (Bratu P = 225) and 9.1e-11
  (Burgers P = 625).
- **On Grace,** with the 48-thread setting of the original runs, the
  agreement should be tighter.

**The stage names.** `LILQ_WAVE` also takes `p2s1` and `p2s2`, Package 2's
two stages.
- **Folders:** `env.sh` points them at `results/package2_stage1` and
  `results/package2_stage2`, each locked to its own commit like a wave.
- **`sbatch.sh`** accepts them.
- **The job scripts** are in `scripts/cluster/package2/`. They are not in a
  wave's submit script, so the "every job belongs to a wave" test does not
  apply to them.

**`scripts/cluster/package2/p2s1_scalar_reruns.slurm`** (timed-cpu,
exclusive, 48 threads, 30 min) runs:
- the two references, so that the stage is self-contained and locked;
- the 8 reruns;
- check C4 against `results/wave2/B_instrumentation`.

Requested 24 SU (walltime x 48 SU/h); about 2 SU expected
(`package2_results/su_plan.csv`).

Tests: `tests/test_package2_stage1_grace.py`.

---

## 2026-10-04 -- Package 2, Stage 1, batch 4: the Kovasznay square-system pilot (P_N - P_{N-2})

**The system** (Package 2, Section 4.2, step 1). `baselines/square_kovasznay.py`
is square spectral collocation of steady Navier-Stokes, Re = 40, on [-0.5, 1]
x [-0.5, 1.5], in the code's convective form.
- **Spaces:** velocity in the tensor Chebyshev space of p_d modes per
  direction, pressure in p_d - 2.
- **Rows:** the momentum and continuity equations at the (p_d - 2)^2
  interior CGL points, and the exact velocity at the 4 p_d - 4 boundary CGL
  points. That gives 3 p_d^2 - 4 p_d + 4 unknowns and equations.
- **The pin:** the continuity row at the interior point nearest (-0.5, -0.5)
  is replaced by a pressure pin, either at that corner or, as the fallback,
  at that interior point.
- **Newton:** from zero, stopping at relative coefficient change < 1e-9,
  K_max = 60, solved by LU.
- **Errors:** on the 301 x 401 test grid, as in the paper.
- **Rank and kappa_2:** by SVD of the Jacobian, rank threshold n eps
  sigma_max.

**Results** (`experiments/p2_3_classical.py kovasznay-pilot`;
`P2_3_classical/kovasznay/pilot.md`):

| p_d | unknowns | rank (corner pin) | kappa_2 (corner / interior pin) | Newton | eps_u | eps_v | eps_p, mean-free | eps_p, pin gauge (corner / interior) |
|---|---|---|---|---|---|---|---|---|
| 10 | 264 | 264 | 1.8e4 / 4.4e3 | 10 | 2.3e-2 | 8.8e-2 | 3.8e-2 | 0.92 / 0.18 |
| 15 | 619 | 619 | 3.4e5 / 1.0e5 | 6 | 1.85e-5 | 1.16e-4 | 9.1e-5 | 7.2e-3 / 2.1e-3 |

**Reading:**
- **The system has full rank with the corner pin at both sizes,** so the
  fallback is not needed. It is shown for comparison.
- **The pin changes only the pressure gauge:** the velocity and mean-free
  pressure errors are identical.
- **At p_d = 15** the velocity space has 450 coefficients against LiL-Q's
  675 at P = 675, and eps_u is 1.85e-5 against LiL-Q's 1.23e-5 there.
- **The decision** on the Stage 2 Kovasznay baseline is the advisor's, in
  his Stage-1 reply.

**Tests** (`tests/test_square_kovasznay.py`):
- the counts;
- **the Newton matrix against the finite-difference Jacobian of the
  nonlinear residual**, which checks the linearization itself;
- the boundary data and the pin at the solution;
- convergence with p_d;
- that the pin changes only the gauge.

---

## 2026-10-04 -- Package 2, Stage 1, batch 3: the Burgers reference (Cole-Hopf), the eps_ref column, the scalar network models' reference errors

**The Burgers reference** (Package 2, Section 6.2; check C4).
`lilq/references.py` gives the Cole-Hopf solution of the Section 6.3 problem
by adaptive quadrature of the two Cole-Hopf integrals (`quad_vec`, relative
tolerance 1e-13).
- **Overflow:** the exponent is shifted by its bound 1/(2 pi nu), so nothing
  overflows at small nu.
- **Range:** the integrals run over |eta| <= s sqrt(4c + 60) (s =
  sqrt(4 nu t), c = 1/(2 pi nu)), beyond which the integrands are below
  1e-26 of their peak.
- **Basdevant check:** u_x(0, 1.6037/pi) at nu = 0.01/pi is -152.0051616
  (published: -152.00516).
- **Independent check:** at nu = 0.1, the cosine series of the Cole-Hopf
  transform (modified Bessel coefficients, scaled) agrees to 4.3e-15 on the
  whole test grid. The series cannot check small nu: there its terms cancel
  to about exp(-2c) near x = 0.
- **The tests also show:** u(+-1, t) is 0 to 2e-16, the initial condition
  holds, and the PDE residual is below 1e-5 by central differences.
- **Output:** `experiments/p2_references.py burgers` writes
  `reference/burgers_cole_hopf.npz` (x, t, u on the 201 x 201 test grid of
  `problems.burgers`) and the checks.

**`eps_ref`** is appended to the iteration log's columns, after
`stall_rule_fires`, as Package 2's Section 12.1 asks: empty where there is no
reference. `lilq.references.make_eps_ref_fn(basis, npz)` gives the error for
an iteration logger's `test_error_fn`. The paper runs are wired to it in the
Grace batch that reruns them.

**The scalar network models** (Section 6.2):
`network_reference_errors.py --benchmarks bratu burgers` evaluates the saved
four-method Bratu and Burgers models (70) against the two references, into
`P2_12_reference_errors/scalar_reference_errors.csv`. As before, only final
models exist, so `eps_ref_min` and `k_min` are empty.

**Preview, not the deliverable:** LiL-Q's final reference errors from the
saved paper-pass solutions. The deliverable is the per-iteration `eps_ref`
of the Grace reruns.
- **Bratu:** 1.5e-1, 3.5e-4 and 5.8e-5 at P = 25, 100 and 225.
- **Burgers:** 1.8e-1, 5.7e-3, 2.2e-3, 1.9e-4 and 4.6e-5 at P = 25-625.
- **C2:** every one is far above the references' errors (Bratu 3.6e-9;
  Burgers about 1e-15).

Tests: `tests/test_references.py`, `tests/test_iteration_log.py`.

---

## 2026-10-04 -- Package 2, Stage 1, batch 2: the Bratu classical baseline ported, its reference, checks C1 and C2

**The port** (Package 2, Section 4.1). `baselines/square_chebyshev.py`
ports the advisor's pilot (`bratu_pilot.py`). Every method is one Newton
iteration through one `assemble`, and the methods differ only in:
- **the trial space:** tensor T_i(2x-1)T_j(2y-1), or the same times
  x(1-x)y(1-y);
- **the rows and their weights:**
  - square on the p x p CGL grid (LU);
  - interior CGL rows with Clenshaw-Curtis weights (`hard`);
  - weak boundary rows with equal mean-square weights and lambda_bc
    (`weakMS`), or with Clenshaw-Curtis weights (`weakCC`).

The iteration starts from zero, with K_max = 20, and stops at a relative
coefficient change < 1e-9. It records:
- per iteration: the residual norms, the coefficient change, chi, and the
  error against a reference;
- the assembly and solve times, kept apart and outside the diagnostics, for
  the Stage 2 timing;
- the plateau iterate (the first within 5% of the final error);
- kappa and the rank (SVD).

**The reference** (Section 6.2; check C2).
`experiments/p2_3_classical.py reference` writes
`reference/bratu_ref_p48.npz` and `bratu_ref_p64.npz`: the coefficients,
p, and the field on the 201 x 201 test grid.
- **Its error:** p = 48 against p = 64 is 3.6e-9 relative (max 2.2e-8);
  p = 64 against p = 80 is 6.9e-10.
- **The smallest Bratu error the package will report** is LS-hard at p =
  24, 2.4e-7; SQ-CGL at p = 24 is 5.1e-7.
- **So C2 holds by a factor of about 67** (10 required).

**Check C1** (`check-pilot`; Section 3, item 2). On the pilot's 101 x 101
error grid, against the same p = 48 reference:
- **The pilot's errors are reproduced to 1e-11 relative** (three digits
  required): LS-hard-CGL-1.5 1.0827e-5 (p = 12) and 6.4719e-7 (p = 20);
  SQ-CGL 1.2742e-5 (p = 14) and 8.0343e-7 (p = 22).
- **So are its N, its kappa and its plateau** (k = 4).
- **The sanity run:** LS-hard with N = P on the interior CGL points of the
  12-point grid (p = 10) reproduces SQ-CGL at p = 12 to 1.2e-15.
- The package reports errors on the 201 x 201 test grid. The pilot used
  101 x 101, so C1 is checked on the pilot's grid.

**The pilot's weak-boundary variant used lambda_bc = 1; Section 4.1 says
10.** The instructions win. The difference is small:
- **The errors:** 2.840e-4 against 2.819e-4 (p = 12), and 2.282e-5 against
  2.310e-5 (p = 20).
- **The ratio to SQ-CGL** at matched free coefficients stays 22x and 28x.
- **The port at lambda_bc = 1** reproduces the pilot's values exactly.

Tests: `tests/test_square_chebyshev.py`.

---

## 2026-10-04 -- Package 2, Stage 1, batch 1: the release checks (B.5, stall controls, network reference errors, elasticity counts)

Package 2 (the advisor's instructions of 4 October 2026), Section 2 items 3,
5 and 6, and item 9 (P2-6). Four scripts read the final `package1`. Each is
rerunnable and listed for `RELEASE.md`.

**`experiments/b5_check.py`:** what the runs used, per row of the
manuscript's Table `tab:run_settings` (Appendix B.5), from their records,
written to `b5_actual.csv`.
- **One discrepancy:** the four-method row's P range, "25-1051", mixes
  LiL's smallest P (25) with NiL's largest parameter count (1,051). The
  NiL networks have 37-1,051 parameters (h = 4 to 30, h^2 + 5h >= P).
- **Six clarifications:**
  - Buckley-Leverett's N/P is 9.75-9.81;
  - the random grids use collocation seed 42;
  - the paper passes record K_max 20/25/50 against the code's 60, which
    makes no difference: every one stops on its criterion first, and
    gravity BL P = 64 was rerun at 60;
  - the Beltrami row is the pinned run;
  - B9's NiL also ran in float32;
  - B10 also has equal-weight runs.
- **Report:** `b5_check.md`. The code was not changed.

**`experiments/stall_control_departures.py`:** the six controls whose
history departs from their original's.
- **Four depart at the original's last logged row before the stall.** The
  histories are logged every 10 iterations plus the final row; Burgers P =
  625's departure at 5,240 against its stall at 5,241 counts as at that
  row.
- **Two depart 13-14 iterations before the stall:**
  `bl_P1024_NiL-Q_s2_cuda` (8,140 against 8,153) and
  `burgers_P400_NiL-Q_s2_cuda` (5,800 against 5,814).

**`experiments/network_reference_errors.py`:** the saved four-method
models evaluated against a reference on its test grid.
- **For Buckley-Leverett:** both cases, every P on the GPU and P = 1,024
  on the CPU, NiL seeds 0-2. That is 70 models, against the
  finite-difference reference.
- **Limitation:** only the final models were saved, so `eps_ref_min` and
  `k_min` are left empty, with the reason.
- **Skipped:** the stall controls.
- **Bratu and Burgers** follow, with their references (Section 6.2).

**`experiments/elasticity_counts.py`:** per field, the coefficients above
1e-12 max|beta| in the saved solutions.
- u_x has 1 and u_y has 5 at every P from 50 to 1,250: the exact solution
  is in the span.
- Every system has full rank (`gelsy` and SVD).

**Where the outputs go:** to `Post-JCP/package2_results/`, outside the
repository: `G1_release/` and `P2_6_elasticity_counts.csv`. The final
`package1` assembled on 3 October (`package1-final`) now lives at
`Post-JCP/package1_local/package1`, with its models, rather than in a
temporary folder.

Tests: `tests/test_package2_stage1_release.py`.

---

## 2026-10-03 -- Field data for the solution-field figures, from the saved models (the advisor's reply to wave 4, Section 3)

The manuscript's solution-field figures are images from the superseded
code. `experiments/figure_data.py` evaluates the saved models of the
regenerated runs in a package on each figure's grid. Nothing is retrained
or re-solved.

**The output:** one `.npz` per figure, and `manifest.json`.
- **The figures:**

  | File | Runs |
  |---|---|
  | `bratu.npz` | P = 25 and 225 |
  | `burgers.npz` | P = 625 |
  | `buckley_leverett.npz` | viscous and gravity at P = 1,024, with the finite-difference reference |
  | `kovasznay.npz` | P = 1,875, with the exact solution, on the 301 x 401 evaluation grid |
  | `beltrami.npz` | the pinned run at t = 1, the six faces at 41 x 41, with the exact fields |
  | `darcy.npz` | finite-volume, LiL and NiL (seed 0, float64 and float32), at the cell centres |
  | `elasticity.npz` | P = 50, with the exact solution, 200 x 200 |

- **Every file holds:**
  - the coordinates;
  - each method's field (`[i, j]`, `indexing='ij'`);
  - the exact or reference field where there is one;
  - a `meta` JSON with each model's run folder, device and commit, and
    the checks.

**The runs:**
- **LiL-Q:** the Section 3.3 CPU paper passes.
- **LiL-N and NiL (seed 0):** the CPU four-method runs.
- **Missing CPU models:** where there is none, the GPU run's model is used,
  labelled in its source and listed in the manifest. That is the case only
  for Bratu at P = 25: wave 1's CPU four-method job ran P = 225 only, so
  P = 25's LiL-N, NiL-N and NiL-Q come from the GPU job.
- **Elasticity:** included because its image
  (`elasticity_fields_and_errors_N*.png`) exists only in
  `pre-v2-local-codebase/`, the superseded code.

**Evaluation and its round-off:**
- **How LiL fields are evaluated:** as tensor products
  (`lilq.test_errors.tensor_grid_values`), the code path of the runs'
  logged test errors.
- **What is recorded:** `meta['evaluation_roundoff']` gives each field's
  largest difference from the pointwise evaluation. It is below 1.3e-13
  everywhere except the viscous Buckley-Leverett LiL-Q solution at P =
  1,024.
- **Viscous Buckley-Leverett:** its coefficients reach 2.5e9, and the two
  evaluations differ by up to 1.7e-5. So its eps_u (1.958e-4) reproduces
  only to about 2e-5 relative: laptop against Grace, and tensor-product
  against pointwise evaluation.

**Checks against the logged errors,** on the final package:
- **Darcy:** delta_FV of every LiL and NiL model equals B9's table:
  exactly for LiL and NiL float64, to 1e-7 for float32. The re-evaluated
  LiL pressures equal the stored `P_lil` to 6e-12.
- **Kovasznay:** eps_u, eps_v and the mean-free eps_p agree to 3e-7
  relative.
- **Gravity Buckley-Leverett:** eps_u to 2e-9.
- **Elasticity:** both errors are of order 1e-16 (the basis reproduces the
  exact solution), so an error panel would show round-off only.

Tests: `tests/test_figure_data.py`.

---

## 2026-10-03 -- The final package: option B's clean times, the stall-based rule's tables, status final (the advisor's reply to wave 4, item 1)

The advisor adopted option B of the clean-timing fix: every quoted CPU time
is the median of three independent clean timings. These are wave 4's job
13a and the two replicates of job 14b, run on 3 October at 17b3539 on
exclusive CPU nodes. The package is to be reassembled as final.

**Option B in the code:**
- `clean_timing.compose_median` makes every CPU row of wave 4's table the
  median of its clean time and the replicates'.
  - It records the timings (`clean_time_timings_s`), their spread and the
    rule.
  - It refuses a replicate that lacks the row, failed, or took a different
    number of iterations (not the same solve).
  - GPU rows (13b: all under 1 s, medians of five) stay as they are.
- **`compose_package`** (`clean_timing.py --compose-package`) does this in
  an assembled package:
  - wave 4's table is kept as `clean_timing_single.csv`;
  - the composed table becomes `clean_timing.csv`, which Section 4.6 and
    the four-method LiL-Q rows read;
  - `WAVES.json` gains `clean_timing`: the rule, the replicate tables and
    every run's three timings.
- **`90_finalize` runs it right after the assembly,** before anything reads
  the clean times. The replicates reach the package as `fixes/`, copied
  from `results/wave4/fixes/`.
- **The two job scripts are kept with the code** in
  `scripts/cluster/after_wave4/`. They are not in a wave's submit script:
  they were submitted by hand, from outside the bundle, so that wave 4's
  lock held.

**The rest of the final package:**
- `assemble_package.py --final` (`LILQ_PACKAGE_FINAL=1` in `90_finalize`)
  sets `status` to final.
- `WAVES.json` records the commit of the assembling code
  (`assembled_by_commit`).
- `90_finalize` adds `B_instrumentation/stopping_rule/`, from
  `stopping_rule_table.py`:
  - Table 16 and the held-out evaluation;
  - **`stall_columns.csv`:** for every `iterations.csv` in the package, the
    first stall as logged (tau_r = 0.1), the first at tau_r = 0.01, and the
    n_s = 2 rule's step, returned iterate and censoring. These sit beside
    the tables' `first_stall_iteration` columns, which stay as logged.

**On Grace:** `env.sh` locks `results/wave$LILQ_WAVE` to the code's commit,
so `90_finalize` on code newer than wave 4's (17b3539) would fail the wave
4 lock. This final assembly is run on the downloaded waves, as the advisor
asked for no further Grace jobs. Grace's `package1` stays provisional until
it is reassembled there at the code release, under a lock root of its own.

Tests: `tests/test_wave4_reply.py`.

---

## 2026-10-03 -- The revised stall-based termination rule: tau_r = 0.01, n_s = 2 (the advisor's reply to wave 4, item 2(b))

**The manuscript's rule has changed.** Both conditions of the stall
detector,

    chi_k <= tau_chi   and   | ||R_lin^(k)|| - ||R_lin^(k-1)|| | <= tau_r ||R_lin^(k)||,

must now hold at n_s consecutive steps, with tau_chi = 0.1, tau_r = 0.01
and n_s = 2. The rule returns u^(k+1) at the first step k where this holds.
The code had tau_r = 0.1 and a single step.

**`lilq/instrumentation.py`:**
- `DEFAULT_TAU_R` = 0.01, and the new `DEFAULT_N_S` = 2.
- `stall_flags`: the single-step flag over a solve.
- `stall_rule_index`: the first k at which the flag has held at n_s
  consecutive steps k - n_s + 1, ..., k, all >= 1; `None` if the steps run
  out. n_s = 1 is the earlier rule.

**`lilq/iteration_log.py`:**
- `stall_flag` stays the single-step indicator, now at tau_r = 0.01.
- **New last column `stall_rule_fires`:** the flag has held at n_s
  consecutive steps, this one included. `n_s` is a tracker argument,
  default 2.
- **Logged only:** neither column stops a run, as before.
- **Existing logs keep their columns:** `stall_flag` at tau_r = 0.1, and
  no `stall_rule_fires`.

**`experiments/stopping_rule_table.py`** is new. The `stopping_rule_table.py`
the advisor mentions is his own analysis script, not in this repository.
It recomputes from existing logs, rerunning nothing, and evaluates the
rule on `chi` and `norm_Rlin_h`, not on the logged `stall_flag`.
- **Table 16:** from the 22 kmax passes. For each run and n_s it gives the
  returned iterate, its class (round-off limited or not), R (its
  ||R||_h over the pass's smallest), and E (its eps_u over the eps_u at the
  paper pass's stop). Over the runs it counts early stops (R > 1.02) and
  stops before the error minimum (eps_u > 2x the smallest).
- **The held-out evaluation:** on Component C's 192 runs without the
  paper-grid ones. How many fire, how many are censored (the single-step
  conditions are met only at the last logged step), the largest R, and the
  Kovasznay E statistics.

**It reproduces the advisor's `ns_eval.py`** (his `termfix/`) on the
downloaded data:
- **Row by row:** 66 Table 16 rows (22 runs x n_s = 1, 2, 3) and 576
  held-out rows, with no difference in returned iterate, class, R, E or
  outcome. This holds although this script reads package1's paper passes
  (wave 2's and wave 4's gravity BL P = 64), where `ns_eval.py` reads wave
  1's.
- **Table 16 summary:** early 0 and errstop 0 at every n_s; R max 1.0055,
  1.0023 and 1.0013 for n_s = 1, 2 and 3.
- **Held out, the manuscript's numbers:** for n_s = 2, 69 of 192 fire, 76
  are censored, and R max is 1.00039 (within 0.04% of the smallest). For
  n_s = 1: 145, 0, 1.00320. For n_s = 3: 59, 86, 1.00009.
- **Kovasznay, for n_s = 2:** 55 fire. E against each run's smallest eps_u
  is at most 1.746 (median 1.0039); against its last eps_u, 0.982-1.041.
  At P = 1,200, all 14 are of class C.

**Not changed: columns that read the logged single-step flag.** These
are, in existing results:
- `first_stall_iteration` in the run index, B8, B10, Component C and Table
  3;
- B8's `stall_flag_ever`.

They were computed with tau_r = 0.1 when the runs were logged. They are
left as they are, and a recomputation at tau_r = 0.01 can sit beside them
if the manuscript quotes any.

Tests:
- `tests/test_stopping_rule_table.py`: the rule against a copy of
  `ns_eval.py`'s `fire_ns` on 40 random sequences; persistence and
  censoring; the logged column; Table 16 and the held-out evaluation on
  synthetic logs; the command line.
- `tests/test_instrumentation.py`, `tests/test_iteration_log.py`.

---

## 2026-10-03 -- Section 4.6 reports the final errors (the advisor's reply to wave 4, item 2(a))

**The discrepancy.** For the baselines, `kovasznay_comparison.csv`'s
`eps_*_median`, `_min` and `_max` were "best within the budget" (entry of
2026-09-29, item 2): per seed, the minimum over the log rows with t <=
budget and, if the run did not overrun, its final value. The manuscript
quotes the final errors in each run's `run.json`.
- **F1_18 on the GPU:** two seeds' minimum logged eps_u lies below their
  final one (seed 1: 2.42e-5 against 2.43e-5; seed 3: 4.11e-5 against
  4.19e-5). The table's median was 4.11e-5 where the medians of the final
  errors are 4.19e-5 (GPU) and 5.88e-5 (CPU; the table had 5.66e-5).
- **The README and report note** gave the final values.
- **F2_05:** the minimum and the final coincide in all ten runs.

**The second problem.** The final value was dropped whenever the run's wall
time passed its budget. That happens to every run, by at most 0.2 s for F1
and 10.3 s for F2 (an LM step, or the last F1 evaluation, finishes after the
clock check). It changed no number here, because each overrunning run's last
logged row within the budget equals its final value.

**Now:**
- `eps_*_median`, `_min` and `_max` are of the final errors from `run.json`,
  regardless of the overrun.
- The old quantity is kept beside them, unchanged, as
  `eps_*_best_logged_median`, `_min` and `_max`.
- `n_final_past_budget` and `max_final_overrun_s` say how many seeds'
  final evaluation finished past the budget, and by how much at most.
- Time-to-accuracy is unchanged: it reads the log rows within the budget.

**On wave 4's data:**

| Representative | Device | Final median eps_u | Best logged median eps_u |
|---|---|---|---|
| F1_18 | GPU | 4.19e-5 | 4.11e-5 |
| F1_18 | CPU | 5.88e-5 | 5.66e-5 |
| F2_05 | GPU | 1.66e-10 | 1.66e-10 |
| F2_05 | CPU | 1.99e-9 | 1.99e-9 |

Mean-free eps_p on the CPU moves from 3.42e-4 to 3.45e-4. Every other
median is unchanged.

Tests: `tests/test_kovasznay_comparison.py`.

---

## 2026-10-01 -- The advisor's reply on wave 3, Section 4: the wave 4 report, and package1 provisional

The wave 4 report (`scripts/cluster/wave_report.py`, job 91) adds what the
reply's Section 4 asks for.

**B8's reruns beside wave 3.** `b8_reruns_beside_wave3` writes
`B_instrumentation/b8_kmax60_vs_wave3.csv`: each LiL-Q row rerun with K_max
= 60 (`b8_jobs/lilq_kmax60`) beside wave 3's row of the same case, guess and
P, with K_max (wave 3's from `iterations_cap`), iterations, final loss and
stopping reason. The report log lists them too.

**Clean against logged, every quoted LiL-Q time.** `clean_vs_logged`
writes `B_instrumentation/clean_timing/clean_vs_logged.csv`, one row per row
of `clean_timing.csv`, both Beltrami runs included:
- the clean and logged times, their ratio, and the warm-up run's time;
- both runs' iteration counts (the logged one from the logged run's
  `summary.json`, or the pinned run's `report.json`), whether they match,
  and both K_max.

The report log counts any mismatches and failed clean runs.

**SUs per job.** `su_per_job` writes `su_per_job.csv` from `sacct -P`
(`AllocTRES`, `ElapsedRaw`) at Grace's rates (`su_rate`):
- one SU per core-hour, plus 72 per A100-hour;
- 192 per hour for a job holding a whole A100 node (48 cores), which Grace
  charges for both of the node's A100s;
- 80 for a shared A100 (8 cores); 48 for a timed CPU node; 24 for a 24-core
  job.

These reproduce wave 3's charge exactly (1,019 SU) and wave 2's to 1% (8,402
of 8,498; the rest were jobs outside its `sacct` record). Grace reports only
the account's total (`myproject`), so the per-job figures are computed, and
labelled so. The report job is still running when it reads `sacct`, and the
finalize job comes after it, so neither is complete in the table. No rates
are applied on another cluster (`CLUSTER`).

**Rehearsal** on the audit's rehearsal data: waves 1-3 extracted fresh from
the downloaded results, and wave 4 from this laptop's outputs, including
Component A's whole chain at a budget of seconds.
- **B8:** the three reruns: gravity BL, initial-condition guess, P = 256
  and 576, from 20 iterations (cap) to 23 (target) each; zero guess, P =
  64, from 20 (cap) to 43 (target).
- **Clean against logged:** 42 rows, all iteration counts equal.

**package1 is provisional** (the reply, Section 4: "treat its package as
provisional until we have reviewed the wave 4 report"). `WAVES.json` gains
a `status`, provisional by default. `assemble_package.py --final` marks it
final once the advisor has approved.

Tests: `tests/test_wave3_reply.py`.

---

## 2026-10-01 -- The advisor's reply on wave 3: items 2.2, 2.5, 2.7 and 2.8

**2.2, the four-method LiL-Q rows (`four_method_tables.lilq_rows`).**
- **Iteration check:** each clean run's iteration count must equal its
  logged paper pass's (`summary.json`'s `iterations`). Otherwise they are
  not the same solve, and the function raises after writing the file, with
  every mismatch listed.
- **New columns:** `four_method_lilq.csv` gains `logged_iterations`,
  `logged_K_max` and `logged_summary`, beside the four-method schema. The
  shared schema (`FOUR_METHOD_CSV_COLUMNS`) is unchanged.
- **Warnings:** a missing `summary.json`, a failed clean run and a missing
  clean-timing table each give a warning. The last used to return no rows
  silently.
- **Finalize order:** `90_finalize` now runs this step last, so that a
  mismatch fails the job only after every other step is done.
- **On the rehearsal data** (entry "Wave 4 audit"), all 16 rows match. The
  logged passes' K_max is 20, 25, 50 or 60: wave 2's caps, which no run
  reached except gravity BL P = 64, rerun at 60.

**2.5, robustness and JSON.**
- **Component A validation:**
  - **A failure no longer ends the stage.** In a new run (`run_one`), the
    traceback goes into `run.json` as `val_error`, with no residual stored,
    so the residual is computed again when the runs are ranked.
  - **In the full stage** (`_validation` with a `failures` list), a failure
    ranks the run last (inf). The run's name goes into the representative
    file's `validation_failures`.
  - **If no finalist has a finite median, the stage raises** and writes no
    representative.
  - **The selection still raises on a failure,** so that a broken validation
    cannot silently pick the finalists.
- **Clean timing:**
  - **A failed run gets a row** with its traceback (`error`) and no time,
    and the job continues. It exits non-zero at the end.
  - **A resubmission reruns the failed run** and replaces its row.
  - The readers (`load_clean_times`, `lilq_rows`) skip failed rows.
- **Standard JSON:**
  - `validation.json_safe` writes inf and NaN as `null`, in `run.json`'s and
    `validation.json`'s validation fields, the selection files and the
    representative files.
  - `validation.as_residual` reads `null` back as inf.
  - No file in waves 1-3's downloaded results contains `Infinity` or `NaN`,
    so nothing written earlier needs changing.

**2.7, superseded runs.** `assemble_package.mark_superseded` writes a
`SUPERSEDED.json` (wave and reason) into each earlier-wave Component A run
in `package1` that wave 4's selection replaced, and lists them in
`WAVES.json` (`superseded`). Per family, from wave 4's own files:
- `full/` runs not among the finalists (the selection's `top`);
- `full_cpu/` and `float32/` runs not of the representative.

**Reassembly:** `assemble` now removes an earlier assembly first (an `out`
holding a `WAVES.json`) and builds it again from the waves. It refuses any
other non-empty `out`. Without this, a `package1` assembled by hard links
would have kept its identical files as links, and they would still have
written through to the waves (removing a link leaves the wave's file). It
also clears stale files and markers. On the rehearsal data it marks
F2_03, F2_16 and F2_17's full runs, and F1_04's CPU and float32 runs. F2_16's
CPU runs follow once wave 4 has written its F2 representative. That is the
advisor's list. The wave folders are not touched: the package holds copies
(entry below).

**2.8, B10's boundary weight.**
- **New columns:** `b10.csv` gains `bc_weight_total`, and
  `b10_vs_paper_grid.csv` gains `b10_bc_weight_total` and
  `paper_grid_bc_weight_total`. Each is the total weight of one velocity
  component's boundary rows in units of lambda_bc: 4 with equal weights
  (lambda_bc per edge, as on the paper grid), 1 with Clenshaw-Curtis (over
  the perimeter, Addendum v2.3).
- **Note:** `b10/README.md` says the two variants differ in total boundary
  weight as well as in quadrature.
- **Test:** the totals are checked against `_row_weights`'s assembled
  weights.
- **Incidental fix:** `b10_cgl_cc.main` now closes the table it reads
  before replacing it.

**The wave report** also packs `clean_timing/provenance_*/` and
`b10/README.md`.

Tests: `tests/test_wave3_reply.py`, `tests/test_b10_clenshaw_curtis.py`.

---

## 2026-10-01 -- The advisor's reply on wave 3 and the wave 4 commit: the two required fixes, and the smaller items of a first batch

The advisor approved wave 4 once two fixes are in (his reply on wave 3,
Section 1). Both are here, with some of his Section 2 items.

**1. The clean-timing jobs no longer overwrite each other's table.**
- 13a (CPU) and 13b (GPU) both rewrite
  `B_instrumentation/clean_timing/clean_timing.csv` after every run, and
  both followed only the preflight, so they would have run at once. The CPU
  job's later writes would have dropped the GPU Kovasznay rows, and Section
  4.6 would have fallen back silently to the logged GPU times.
- **The fix:**
  - 13b now waits for 13a (`--dependency=afterok:$pre,afterany:$tcpu`). It
    finishes about an hour later, inside F2's ~15.5 h full stage, at no
    extra cost.
  - `clean_timing.write` also keeps the rows of other runs already in the
    file.
  - Each job's `hardware.json` and `environment.txt` go to
    `clean_timing/provenance_<devices>/`, so neither job overwrites the
    other's.

**2. `package1` can no longer change a wave's files.**
- The advisor found that B9's annotation rewrote its tables in place. Since
  `assemble_package.py` hard-linked the waves' files into `package1`, that
  would have changed wave 3's own tables.
- The problem was wider. `90_finalize` also rewrites `runs_index.csv` and
  `reproduction_check.csv` in place, and both come from the waves. In the
  rehearsal of 1 October (entry "Wave 4 audit"), it changed the scratch
  copy of wave 2's two files; the downloaded originals differ from them.
- **The fix:**
  - `assemble_package.py` now copies (`shutil.copy2`) instead of linking.
    The waves come to under 1 GB (3,252 files in the rehearsal).
  - As the advisor proposed, `annotate` also writes a new file and swaps
    it in (`os.replace`).

**3. A test that would have failed wave 4's preflight.**
- `test_wave_4_submission_copies_component_a_and_chains_the_jobs` runs
  only under Linux, so only in the preflight. Its fixture had no runs of
  F1_15, and `submit_wave4.sh`'s copy loop failed on the unmatched glob.
  The preflight would have failed, and with it every job after it.
- The local rehearsal of the submission had F1_15, which hid it.
- **The fix:**
  - The fixture has F1_15.
  - `submit_wave4.sh` checks that all three finalists have runs, in the
    dry run too, and names any that are missing.
  - The Linux-only tests have now been run under Git Bash on this laptop:
    all pass.

**Section 2 items in this batch:**
- **2.1:** `run_beltrami.py` passes `max_iter=LILQ_PAPER_KMAX`. No paper
  number comes from it.
- **2.3:** `ELMBasis2D`'s sqrt(3 / fan_in) is LeCun uniform, not Kaiming
  with a = 0 (that is sqrt(6 / fan_in)), and Table 3's existing ELM row uses
  `ELMBasis2D_Xavier`. Corrected in `basis.py`'s docstrings, the entry of
  the follow-up items below, and the test comment. The code is unchanged.
- **2.4:** job 35 pins F2_10, F2_05 and F2_14 with `--keep-top`. The
  recomputed validation top three is still recorded (`top_by_validation`),
  with `top_kept_matches_validation`, and a note is printed if they differ.
- **2.6, labels:**
  - The timed class costs 192 SU/h on Grace, not 120 (`sbatch.sh`,
    `provenance.py`, the cluster README).
  - `allocation_label` ends with `, exclusive`, `, whole node` (the timed
    class) or `, shared`. Wave 3's B9 rows read "8 cores + 1 x
    NVIDIA A100-PCIE-40GB, shared".
  - `43_basis_study.slurm` says 10 bases.

Tests:
- `tests/test_cluster_waves.py`: copies, not links; 13b after 13a; F2
  pinned.
- `tests/test_clean_timing.py`: a write keeps other runs' rows.
- `tests/test_followup_items.py`: annotating leaves a hard-linked original
  unchanged; the allocation labels.
- `tests/test_validation_residual.py`: `top_kept_matches_validation`.

---

## 2026-10-01 -- Wave 4 audit: four fixes, and a rehearsal of the package assembly on real data

An audit of the wave 4 commit against the advisor's reply to wave 2, his
follow-up and Addendum v2.3 found four things to fix.

1. **Section 4.6 used logged LiL-Q times.** `kovasznay_comparison.load_lilq`
   read `solve_time_total` from the logged runs. Every quoted LiL-Q time now
   follows the clean-timing rule, so it takes the clean time from
   `B_instrumentation/clean_timing/clean_timing.csv`, keeps the logged one
   beside it (`time_s_logged`), and says which it used (`time_source`). It
   falls back to the logged time only where there is no clean time.
2. **The four-method tables had no LiL-Q rows from the clean-timing runs.**
   The follow-up says those rows "come from them".
   `four_method_tables.lilq_rows` (`--lilq-from-clean-timing`; run by
   `90_finalize`) writes `four_method_lilq.csv` in the four-method schema:
   - time and iterations from `clean_timing.csv`;
   - final loss and convergence from the logged paper pass of the same
     configuration, which follows the same solver path.
3. **The B8 merge in `90_finalize` would have failed.** It took its columns
   from the first table (wave 3's, without `K_max`), and `csv.DictWriter`
   raises on the K_max = 60 rerun rows' extra field. It now takes the union
   of the columns, and gives wave 3's LiL-Q rows their K_max from
   `iterations_cap`: 20 for gravity BL, 50 for viscous.
4. **Table 3's solutions stored a `ComparisonConfig`, a class defined in a
   script run as `__main__`**, so they loaded only with that class made
   visible (wave 1's B6 solutions; the note to the advisor after wave 1).
   Since wave 4 reruns Table 3, the configuration is now saved as a plain
   dict (`dataclasses.asdict`). A test loads one in a separate process.

**Rehearsal of `90_finalize`**, every command, on four waves:
- waves 1-3: the real downloaded results;
- wave 4: this laptop's rehearsal outputs -- clean timing, the gravity BL
  rerun, B10, the B8 K_max = 60 reruns, Table 3, and F1's re-pick
  (selections, F1_18).

Results, all steps completed:
- **Assembly:** 210 files replaced by later waves, all intended. Wave 2:
  the timing reruns and the gate checks. Wave 3: the root `sacct.txt` and
  `report_log.txt`, as every wave writes them (each wave's own copies stay
  in its folder). Wave 4: F1's representative, both selection files, Table
  3, and the gravity BL P = 64 rerun.
- **Check B1:** the same 15 violations.
- **Four-method tables:** 140 rows, 54 controls, and 16 LiL-Q rows (gravity
  BL P = 64: 43 iterations, target).
- **Section 4.6:** LiL-Q on clean times.
- **B8:** 131 rows, K_max 20 / 50 / 60.
- **B9:** 28 rows, the allocation filled.
- **Table 3:** 10 rows, each with `eps_u`.

Tests: `tests/test_kovasznay_comparison.py`, `tests/test_followup_items.py`.

---

## 2026-10-01 -- Wave 4's cluster setup (the advisor's reply to wave 2, Section 4; his follow-up; Addendum v2.3)

**One new commit, the advisor's order:** he reviews the commit; then the
preflight and the gate (29: the search, A1, A2, the F2 Jacobian on the new
code); then the jobs. `submit_wave4.sh`; `LILQ_WAVE` may now be 4
(`sbatch.sh`), the wave report takes `--wave 4`, and `assemble_package.py`
assembles waves 1-4.

**Component A from wave 2's runs (option 3).** Before submitting,
`submit_wave4.sh` copies into `results/wave4/A_calibration`:
- wave 2's `search/`, `screening/` (all 24 + 24 runs) and `tuning_log.md`
  (wave 4's runs are appended to it, as wave 2's were to wave 1's);
- the full runs of F1's finalists F1_04, F1_15, F1_18.

Not F2's: F2_16, F2_03 and F2_17 are not wave 4's finalists. Wave 2's
selection and representative files stay in `results/wave2`; wave 4 writes
its own, which replace them in `package1` (`WAVES.json` overrides).
- `34_A_f1_repick` (shared GPU, 1 h): `select --family F1 --keep-top F1_04
  F1_15 F1_18`, then `full`, which finds the 15 runs, trains nothing, and
  picks F1_18 (rehearsed on wave 2's data: entry "Component A selects by
  validation residual").
- `35_A_f2_full` (timed, 18 h): `select --family F2` -- F2_10, F2_05, F2_14
  -- then `full`: 15 runs of 60 minutes on the allocation of wave 2's job 31
  (one A100, 48 cores; 192 SU/h), the representative by median validation
  residual.
- `32_A_cpu` is submitted once per family (`--array=0` after 34, `--array=1`
  after 35), so F1's CPU reruns do not wait for F2's full stage.
- `33_A_float32` (F1_18) runs after 34.

**The other jobs**, each after the preflight:
- `13a_clean_timing_cpu` (timed-cpu, 1.5 h): first gravity BL P = 64's
  logged paper pass again, now with K_max = 60, for its monitor values; then
  `clean_timing.py` on the CPU for every quoted time, with the logged times
  looked up in wave 4, then wave 2's run folders, then wave 1's (the pinned
  run);
- `13b_clean_timing_gpu` (timed, 1 h): Kovasznay on the GPU;
- `43_basis_study` (cpu): Table 3 again, with the default-init ELM row and
  the new columns;
- `44_b10_cgl_cc` (cpu, 2 h): B10's 16 runs and the comparison with wave 1's
  paper-grid runs;
- `45_b8_kmax60` (cpu, 1 h): wave 3's three capped LiL-Q rows with K_max =
  60, into `b8_jobs/lilq_kmax60/`.

Then `91_wave_report`, after all of them, and `90_finalize`, after the
report. Finalize was moved here from wave 3, where it was submitted and
cancelled; `submit_wave3.sh` no longer submits it. It now fills B9's
`allocation` / `resumed_at` before merging.

**The wave report** adds `A_calibration/representative_thresholds.csv` (the
reply's Section 2). For each family's representative and each of its runs
-- full stage, CPU reruns, float32 -- it gives:
- the first logged time (`t_cum_s`) at which eps_u <= 1e-4, 1e-6, 1e-8 and
  1e-9 (empty if never);
- the final eps_u and the mean-free pressure error;

all read from the runs' `log.csv`. It also packs the validation files, the
clean-timing table, B10's table and logs, and the paper-pass run files.

**Expected cost** (rates of waves 2 and 3), about 3,900 SU, the advisor's
estimate; about 5,300 if every job ran to its walltime:

| Item | Hours | Rate (SU/h) | SU |
|---|---|---|---|
| F2 full stage | ~15.5 | 192 | ~2,980 |
| CPU reruns (F1, F2) | 2 x 5.2 | 48 | ~500 |
| F1 float32 | ~1.15 | 192 | ~220 |
| Preflight and gate | | 80 | ~110 |
| Clean timing | | | ~50 |
| B10, Table 3, B8, report, finalize | | 24 | ~20 |

`submit_lib.sh`'s results folder can be overridden with `LILQ_RESULTS`, so
that the tests can exercise the copy step on a temporary folder. The local
fake-`sbatch` submission: the copies as above (F2_16 not copied), 14 jobs
with the dependencies above.

**Local rehearsal of the clean-timing jobs** (this laptop, with the logged
times of waves 1 and 2 from the downloaded results):
the gravity BL P = 64 logged rerun reached its target at 43 iterations;
`clean_timing.py` wrote 42 rows (CPU and GPU), every one with its logged
time found -- wave 4's gravity rerun, wave 2's run folders, wave 1's pinned
run. The laptop's times are not comparable with Grace's logged ones; the
rehearsal tests the jobs, not the numbers. One observation: on this laptop
Beltrami's timed and warm-up runs agree (301 and 300 s), whereas in wave 2
on Grace the logged run took 303 s and its warm-up 227 s; wave 4's clean run
on Grace settles it.

Tests: `tests/test_cluster_waves.py`.

---

## 2026-10-01 -- The advisor's follow-up items: the default-init ELM row, B9's time records, B8's K_max = 60 reruns, the four-method history cost

**Table 3, default-initialization ELM (item 3).**
`lilq.basis.ELMBasis2D_TorchDefault` uses PyTorch's default `nn.Linear`
initialization: weights and biases uniform on +-1/sqrt(fan_in) = +-0.7071,
since `kaiming_uniform_` with a = sqrt(5) gives 1/sqrt(fan_in). The existing
`ELMBasis2D`, documented as matching `nn.Linear`, uses sqrt(3 / fan_in) =
1.2247 for the weights, i.e. LeCun uniform (corrected from "Kaiming with a =
0", which is sqrt(6 / fan_in); the advisor's reply on wave 3, item 2.3). It
is not the default, is unchanged, and is not used by Table 3, whose
existing ELM row uses `ELMBasis2D_Xavier`. The basis study has a new key,
`elm_default`, with the same width (625) and P as the Xavier `elm` row.

`write_table3_csv` gains `eps_u`, `kappa_raw`, `kappa_retained` and
`num_rank_qr`. `eps_u` is the mean square of u_t + u u_x - nu u_xx on the
uniform 201 x 201 grid of `problems.burgers.make_test_error_fn`, computed for
any basis through its own derivatives (`make_basis_test_error_fn`; the
existing function needs a tensor-product basis). Sin x Sin's `eps_u` is 0 by
construction: it cannot represent the initial condition, its solution is
essentially zero, and zero satisfies the PDE; its final ||R||_h^2 of 4.98
shows the failure.

Local run of the whole table (about 5 minutes):
- the existing rows reproduce wave 1's to round-off where well conditioned
  (relative 1e-15 to 1e-12); the nearly singular rows differ in the last
  digits between machines (Fourier x Fourier: kappa 1.3e16; Xavier ELM:
  rank 43/625);
- `elm_default` reaches ||R||_h^2 = 4.4e-4 against the Xavier ELM's 4.9e-2,
  with SVD rank 176 of 625 and retained kappa 6.6e11.

Wave 4 reruns the whole Table 3 (an untimed CPU job, about 2 SU), so that
every row has the new columns; its files replace wave 1's B6 files in the
package (`WAVES.json` overrides).

**B9's time records (item 4).** `darcy_fv_comparison.csv` gains `allocation`
(e.g. "8 cores + 1 x NVIDIA A100-PCIE-40GB", with ", exclusive" when the job
held the node; from `lilq.provenance.capture_scheduler_info`) and
`resumed_at` (the epochs a NiL run resumed from a checkpoint; `[]` for
none). Wave 3 ran before these columns existed. `darcy_fv_comparison.py
--annotate <darcy_fv>` fills them in an existing table from the job's
`hardware.json` and each run's `network.pt`; the package assembly runs it
before merging. For wave 3: every run on the shared allocation (8 cores, one
A100-PCIE-40GB, not exclusive), and none of the 24 NiL runs resumed.

**B8's capped LiL-Q rows, with K_max = 60 (item 1).** The B8 table gains a
`K_max` column, the LiL-Q K_max of the run; it is the paper pass's 60 since
the entry "Clean timing and K_max = 60", and was 20 for gravity BL in wave
3. `b8_initial_guess.py --rerun-capped-from <tables>` reruns only the LiL-Q
rows of earlier tables that ended on `iteration_cap`, on the CPU, into a
table of their own. Merged with wave 3's, the new rows sit beside the old
ones, told apart by `K_max`. Wave 3 has three such rows, all gravity BL.
Locally, all three reach their targets: initial-condition guess P = 256 and
576 in 23 iterations (losses 0.0749 and 0.0434 against 0.075 and 0.045),
zero guess P = 64 in 43 (0.237 against 0.24).

**Four-method training time and the loss history (the reply to wave 2,
Section 3, item 3).** `training_time` (`training_time_s`) is the
`MetricsTracker` wall time at the last recorded iteration. It includes the
per-iteration record from which `loss_history_every_10` is subsampled after
the run. `experiments/measure_history_cost.py` times it inside real runs
(2,000 iterations, target 0, this laptop):
- the recording: 0.01% of `training_time` on the GPU, 0.05-0.06% on the CPU
  (Bratu P = 100 NiL-N and NiL-Q, Burgers P = 225 NiL-N);
- the loss lookup at the accepted point: 0.44% (GPU) and 0.7% (CPU). It is
  a stored value, no evaluation, and it is the stopping test, which the
  method needs with or without a history. NiL-Q's stopping test is its
  full-loss monitor evaluation (`monitor_evaluations`).

The same rule therefore holds on both sides: a LiL-Q clean time includes
its own stopping test and its per-iteration metrics record, and nothing
else. The same runs show these small networks iterate about 5x faster on
the CPU than on the A100 (9.1-11.1 s against 47-52 s per 2,000
iterations), as the advisor noted in his wave 1 review.

Tests: `tests/test_followup_items.py`.

---

## 2026-10-01 -- Task B10: Kovasznay on CGL grids with Clenshaw-Curtis weights (Addendum v2.3)

**Why.** The advisor's sampling-constant analysis (`Post-JCP/wave2/
S85_Sampling_Constants_Kovasznay.md`, from the Component C collocation files)
found that the paper's equispaced grids and the equal-weight CGL grids do not
satisfy the hypothesis of Corollary 3 for the algorithm as run, while the
same CGL points with Clenshaw-Curtis quadrature weights do, with a constant
of about 1 from N/P = 5-10. B10 puts one experiment inside the proved theory.

**The option** (`KovasznayConfig.weights`, default `'equal'`):
`'clenshaw_curtis'` is allowed only with `sampling = 'cgl'`; any other
sampling, or any other name, raises. The weights come from
`problems.kovasznay._row_weights`:
- **1-D weights:** `lilq.collocation.clenshaw_curtis_weights(n)`, the
  cosine-sum formula of Trefethen's `clencurt`, divided by 2 so that they
  sum to 1, in the node order of `points_1d(..., 'cgl')` (they are
  symmetric). Exact to degree n - 1.
- **Interior:** point (x_i, y_j), each of the three equations:
  sqrt(lambda_block * w_i * w_j), replacing sqrt(lambda_block / n_int).
  The weights are those of the nodes without the grid's 1e-6 inset.
- **Boundary:** edge e with n_e points, each u and v row:
  sqrt(lambda_bc * w_k^(n_e) * |e| / |dOmega|), |dOmega| = 7. Corners keep a
  row on each of their edges.
- **Pressure pin:** unchanged.

The same per-row weights are used by the assembly, the logged residual
vector and the loss (check B2 holds: relative error 1.5e-15 on a small
case), and by `collocation.npz`; `write_collocation_rows` now takes one
weight per row as well as one per block. `run.json` records `weights` in
`collocation_construction`. The interior-row unweighting of the log
(`norm_R_interior`, `norm_Rlin_interior`) needs one scalar interior weight,
so it is empty for Clenshaw-Curtis runs.

**`'equal'` is bitwise unchanged:** `A_final`, `b_final`, the coefficients,
the saved row weights and the iteration counts are identical to commit
`99c78e6`'s. This was checked on equispaced, CGL and scattered grids on the
CPU and two sizes on the GPU, from a separate checkout of `99c78e6`; the
equal branch keeps the scalar expressions it always had.

**Tests** (`tests/test_b10_clenshaw_curtis.py`), the five of Addendum v2.3
Section 1 and more:
- the 1-D weights sum to 1 and integrate x^k exactly for k <= n - 1 (1e-13);
- the interior weights reproduce the normalized integral of exp(x) cos(y):
  1e-13 at the CGL nodes, 1e-5 at the inset grid;
- the boundary weights reproduce the normalized perimeter integral of
  exp(x) + y^2 (1e-12), and each component's weights sum to lambda_bc;
- `'equal'` gives the scalar expressions and the default's solve, bitwise;
- non-CGL sampling or an unknown name raises;
- a Clenshaw-Curtis solve has per-row weights in `collocation.npz`, which
  sum to lambda per equation, and passes check B2;
- the driver end to end.

**The runs** (`experiments/b10_cgl_cc.py`, 16):
- B10-CC: P = 300, 1,200 and 1,875 x N/P = 5 and 10 x the paper and kmax
  passes, Clenshaw-Curtis weights;
- B10-EQ: P = 1,875, the same with equal weights.

Each uses Component C's density rule (`k_for_ratio`, collocation floor 1),
the paper's stopping rule (relative coefficient change 1e-9), and the full
log, `collocation.npz`, model and `hardware.json`. Output: `b10.csv` with
the addendum's columns, plus `rank_deficient` and `K_max`, and
`b10_vs_paper_grid.csv`: the paper passes' errors beside the paper-grid
runs' at the same P.

**K_max of the paper pass.** The addendum's text says 20; the advisor's
follow-up of the same day says 60 for every LiL-Q paper pass. B10 uses 60.
Every paper pass of the local rehearsal converged in 6-9 iterations, so
this makes no difference here.

**Local rehearsal** (this laptop; the record runs are wave 4's).
All 16 runs completed. Every paper pass reached the
tolerance, in 9 iterations at P = 300 and 6 at P = 1,200 and 1,875; every
system had full rank (300, 1,200, 1,875). Raw and retained kappa are equal:
about 6.3e2, 1.0e4 and 3.1e4 for Clenshaw-Curtis, and 4.1-4.4e4 for equal
weights at P = 1,875. Velocity errors eps_u, against the wave 2 paper-grid
paper pass at the same P (equispaced, k = 4):
- P = 300: 2.34e-2 against 2.90e-2 (0.81x);
- P = 1,200: 6.58e-9 against 7.36e-9 (0.89x);
- P = 1,875: 5.8e-14 against 7.0e-13 (0.083x), and equal-weight CGL also
  5.9e-14 (0.084x).

N/P = 5 and 10 give the same errors to three digits. At P = 1,875 the gain
comes from the CGL points rather than the weights: equal and
Clenshaw-Curtis weights agree. The kmax passes stay at the paper passes'
errors.

---

## 2026-10-01 -- Clean timing and K_max = 60 for every LiL-Q paper pass (the advisor's reply to wave 2, Section 3, and his follow-up)

**Why.** In wave 2 Beltrami's untimed warm-up run (no logger) took 227 s and
the timed, logged run 303 s. The advisor set one rule for every LiL-Q time
the paper quotes: it comes from a run with the per-iteration diagnostics
off, after the warm-up. The logged runs stay the source of every monitor
value.

**The switch** (`diagnostics=True` by default) on `lilq.solvers.solve_lil_q`
and the scalar `run_lil_q` wrappers (Bratu, Burgers, BL), `solve_kovasznay`,
`solve_beltrami`, `solve_elasticity` and `solve_lilq_darcy`. With
`diagnostics=False`:
- no `iteration_logger`, `run.json` or conditioning (`analyze_conditioning`,
  `diagnostics_callback`); passing any of them raises;
- the passive work that otherwise ran on the clock is skipped:
  - Kovasznay and Beltrami: the nonlinear residuals recorded in `history`
    every iteration (`NaN` instead, and `pde_mse` / `cont_mse` are `NaN`;
    the stopping rule is the coefficient change);
  - Kovasznay on the GPU: the CPU `gelsy` rank cross-check of a flagged
    iteration (the flag itself is still recorded);
  - Darcy: `total_time` stops when the LiL solve ends. The FVM reference
    solve and the error metrics after it (0.18-0.27 s of 21-28 s in wave 2)
    are not part of the method.
- The scalar loop has no other passive work (its nonlinear loss is the
  stopping test), and neither has elasticity's assembly + solve phase.

The solver path is unchanged: with the switch on and off the final
coefficients are bit for bit identical for Bratu, gravity BL, Kovasznay (CPU
and GPU), Beltrami (one and three pins), elasticity and Darcy
(`tests/test_clean_timing.py`).

**The runs** (`experiments/clean_timing.py`). For every paper-pass
configuration -- Bratu, Burgers, both BL, elasticity, Kovasznay (CPU and
GPU), Beltrami, the pinned Beltrami run of Section 3.7 (the paper's Beltrami
run, the advisor's follow-up item 2; `run_beltrami_pinned.pinned_config`)
and Darcy -- it runs one untimed warm-up, the timed run with
`diagnostics=False`, and, under 1 s, five more reported as their median
(Addendum v2.2 2.10). Recorded quantities are those the logged runs report:
- `training_time` for the scalar benchmarks;
- `solve_time_total` for Kovasznay, Beltrami and pinned Beltrami;
- `time_lil_s` and `solve_time_qr` for elasticity;
- `total_time` for Darcy.

`clean_timing.csv` has one row per run and quantity, with the warm-up time,
the iterations, K_max, and the logged time beside it, found under
`--logged-roots`: wave 2's run folders, and wave 1's `beltrami_pinned`
report for the pinned run.

**K_max = 60** (`lilq.solvers.LILQ_PAPER_KMAX`) for every LiL-Q paper pass:
- the scalar `paper_setup`s' `max_quasi_iters_lil`;
- Kovasznay's `MAX_ITER` (also used by B3 and Component C);
- Component B's Beltrami paper pass;
- the pinned run's configuration (its K_max pass stays 8).

LiL-Q's K_max had shared a constant with NiL-Q's outer-iteration budget
(`MAX_QUASI_ITERS` in each `run_*.py`), so it is now a separate constant.
NiL-Q's budgets are unchanged: 25 (Bratu), 20 (Burgers), 50 (viscous BL)
and 20 (gravity BL).

Only gravity BL at P = 64 behaves differently. It now reaches its target
(0.24) at 43 iterations, as the advisor predicted from wave 1's kmax log;
wave 2's paper pass ended at K_max = 20 without reaching it. Its logged paper
pass is rerun in wave 4 for its monitor values. Every other paper pass
converges within its old cap, e.g. gravity BL P = 256 at 12 as before. The
four-method tables' LiL-Q rows come from the clean-timing runs; B8's LiL-Q
rows that ended on `iteration_cap` in wave 3 are rerun with K_max = 60 in
wave 4. Component C is not rerun.

Tests: `tests/test_clean_timing.py`; `tests/test_component_b.py` and
`tests/test_four_method_tables.py` updated.

---

## 2026-10-01 -- Component A selects by validation residual (the advisor's reply to wave 2)

**Why.** Wave 2 selected each family's top three and representative by
final training loss (Package 1 v2.0 Section 4.3; F1 by its unweighted
loss). In both families that chose the configuration that generalizes
worst: F1_04 (median eps_u 7.4e-2, pressure about 100%) and F2_16 (7.6e-3),
against 3e-5 to 1e-4 (F1) and 4e-9 to 2e-6 (F2) for the other finalists.
These configurations fit their collocation points without learning the
solution: the six lowest-training-loss F2 configurations have fewer residual
rows than parameters, and F1_04 has sigma_FF = 5 with 2,000 points. The
residual on fresh points ranks the 24 screening runs of each family in
nearly the order of the test error (Spearman 0.93 / 0.94, against 0.70 /
0.11 for the training loss; on the 15 full runs 0.90 / 0.95 against -0.62 /
-0.86). The advisor checked this with his own code and a different design (a
96 x 96 Gauss-Legendre quadrature: 0.92 / 0.95, the same leaders) and
adopted our design as the protocol.

**The protocol** (`baselines/validation.py`): F1's unweighted loss on points
no run trained on -- the x-momentum, y-momentum and continuity mean squares
on 20,000 uniform interior points (`torch.Generator().manual_seed(20261001)`,
float64, drawn on the CPU so the points do not depend on the device; chunks
of 2,500), plus, for soft-BC F1 runs, the u and v boundary mean square on
400 points per face. Hard-BC F1 runs and every F2 run satisfy the boundary
data exactly. No exact solution is used beyond the boundary data. On wave
2's saved models the module reproduces the analysis of `wave2-results`
(`results/wave2/analysis/`) bit for bit on the GPU and to round-off (1e-10
relative at 4e-16) on the CPU.

**Selection** (`component_a.select`): the top three distinct configurations
by the screening runs' validation residual. The selection file records the
validation ranking (with each run's training loss), the training-loss
ranking (`ranking_by_training_loss`, `top_by_training_loss`: the paper shows
all 24 screening runs under both criteria), `top_by_validation`, and the
protocol (seed and counts). `--keep-top` keeps given configurations as the
top three instead, recording why (`top_kept`) and the validation top three
beside them.

**Representative** (`component_a.full`): the lowest median validation
residual over the five seeds. The file records the medians of both
criteria and every run's validation residual. Full runs that exist are not
retrained.

**Where the residual is kept.** A run made by this code records
`val_residual`, `val_terms` and `val_protocol` in its `run.json`, computed
off the budget clock after training. An older run's `run.json` is left as it
was written, and the residual goes into `validation.json` beside it. Either
is read back, never recomputed. A run without a saved model (a failure)
ranks last, with `inf`.

**Wave 4, per the advisor's decision (option 3).** F2: `select --family F2`
on wave 2's screening runs gives F2_10, F2_05 and F2_14, the advisor's three;
a new full stage, then the CPU reruns of its representative. F1: no new full
stage; `select --family F1 --keep-top F1_04 F1_15 F1_18`, then `full`, which
finds wave 2's 15 runs and re-picks the representative: F1_18 (median
validation residual 3.4e-6, against 2.3e-5 for F1_15 and 3.8 for F1_04).
Then its CPU reruns and its float32 run. Both were rehearsed on a copy of
wave 2's Component A folder: the selections and F1_18 as above, nothing
retrained. F1_04's and F2_16's runs and their wave-2 selection files stay
in `results/wave2`.

Tests: `tests/test_validation_residual.py`; `tests/test_component_a.py` and
`tests/test_baseline_search.py` updated.

---

## 2026-09-30 -- Beltrami and Darcy timing reruns (the advisor's reply to our note before wave 2)

The advisor accepted the note's three departures but one: every reported
LiL-Q time should follow one rule, so the Beltrami and Darcy paper passes
are rerun with the warm-up run too. `12_timing_reruns_beltrami_darcy.slurm`
(timed-cpu, 1 h, about 20 minutes of work) runs `component_b.py --passes
paper --devices cpu --benchmarks beltrami darcy` into wave 2's run folders,
which have wave 1's names, so the assembled package takes them and
`WAVES.json` lists them under `overrides`.

It could not be added to job 11a, already queued: Slurm copies a job's
script at submission, and replacing 11a with an edited one would change a
file of the bundle, which the lock refuses. So it was submitted by hand as
its own wave-2 job, from a copy of this file on the cluster. The lock hashes
only the bundle's listed files, so the extra file leaves wave 2's tree hash
unchanged (checked on the extracted 905e58c bundle); the code it runs is
905e58c's. This commit adds the file and its line in `submit_wave2.sh` for
the record; the Grace bundle stays 905e58c.

The Section 3.7 runs (`run_beltrami_pinned.py`, wave 1) have no warm-up and
are not rerun; their `solver_time_s` comes from a single cold run each.

---

## 2026-09-29 -- Wave 2's cluster setup: one GPU per timed job, job 21 at 4 h, timing reruns, controls (reply to wave 1, items 2.2-2.5 and Section 3)

**Timed GPU jobs at 120 SU/h (item 2.3).** Wave 1's `sacct` showed that
`--exclusive` on an A100 node allocates both A100s (`gres/gpu=2`) and Grace
charges both: 192 SU/h. The `timed` class (`sbatch.sh`) now requests
`--gres=gpu:a100:1 --ntasks=1 --cpus-per-task=48 --mem=360G` without
`--exclusive`: the job holds every core and all the memory, so no other
job can be placed on the node, and the timing conditions are those of wave
1, at 120 SU/h. The advisor estimates the saving at 3,300-3,600 SU in wave 2.
`hardware.json`'s `scheduler` now records `holds_whole_node` (every core and
all the memory of the node), `gpus_allocated` (from `AllocTRES`) and the
job's `ReqTRES`/`AllocTRES`; `exclusive` stays, and is false from wave 2 on
for the A100 jobs. The `timed-cpu` class keeps `--exclusive` (CPU nodes
carry no GPU surcharge). To verify on wave 2's first timed job: `sacct`
shows `gres/gpu=1`, and its charge in `myproject` is 120 SU/h
(`submit_wave2.sh` prints the command).

**Job 21 at 4 h (item 2.4):** gravity BL's worst case reached 3 h.

**Timing reruns (item 2.2).** `11a_timing_reruns_cpu` (timed-cpu): the paper
passes of Bratu, Burgers, both BL, elasticity and Kovasznay on the CPU;
`11b_timing_reruns_gpu` (timed): the Kovasznay GPU paper passes and check
B3. Both 1 h, both in wave 2, both with the warm-up runs of the entry below.
They write to `results/wave2` under the same run folder names as wave 1, so
`assemble_package.py` takes wave 2's and lists every replaced file in
`WAVES.json` (`overrides`); `90_finalize` rebuilds the run index, check B1
and the merged four-method tables from the assembled package.

**Stall controls in jobs 20 and 21 (item 2.5).** Every task runs its own
controls after its runs (`--controls`). In wave 2 the Bratu task (index 0)
runs only the controls of wave 1's stalled Bratu runs
(`--controls-only --controls-from results/wave1/.../bratu_<device>`), into
`four_method_jobs/bratu_<device>_controls` -- its own folder, so that wave
1's `bratu_<device>` tables are not overridden when the package is
assembled. Wave 2 submits 20 and 21 with `--array=0-3`.

**Wave 2 report (Section 3).** `wave_report.py` writes
`A_calibration/run_endings.csv`: every Component A run's end reason,
budget and wall time, with `ended_before_budget`, and lists those in the
report log. It now also packs each screening run's `log.csv`; it already
packed the search lists, the screening `run.json`s and selections, and the B root's CSVs, which now include
`four_method_controls.csv`.

**Wave 3** compares the code with the last wave's commit (wave 2's when it
exists) to decide whether the preflight runs again, not with wave 1's.

Tests: `tests/test_cluster_waves.py`, `tests/test_provenance.py`. The wave 2
dry run was checked locally with a stand-in `sbatch`: 11 jobs, the five
timed GPU jobs each `--gres=gpu:a100:1 --cpus-per-task=48 --mem=360G`
without `--exclusive`.

---

## 2026-09-29 -- Four-method stall control (reply to wave 1, item 2.5)

**Why.** At Bratu P = 100 and 225, wave 1's nonconvex runs ended on
`optimizer_stall` while their loss was still falling (0.8-19% per 500
iterations). The stop comes from PyTorch's absolute L-BFGS tolerances
(`tolerance_grad` 1e-8, `tolerance_change` 1e-9) -- the settings behind the
published tables, and the loss-scale dependence F1 no longer has. The
control shows whether the stall, rather than the method, limits the
reported loss.

**The rule** (`lilq.solvers.STALL_RULES`; `stall_rule` on `solve_nil_n`,
`solve_nil_q`, `solve_lil_n` and on the Bratu, Burgers and BL `OptConfig`s).
`'pytorch'` is the default and unchanged: the golden-trajectory tests still
pass bit for bit. `'f1'`: tolerances 0, and F1's rule -- a step that does not
lower the loss is followed by a step with a fresh L-BFGS optimizer, and the
run ends on `optimizer_stall` only if that step does not lower it either. A
step that lowers the loss clears the restart, as in F1. "Lowers the loss"
is the loss L-BFGS minimizes: the full loss for NiL-N and LiL-N; for NiL-Q
the linearized loss of the current outer iteration. NiL-Q's inner loop ends
on a second no-progress step in a row, the fresh optimizer serves the rest
of the run, and each linearization starts with no restart used; the outer
stall is unchanged (an outer iteration that leaves the parameters bitwise
unchanged). F1's rule is per 500-iteration call; here it is per step, the
unit at which these loops stop. The summary records `stall_rule` and
`lbfgs_restarts`.

**The runs** (`experiments/four_method_tables.py`). `--controls` reruns,
after a job's own runs, each of them that ended on `optimizer_stall`;
`--controls-from <job dirs>` does the same for an earlier table's rows
(wave 1's Bratu); `--controls-only` skips the job's own runs. Each control
has the original's benchmark, size, method, seed and device, and its
budgets and caps; the configuration comes from the problem's paper settings
(every size, also on `--quick` budgets) and is checked against the row's
collocation seed; a mismatch is logged as a failed control row and the other
controls still run (found in a local rehearsal of wave 2 on wave 1's results). Rows are logged with
`variant = f1_stall_rule`, the history and model saved like any other
(model folder suffixed `_f1_stall_rule`). New columns: `variant`,
`lbfgs_restarts`, `same_start` (iteration 0's loss equals the original's,
bit for bit -- the check that the control started from the same point),
`departs_at_iteration` (the first logged iteration whose loss differs from
the original's), and `original_total_iterations` / `original_final_loss`.
`variant` is part of a row's key, so a control never collides with its
original; wave 1's CSV, which has no such column, reads as the tables' own
runs. `--merge-from` writes the tables' own runs to `four_method_tables.csv`
and the controls to `four_method_controls.csv` beside it.

**Device.** The reply says to run the controls on the GPU. Five of wave 1's
fourteen stalled Bratu runs were on the CPU (the P = 225 CPU pass), and a
CPU run's trajectory cannot be reproduced on the GPU (floating-point order;
the review notes GPU and CPU runs from one seed diverge). So a control runs
on its original's device: GPU rows in job 20, CPU rows in job 21, each after
that job's own runs. For wave 1's Bratu rows: job 20's Bratu task (9 GPU
controls) and job 21's Bratu task (5 CPU controls), both with
`--controls-only --controls-from` wave 1's job folder. Stated in the note to
the advisor.

`same_start` also tests reproducibility across commits and nodes: a
control of a wave-1 row runs in wave 2, on another commit (the solvers'
default path is unchanged) and possibly another node of the same type. If
it is false, the control is still a valid control of that seed, but not of
that trajectory; it will be reported, with `start_loss_rel_diff`. A local
check shows why this matters: the five stalled wave-1 Bratu CPU runs,
controlled on this laptop's CPU with `--controls-only --controls-from`,
start within 2e-16 to 9e-16 of Grace's iteration-0 loss but not bit for
bit, and depart at once; their final losses then differ from the originals
in both directions (e.g. seed 1 NiL-N 5.2e-4 against 1.7e-4). On Grace the
controls run on the originals' node type and core count (jobs 20 and 21),
where the start is expected to be identical.

Tests: `tests/test_stall_control.py`.

---

## 2026-09-29 -- Timing: a warm-up run before every timed run; elasticity's repeats; Beltrami flag removed (reply to wave 1, item 2.2 and Section 1)

**Warm-up.** The advisor found in wave 1's logs that the first solve at
each new matrix size costs 0.6-1.35 s on the A100 against 5-40 ms
afterwards (1.35 s against a 0.60 s median on the CPU at Kovasznay
P = 1,875). The existing warm-up -- one small Kovasznay solve per device --
does not cover a new size, so Kovasznay's GPU paper-pass totals at P = 300
(1.42 s) and 1,200 included one-time setup. Worse, a run pushed over 1 s by
that cost skipped the median-of-five that runs under 1 s get.

`component_b._warm_up_run`: every paper-pass run -- the timed one -- of
every benchmark (Bratu, Burgers, both BL, elasticity, Kovasznay on CPU and
GPU, Beltrami, Darcy) is preceded by one complete untimed run of the same
configuration on the same device. A complete run rather than a single
solve of the same shape: it covers the solve and any other first-call cost
(allocation, basis evaluation), and the runs being rerun take seconds. Its
own time, the cold time, is recorded as `warmup_run_time_s` beside the
timed one. The kmax pass is not timed and gets none. On this laptop's GPU,
Kovasznay P = 300 goes from 0.109 s cold to 0.059 s warm. Check B3's
same-system timings already had a same-shape warm-up (`_median_time` and
`_median_gpu_parts` solve the same system once untimed before timing), so
B3 is unchanged; it is rerun in wave 2 as asked.

**Elasticity's median of five.** Wave 1 repeated `solve_time_total`, which
counts collocation and the error evaluation as well as the solve, so its
medians (0.069-0.862 s) were 2-35x the logged solve (0.0019 s at the
smallest size). `solve_elasticity` now also returns `time_lil_s` = assembly
+ solve, the phase of `iterations.csv`'s `t_cum_s`, and `time_assemble_s`.
Each repeat records `time_lil_s` and `solve_time_qr` (Table 7's quantity);
the summary reports both as medians, their single-run values beside them,
and `solve_time_total` as the logged run's own. The 1 s threshold applies to
`time_lil_s`.

**Beltrami.** `run_beltrami_pinned.py`'s report drops
`improved_over_paper_baseline`: it compared 0.75145% with the paper's
rounded 0.752%, which is no difference. The report keeps both numbers.
Wave 1's `report.json` still carries the flag; it must not be quoted.

**Reruns in wave 2** (on the new commit, in timed jobs): the Kovasznay CPU
and GPU paper passes at all five sizes, check B3, the elasticity paper
passes and the scalar benchmarks' paper passes. Their rows replace wave 1's
in `package1` (`WAVES.json` lists the replaced files); the kmax passes and
every non-timing result of wave 1 stand. Beltrami and Darcy are not rerun,
as the reply lists: their wave-1 times (285 s, 24-30 s) come from a single
cold run each.

Tests: `tests/test_wave1_reply_timing.py`.

---

## 2026-09-29 -- F2 search: 24 distinct configurations (reply to wave 1, item 2.1)

Wave 1's `search/F2_configs.json` had only 17 distinct configurations of
24: `draw_f2` redrew draws over the n_theta cap but kept repeats, so the
top-3 selection could pick one configuration twice and spend 5-10 GPU-hours
of the full stage on it.

**Rule.** `baselines.search.draw_f2` now redraws a repeat from the same
generator (`default_rng(12345)`), exactly as it redraws a draw over the cap,
until it has 24 distinct configurations. Two configurations are the same
when width, depth, m, sigma_FF and N_int all agree. 42 of the space's 72
configurations satisfy n_theta <= 20,000 (width 100 never does, nor does
width 64 at depth 4 with m = 64), so 24 distinct exist; `draw_f2` raises if
a cap ever leaves fewer than 24. The list's first 11 configurations are
wave 1's; from the first repeat on, the draws differ. The saved list
records `redrawn_violators` (32) and `redrawn_duplicates` (11). It is
written once, by wave 2's gate job (29), into `results/wave2`.

`component_a.select` takes the top three *distinct* configurations by
screening loss, whatever the list, and F1's list (already 24 distinct) is
unchanged.

Tests: `tests/test_baseline_search.py` (24 distinct entries in both saved
lists; wave 1's first 11 draws kept; the too-few-under-the-cap error;
selection of three distinct configurations from a list with repeats).

---

## 2026-09-29 -- Wave-1 preflight failed on a test import; tests share helpers through conftest

The wave-1 preflight at `111dddc` (job 19898134, 14 min) failed on two of
its tests: 412 passed, 4 skipped (git), and the two lock-race tests of
`test_advisor_reply_fixes.py` failed with `ModuleNotFoundError: No module
named 'tests.test_data_retention'`. They borrowed a helper by importing
another test file (`from tests.test_data_retention import _fake_bundle`),
which works only when Python resolves the name `tests` to this folder; on
the laptop it did (also from an extracted bundle), in Grace's Python it did
not -- most likely a site-packages package named `tests` came first. The
science code was not involved. As designed, the eight dependent jobs were
cancelled (`--kill-on-invalid-dep`); the preflight had written
`results/wave1/COMMIT` (111dddc), removed for the rerun as the README says.

Fix: the helper is a pytest fixture (`fake_bundle`) in `tests/conftest.py`,
and a new test fails if any test file imports another. Checked by running
the whole suite from an extracted bundle with a decoy package named `tests`
first on `PYTHONPATH`, reproducing the likely Grace condition.

---

## 2026-09-29 -- Grace refuses `--mem=0`; a dry-run mode for the wave scripts

The first wave-1 submission (at `e69dbf7`) stopped at job 10a: Grace's
job_submit plugin refuses `--mem=0` ("all of the node's memory"), which the
whole-node classes (`timed`, `timed-cpu`) used; FASTER had accepted it.
Jobs 00 and 29 had been queued; they were cancelled while pending, before
the preflight wrote `results/wave1/COMMIT`, so nothing ran or was charged
and the wave folder stayed empty. The profiles now give the memory:
`NODE_MEM` and `CPU_NODE_MEM` = 360G on Grace (RealMemory 368,640 MB, the
same on the A100 and CPU nodes), 0 on FASTER, where it works.

`DRY_RUN=1 bash scripts/cluster/submit_wave<N>.sh` now puts every job of a
wave through `sbatch --test-only` -- the scheduler's own checks and an
estimated start -- with the profile's resources, leaving out the
dependencies and submitting, charging and writing nothing; it reports each
job and fails if any would be refused. The README asks for it before every
wave. Tested with a stand-in `sbatch` that refuses `--mem=0` as Grace does.

Also in the README: the venv install keeps the module's numpy
(`-c constraints.txt` with `numpy==1.26.4`). On Grace the newest matplotlib
pulled numpy 2.5.3 into the venv, which the module's scipy 1.13.1 cannot
import; caught before submission by the version check, and removed.

---

## 2026-09-29 -- The advisor's reply, Section 2: fixes for waves 2 and 3, made before wave 1

Made now rather than before waves 2 and 3, so all three waves run on one
commit (a later commit would make wave 2 rerun the preflight and the A1
gate). Each has a test (`tests/test_advisor_reply_fixes.py`).

1. **Component A selection:** `_final_loss` maps every non-finite value
   (NaN from a diverged run, as well as None) to inf, so a failed
   configuration can never enter the top 3 or become the representative.
2. **The lock's creation** (`source_lock.py`): each job writes its own
   temporary file (named by host, process and a random token) and creates
   `COMMIT` by a hard link, which fails if it already exists, so the first
   job's lock is never overwritten; reading retries a half-written file.
   (Jobs 30, 20 and 21 of wave 2 start together.) Tested with 8 threads and
   with 8 separate processes; the suite run from the extracted bundle caught
   a first version whose temporary name (host and process only) collided
   between threads of one process.
3. **PyTorch version in every job:** `env.sh` runs `assert_torch_version`
   after the lock check (waves 2 and 3 skip the preflight on unchanged code).
4. **B8's LiL-Q rows:** a non-finite final loss is `failure`, as 2.3's rule.
5. **Diverged L-BFGS runs stop** (`lilq.solvers`): NaN parameters never
   compare equal, so the stall test cannot fire and a diverged run spent its
   whole budget. NiL-N and LiL-N stop at the first non-finite loss, NiL-Q
   at the first non-finite full loss; the run is classified `failure`.
6. **B9:** `commit` and `n_params` columns in the B9 CSV; the finalize
   counts 24 NiL rows (4 fields x 3 seeds x 2 precisions); the `DarcyPINN`
   docstring says plain MLPs (it said ResNet-style); walltime 4 h per task
   (small networks are overhead-bound, and a Grace core may be slower than
   the laptop's).

From the reply's notes: Component C checks that `N_rows` equals the rows
the solver saved in `collocation.npz`; `run_gpu_equivalence`'s docstring
describes the tightened B3 waiver; `run_beltrami_pinned.py` skips a
completed run, so a resubmitted job 10a does not redo it. **The golden-
trajectory check the advisor suggested**, against the reviewed commit
itself: Bratu P = 100, 300 iterations on the CPU, the loss trajectory from
`2263cef`'s own code and from this commit is bitwise identical for NiL-N and
for LiL-N (301 entries each), with real evaluations 636 -> 337 and
612 -> 313.

---

## 2026-09-29 -- The advisor's reply to our Addendum v2.2 response: before wave 1

The advisor accepted the implementation (a static reading of `2263cef`..
`2c2d0fd`) and gave wave 1 the go-ahead after three changes:

1. **A test needed a git checkout** (`test_bundle_records_a_tree_hash_its_
   files_reproduce` builds a bundle, which runs `git`); on the cluster copy
   it would have failed the preflight, and every wave-1 job waits on the
   preflight. Now `@requires_git_checkout`. The whole suite was then run
   from an extracted bundle on the laptop (no `.git`): 439 passed, 4
   skipped (the four tests that need git), none failed.
2. **A failed preflight no longer leaves the wave stuck.** `sbatch.sh`
   submits with `--kill-on-invalid-dep=yes`, so jobs whose dependency can
   never be met are cancelled rather than left queued with their SUs booked.
   The preflight writes only `results/wave<N>/COMMIT` into the wave folder
   (checked); recovery at a new commit is to remove it and resubmit (README).
   The preflight's smoke outputs now go to a folder per job
   (`$SCRATCH/lilq-run/preflight/<job id>/`): the drivers skip completed runs,
   so a rerun after a fix would otherwise have reused the old smoke results.
3. **Job 20 for Bratu only is submitted with `--time=03:00:00`** (the 8 h
   is sized for the largest benchmark): wave 1 requests ~1,410 SU (~1,770
   if both GPUs of an exclusive node are charged). Walltimes are changed
   only on the command line; editing a `.slurm` file on the cluster would
   change the tree hash and trip the lock.

Records he asked for: our response's items 3.4 (an evaluation finishing
past the budget does not count as reached) and 3.5 (L-BFGS calls in pieces
of 10) are in the 2026-09-29 F1 entry and now also in the Component A
tuning log's header, with his note that capped pieces stop at 624 real
evaluations rather than 625 and iteration counts can differ by one in
exact-zero edge cases; item 3.11 (Kovasznay's tracker and test grid off the
clock, as diagnostics) is in the 2026-09-29 timing entry. The B9 decision:
the manuscript's network, and the single formulation (next entry).

---

## 2026-09-29 -- B9: the manuscript's NiL network; float32 runs beside float64

**The network.** The repository's NiL Darcy network (GitHub `main` and
this branch alike) was three 8 x 200 networks, about 847,000 parameters. The
manuscript's is not (Table 13: three networks of 2 hidden layers x 32,
3,555 parameters, close to LiL's 3,169). The pre-GitHub notebook
(`SPE10/PINN_3_network_for_SEP10.ipynb`) has both, and its printed outputs
identify the paper's runs exactly: the S2, S3 and SPE10 NiL rows of Table
14 (e.g. S2 3.43e-2 / 4.21e-2 / 8.05e-10; SPE10 24.98 / 94.69 / 2.97e-7) come
from the 2 x 32 cells; the 8 x 200 run is a later cell, not in the paper.
The consolidation into the repository kept the large one: GitHub `main` and
this branch carried an 8 x 200 network that produced no number in the paper. `DarcyPINN` and
`run_nil_n_darcy` now default to 2 x 32 (3,555 parameters, tested). The
runtime confirms it: 150,000 epochs take ~19 min in float32 on the laptop
GPU (the manuscript: ~16 min for S1), ~113 min in float64; the 8 x 200
network took 3.2 h in float32 and ~23 h in float64.

Kept from the repository's formulation (the one LiL and the FV reference
solve, and B9 compares against): the SiLU networks with the sqrt(K)
normalization, the loss weights 50 / 20, Adam with cosine annealing and
gradient clipping (all as in the paper's SPE10 NiL run), and pressure
fixed on both the top and bottom faces (as LiL and the FV reference have
it; the paper's SPE10 NiL run differed here, below). The notebook's
synthetic-field runs (S1-S3) used tanh networks with a sigmoid output for
h*, unit loss weights and StepLR, and its SPE10 run a flux condition on
the top face against a matching FV solve. **The advisor's decision
(reply to our Addendum v2.2 response, Section 3): keep the single
formulation for all four fields**, because delta_FV is meaningful only if
NiL, LiL and the FV reference solve the same boundary-value problem; the
report states the NiL recipe, notes that the manuscript's NiL residual MSEs
came from per-field notebook formulations and are replaced, and reports
delta_FV as the primary measure, with the 20% rule applied to delta_FV.

**float32 beside float64.** Addendum v2.2's rule reports the float32 value
where delta_FV differs by more than 20% from float64's, but no float32
delta_FV existed (the manuscript reports only residual MSEs). Each B9 task
now trains its seed in float64 (the result) and float32 (the manuscript's
precision), from the same code: rows carry `dtype`, models are saved in
`NiL_<field>_s<seed>_<dtype>/`. Job 40's walltime: 6 h -> 3 h per task
(~2.2 h measured on the laptop GPU for both precisions); wave 3's request
drops from ~7,700 to ~4,800 SU.

---

## 2026-09-29 -- Audit against Addendum v2.2 before wave 1: four fixes

A pass through every item of Addendum v2.2 against the code found:

1. **The wave report's run index failed** (`wave_report.py` passed a string
   where `write_index` takes a path), so `runs_index.csv`, one of the files
   the advisor asked for after wave 1, would have been missing. Fixed; a
   test now runs the report on real (smoke) Component B output.
2. **F1's pressure only mean-free** (Section 1, item 9): Component A's
   summary of the best test errors over the full runs no longer lists F1's
   pinned-gauge `eps_p`; F2, which has the corner pin, keeps both.
3. **`threadpoolctl`**, which records the thread pools in `hardware.json`
   (Section 2.12), is now in `requirements.txt` and the cluster install
   line, and the preflight fails if it cannot be imported (without it the
   record would say "unavailable").
4. **`time.perf_counter()`** also in the example scripts and the demo
   notebook (Section 2.10.4: "everywhere a duration is measured").

The same pass found two B9 issues, resolved in the next entry above: the
NiL Darcy network was not the manuscript's, and no float32 delta_FV existed
for the advisor's 20% rule.

---

## 2026-09-29 -- Kovasznay GPU solve: Q applied implicitly, as `gels` does (Addendum v2.2 Section 2.10, optional item)

`_lstsq_gpu_qr` used `torch.linalg.qr(mode='reduced')`, which forms the
N x P matrix Q and then multiplies Q^T b; LAPACK's `gels`, the CPU side of
check B3's same-algorithm timing, applies Q^T from the Householder
reflectors and never forms Q. The GPU solve now does the same: `torch.geqrf`,
`torch.ormqr` for Q^T b, then the triangular solve with R (whose diagonal
still feeds the degeneracy flag). Measured on the laptop's RTX 5080 on
Kovasznay's final systems (median of 5):

| P | rows | explicit Q | implicit Q | solution vs old / vs CPU gelsy | peak memory |
|---|---|---|---|---|---|
| 75 | 381 | 0.93 ms | 1.76 ms | 9e-16 / 9e-16 | 16 -> 16 MB |
| 300 | 929 | 5.28 ms | 4.28 ms | 2e-15 / 3e-15 | 23 -> 23 MB |
| 675 | 2,089 | 20.7 ms | 14.6 ms | 8e-15 / 7e-15 | 60 -> 66 MB |
| 1,200 | 3,524 | 81.6 ms | 56.6 ms | 1e-14 / 1e-14 | 148 -> 175 MB |
| 1,875 | 5,564 | 226.7 ms | 143.5 ms | 1e-13 / 7e-14 | 336 -> 412 MB |

The same |R_pp| at every size. Faster by 1.2-1.6x from P = 300; slower at
P = 75, where both take about a millisecond. The peak memory is higher, not
lower as first assumed (the reflector application's workspace), and
immaterial on an A100. The laptop GPU runs float64 at a small fraction of
its float32 rate, so the A100's ratio is measured in wave 1 (job 10b).
Test: on a real Kovasznay system the new solve matches the replaced one to
1e-12 with the same |R_pp|, and on an inconsistent system CPU `gels` to 1e-12.

---

## 2026-09-29 -- Grace execution: CPU nodes for CPU-only timed work; three waves, one results folder each (Addendum v2.2 Section 4)

- **`timed-cpu` resource class** (`sbatch.sh`): a whole CPU node,
  `--exclusive --cpus-per-task=$CPU_NODE_CORES --mem=0` on `$CPU_PARTITION`
  (Grace: 48 cores, `medium`). Grace's CPU nodes have the A100 nodes' CPU,
  2 x Xeon Gold 6248R, and the same memory (`scontrol`, `lscpu`,
  2026-09-29), without the GPU surcharge. Jobs 21 (B4 CPU pass) and 32
  (Component A on the CPU) move to it, and job 10 is split: `10a`
  (timed-cpu, 4 h: the Section 3.3 CPU runs with their K_max passes, then
  Beltrami pinned and its K_max pass) and `10b` (timed A100 node, 2 h: the
  Kovasznay GPU runs and check B3, whose CPU gelsy/gels timings therefore
  come from the A100 node's CPU -- the same model). **Deviation:** Addendum
  v2.1 Section 2's "one node" becomes "one node type per device".
- **B4's CPU pass on a CPU-only node** used to be skipped: `run_problem`
  added the CPU runs only when the primary device was not the CPU, which on a
  CPU node it is. Fixed (a 'cpu'-only call always runs), with a test.
- **Three waves** replace the single submission (`submit_wave1.sh`,
  `submit_wave2.sh`, `submit_wave3.sh`, sharing `submit_lib.sh`, which shows
  `myproject -l` and asks before submitting): wave 1 (~2,000 SU) is the
  preflight, the A1 gate, 10a and 10b, Components C and B6, and B4 for Bratu;
  wave 2 (~10,000 SU) Component A and B4 for the other benchmarks; wave 3
  (~7,700 SU) B8, B9 and the finalize. `submit_all.sh` and
  `submit_component_a.sh` are removed.
- **One results folder per wave** (the user's choice, 2026-09-29):
  `results/wave<N>`, each locked to its own commit by the provenance lock,
  so a code change between waves (after the advisor's review of wave 1,
  say) is allowed and recorded rather than mixed silently. A later wave on
  new code reruns the preflight, and wave 2 the A1 gate; on wave 1's code,
  wave 2 reuses wave 1's Component A search and checks. `sbatch.sh` refuses a
  job without `LILQ_WAVE`. `90_finalize` assembles `results/package1` from
  the three folders by hard links (`scripts/cluster/assemble_package.py`),
  keeping identical files once and taking a differing one from the later
  wave, with `WAVES.json` naming every wave's commit and every override.
  Whether wave 1 must be rerun after a code change is decided if it arises.
- **`91_wave_report`** ends every wave: the run index, check B1, the merged
  four-method table, `sacct`, and `results/wave<N>_report.tar.gz` with what
  the advisor asked to see after wave 1 (models and collocation files stay
  on the cluster).

---

## 2026-09-29 -- Lazy `lilq` imports; the thread pools in effect recorded (Addendum v2.2 Section 2.12)

`lilq/__init__.py` imported every submodule, so `import lilq.blas_threads`
loaded NumPy, SciPy and PyTorch before it set the BLAS thread counts;
harmless on the cluster, where `env.sh` exports them first, but the
record should be right. Submodules now load on first use (PEP 562);
`lilq.basis` and `from lilq import basis` work as before (tested: importing
`lilq.blas_threads` loads none of the three). `hardware.json`'s `threads`
adds `threadpools`: every native pool loaded (BLAS, OpenMP) with its
library and thread count, from `threadpoolctl`.

**Check B3's amended rule, tightened** (Section 1, item 6): the six-figure
test on ||R_lin||_h is waived when |R_lin^GPU - R_lin^CPU| is at most the
CPU run's round-off floor kappa * eps_mach * ||f||_h
(`rlin_diff_within_floor`), instead of when both residuals are at or below
their floors (`rlin_at_floor`, 2026-09-24). beta must still agree to 1e-8
at every size; both verdicts are reported. The other item of Section 2.12 (Beltrami pinned before
check B3 in job 10) went in with Section 2.7.

---

## 2026-09-29 -- Beltrami pinned: the pressure error in the pin gauge too (Addendum v2.2 Section 2.11)

The snapshot and global pressure errors subtract the mean at each time
level, and the null-space modes T_j(t) * 1 are exactly those means, so the
shifted error cannot change when the pins remove the null space (0.7515%
against 0.752%). `compute_errors` and the snapshots now also give the error
as solved, without the shift (`rel_l2_p_pin_gauge`, `p_pin_gauge`), and
the Section 3.7 report gives both at t = 1. Two harmless deviations from
the spec are recorded in its report: the pins are at Chebyshev-Gauss-
Lobatto times rather than at the temporal collocation levels, and each pin
row has weight sqrt(lambda_bc / 8).

---

## 2026-09-29 -- Timing details (Addendum v2.2 Section 2.10)

1. **LiL-Q's time as the other methods report it.** The scalar benchmarks'
   `summary.json` gains `training_time` (the solver's own clock) beside
   `t_cum_s`. A paper-pass LiL-Q run shorter than 1 s (scalar benchmarks,
   Kovasznay, elasticity) is timed five more times without the log and
   reported as their median (`training_time` / `solve_time_total`), with
   all five (`timing_repeats_s`) and the logged run's own time
   (`*_single_run`) kept; +-25% run to run was seen at 0.03 s.
2. **Kovasznay: the tracker and the test-grid fields off the clock** (they
   count as diagnostics). With the log on, P = 75 now takes 0.0446 s
   against 0.0433 s without (the advisor saw about +30%).
3. **Kovasznay GPU path:** the host-to-device copy, the QR + triangular
   solve and the copy back are timed apart, between synchronizations
   (`gpu_qr.h2d_s`, `qr_solve_s`, `d2h_s` per iteration; totals in the
   summary); `t_solve_s` stays the whole solve. Check B3 adds `t_h2d_gpu_s`
   and `t_qr_solve_gpu_s` beside `t_qr_gpu_s`. The optional `geqrf` +
   `ormqr` rewrite of the GPU solve was done afterwards (entry "Kovasznay
   GPU solve: Q applied implicitly", above).
4. **`time.perf_counter()` for every duration**, replacing `time.time()`
   in `MetricsTracker`, the Kovasznay, Beltrami, elasticity and Darcy
   solvers, the Darcy PINN and the experiment scripts.

---

## 2026-09-29 -- Component C: at least P rows at N/P = 1; distinct rows logged; collocation rows saved (Addendum v2.2 Sections 2.8.3, 2.9)

- **Ratio 1.** `k_for_ratio` now takes the smallest density with at least
  P rows at the nominal ratio 1; the closest-count rule gave fewer rows than
  unknowns (97 for Bratu P = 100, 1,189 for Kovasznay P = 1,200). The
  generators' granularity makes it overshoot: Bratu P = 100 116 rows, P =
  225 228; Kovasznay P = 300 300, P = 1,200 1,308. The other ratios keep
  the closest-count rule (within 10%).
- **`N_distinct`** in `oversampling.csv`: rows of the same equation at the
  same point are identical, and the equispaced and CGL tensor grids put
  each corner on two edges. At Kovasznay P = 300, ratio 1: 300 rows, 292
  distinct (4 corners x the u and v rows), matching the advisor's count;
  scattered points have none. The ratio-1 rule counts rows, as specified,
  so the tensor grids there have slightly fewer distinct rows than
  unknowns; reported beside the nominal and actual N/P.
- **`collocation.npz` per run**: every row's coordinates, block (e.g.
  `bc_u_left`), equation, and weight, in assembly order, plus `n_distinct`,
  written by the solver itself from the points it used
  (`solve_kovasznay`/`bratu.run_lil_q` `collocation_path`); for an
  a-posteriori computation of the sampling constants c1, c2 on each grid.
- Each run's directory gets its own `hardware.json`, and each CSV row the
  commit.

---

## 2026-09-29 -- Data retention and the provenance lock (Addendum v2.2 Section 2.8)

Nothing will be rerun, so:

1. **Four-method sweep and B8: the full per-iteration history** of every
   run (`history.csv`: iteration, real evaluations, weighted loss and its
   components, wall time) next to its saved model; the CSV's every-10-
   iteration history gains the wall time (`[iteration, loss, time]`).
   Time to the manuscript targets and to a common loss are read from these.
2. **B8 writes the LiL-Q log in a `finally` block**, so a run that
   diverges or raises keeps its chi_k history.
3. **Provenance lock** (`lilq/source_lock.py`). The bundle records its file
   list and a source-tree hash (SHA-256 over each file's path and content,
   text normalized to LF). The first job on a package root writes
   `$PKG/COMMIT` (commit and tree hash); `env.sh` checks it before every
   job, which refuses to run if the commit or the tree hash differs, if the
   files on disk no longer match the bundle (edited on the cluster), if the
   bundle was built from uncommitted changes, or if the version is
   unknown. `make_hprc_bundle.py` refuses a dirty tree unless
   `--allow-dirty`. The commit is recorded in every `run.json` (all
   LiL-Q runs and Component A), `summary.json`, and every row of the
   four-method, B8 and oversampling CSVs.

---

## 2026-09-29 -- K_max = 60 pass; Beltrami conditioning every iteration and a K_max = 8 pass; retained kappa and ||beta|| logged (Addendum v2.2 Section 2.7)

For the table of where the proposed stopping rule would stop:

1. **The `kmax` pass runs K_max = 60** (`component_b.KMAX_PASS_ITERS`) for
   Bratu, Burgers, both Buckley-Leverett cases and Kovasznay, with the
   stopping rule disabled, instead of reusing the paper caps of 20-30. BL
   gravity P = 64 reaches its plateau only at k ~ 41-45. The `paper` pass
   is unchanged.
2. **Beltrami, unpinned and pinned (Section 3.7):** the pivoted QR of the
   weighted matrix at every iteration (`BeltramiConfig.
   conditioning_every_iteration`, default on; off the clock, about 70 s per
   iteration at P = 7,984), and a `kmax` pass of 8 iterations with a zero
   coefficient-change tolerance for both (`component_b` for the unpinned
   run, `run_beltrami_pinned.py --kmax` for the pinned one). Before, kappa
   existed only at the final iterate above P = 3,200, so the round-off exit
   could not be evaluated for the largest problem.
3. **`iterations.csv` gains four columns after the spec's thirty**, which
   keep their order: `kappa_raw` (sigma_max/sigma_min, or the pivoted QR's
   |R_11|/|R_PP| over all diagonal entries), `kappa_retained` (sigma_1/
   sigma_r with r the numerical rank, or the QR's retained-diagonal ratio),
   `num_rank_qr` (the QR's retained diagonal entries), and `norm_beta` =
   ||beta^(k)||_2, the iterate row k was assembled at (the terminal row:
   the returned coefficients). At BL P = 1,024 the full kappa is about 1e17,
   so kappa * eps_mach ~ 22 and the round-off comparison fires trivially;
   the retained part's is the meaningful one. The `kappa` column keeps its
   meaning (SVD: full ratio; QR: retained ratio).
4. Elasticity and Darcy: no extra runs (linear, one solve).

Job 10 now also runs the pinned K_max pass, and runs Beltrami pinned before
check B3 (Section 2.12: under `set -e` a B3 failure would have skipped it).

---

## 2026-09-29 -- Component A, F1: unweighted selection, best point in budget, criterion, logging, peak memory (Addendum v2.2 Section 2.5)

1. **Selection on an unweighted loss.** `run.json` gains
   `final_loss_unweighted`: the x-momentum, y-momentum and continuity mean
   squares, plus the soft boundary mean square with weight 1. `component_a.py`
   ranks the screening runs and picks the representative on it for F1 (the
   balanced weights are each at least 1 and soft runs carry up to 100x the
   boundary term, so the weighted loss is not comparable across
   configurations); F2 keeps its final loss, whose weights are the same for
   every configuration. `<family>_selection.json` names the loss used.
2. **An interrupted call's best point is logged inside the budget.** The
   restored lowest-loss point is logged at the time its evaluation
   finished (`best["t"]`), with test errors, not after the budget, so it
   counts in "best within budget" and time-to-accuracy. An evaluation that
   finishes past the budget does not count as reached.
3. **Criterion.** L-BFGS runs with `tolerance_grad = tolerance_change = 0`
   (PyTorch's defaults are absolute and ended runs early, with budget left,
   at configuration-dependent accuracy). A call that does not lower the
   loss is followed by one call with a fresh optimizer; the run ends only
   if that call does not lower it either. A later call that makes progress
   re-arms the retry. `run.json`: `lbfgs_restarts`. This is our reading of
   v2.0 Section 4.3's "the family's own criterion", with the advisor's
   agreement.
4. **Logging inside calls.** A call (up to 500 iterations and 625
   evaluations, PyTorch's `max_iter` and default `max_eval`) now runs in
   pieces of 10 iterations with the optimizer state carried over, and
   `log.csv` gets a row after each piece (v2.0 Section 4.4). Every point is
   evaluated once (`LBFGSObjective`), so the piece boundaries add no
   evaluations and the rows' loss, terms and gradient norm come from
   evaluations L-BFGS already made; a test checks that a call run in pieces
   ends at bitwise the same parameters, with the same evaluation count, as
   one piece. `run.json`: `lbfgs_iters`, `lbfgs_evaluations`.
5. **Peak memory.** `torch.cuda.reset_peak_memory_stats()` at the start of
   every F1 and F2 run, so `peak_gpu_bytes` is that run's own.

---

## 2026-09-29 -- PyTorch 2.10.0 on the cluster, enforced (Addendum v2.2 Section 2.4)

Grace's newest module is `PyTorch/2.9.1-CUDA-12.6.0`. Its L-BFGS calls the
strong-Wolfe line search without `max_ls`, so a step can make up to about
27 evaluations regardless of `max_eval`; 2.10.0 passes `max_ls=max_eval -
current_evals` (both checked on Grace by reading `LBFGS.step`). The
evaluation counts and the 16x cap above assume 2.10. The job venv
therefore gets `torch==2.10.0` (CUDA 12.6 wheel) by pip. The module
still supplies Python, numpy 1.26.4 and scipy 1.13.1, but it also puts
its own torch on `PYTHONPATH` ahead of the venv (installing 2.10.0 alone
still imported 2.9.1), so `env.sh` puts the venv's site-packages first.
The preflight fails unless the imported torch is 2.10.0
(`lilq.provenance.assert_torch_version`); `environment.txt` records the
version, the path it was imported from, and `torch.__config__.show()`.

---

## 2026-09-29 -- L-BFGS: each point evaluated once; 16x evaluation cap; `optimizer_stall` (Addendum v2.2 Sections 2.1-2.2)

**Each point evaluated once.** PyTorch's L-BFGS starts every `step` with a
closure call at the point the previous line search already evaluated, and
the loops' stopping test then evaluated the accepted point once more; both
were on the clock and the first was counted. `lilq.solvers.LBFGSObjective`
stores the loss and gradient of every evaluated point (up to 32) and
returns them, gradient restored into `.grad`, whenever the parameters are
bitwise equal to a stored point. The optimizer sees exactly the values it
would have recomputed: on the CPU the loss trajectory and the final
parameters are bitwise identical to the earlier loops over 300 iterations
(Bratu P = 100, NiL-N and LiL-N; NiL-Q over 3 x 40), with real evaluations
636 -> 337 and time 1.77 -> 0.97 s for NiL-N (the advisor measured 623 ->
324, 5.11 -> 2.60 s). `total_line_searches` now counts real evaluations
only. NiL-Q's store is cleared at each linearization (the objective
changes); its stopping test is the full nonlinear loss, a different
function, evaluated after every inner step and counted separately
(`monitor_evaluations`). `memoize=False` reproduces the earlier loops, for
the tests only. Every baseline time in the four-method tables drops by
about half; LiL-Q is unaffected (no line search).

**Cap.** `LINE_SEARCH_CAP_FACTOR` 3 -> 16. With PyTorch 2.10 a step with
`max_eval=15` makes at most 16 evaluations (`_strong_wolfe` gets
`max_ls=max_eval - current_evals`; checked in the installed 2.10.0), so
the cap cannot bind before the iteration cap. The 3x cap left about one
spare trial point per iteration and could only shorten the gradient-trained
methods.

**`optimizer_stall`.** NiL-N and LiL-N end when an L-BFGS step returns with
the parameters bitwise unchanged (the optimizer state is then unchanged,
so every later step would be the same no-op); the stall's iteration
(counting the no-op step), evaluation count and time are recorded. In
NiL-Q a stalled inner step ends that outer iteration (`n_inner_stalls`);
the run ends with `optimizer_stall` only when a whole outer iteration
leaves the parameters unchanged. The target test comes first. NiL-Q keeps
one L-BFGS object across outer iterations, so curvature pairs carry over
between linearizations -- the method behind the published tables, kept
and to be stated in the report.

---

## 2026-09-29 -- Stopping reasons: one rule for the four-method and B8 CSVs (Addendum v2.2 Section 2.3)

`lilq.four_method_log.stopping_fields(method, summary, opt)` gives both
drivers the same iterations, caps, reason and stall fields. NiL-Q is now
classified on its inner-iteration total against (outer cap) x (inner cap);
the four-method CSV used to report the outer count against the outer cap
while B8 reported the product, so `iterations_cap` meant different things
in the two files. Reasons, in order: `failure` for a non-finite final
loss; `target`; `optimizer_stall`; `line_search_cap` when the evaluation
cap is reached with iterations below theirs (an anomaly now); then
`iteration_cap`, including NiL-Q runs that used every outer iteration
after inner loops ended early on stalls. Both CSVs gain `stall_iteration`,
`stall_evaluations`, `stall_time_s`.

---

## 2026-09-28 -- Every trained model is saved, reloadable; long runs checkpoint

Nothing will be rerun after the Grace pass, so every run keeps what later
analysis and figures need (`lilq/saved_models.py`):

- **LiL runs** (Component B's Section 3.3 runs on every benchmark, LiL-N in
  the four-method sweep and B8, LiL-Q in B8, Component C, the B6 basis
  study, Section 3.7, B9's LiL): `solution.pt` next to the run's logs,
  holding the basis objects, the final coefficients and the configuration
  objects. `load_solution`, `evaluate_field`.
- **Network runs** (NiL-N and NiL-Q in the four-method sweep and B8):
  `models/<run>/network.pt`, the network, its `state_dict` and the
  configuration objects. `load_network`.
- **Component A:** F1's `model.pt` and F2's `theta.pt` were already saved;
  they are now written before `run.json` (the completion marker), and
  `load_f1`, `load_f2`, `predict_f2` rebuild the models from `run.json`.
- **B9's Darcy PINN** (2-4 h per run): a checkpoint every 5,000 epochs
  (networks, Adam and scheduler states, history, time so far); a rerun
  of the job resumes from it and continues exactly as if it had not
  stopped (a test crashes a run and checks the resumed weights are
  bit-identical). The trained networks go to `network.pt`
  (`problems.darcy.load_darcy_pinn`), which a rerun loads instead of
  training again.

Models are saved when a run ends, outside every timed phase, and before
the row or file that marks the run complete, so a completed run always has
its model. Runs other than B9's are at most minutes to an hour, and a
crash reruns only the run in progress (the drivers already resume per
run); a timed run cannot be resumed mid-way without corrupting its time.
A failed save in the four-method sweep is recorded in the row's `error`
column instead of stopping the sweep. All files are `torch.save` pickles of
this repository's classes, loadable on a CPU-only machine, with the code
at the commit in the run's `hardware.json`. Sizes: kilobytes to a few MB
per run, about 25 MB for a Darcy checkpoint.

---

## 2026-09-28 -- Buckley-Leverett test error: against a reference solution, not the residual

The spec sets no test metric for Buckley-Leverett; the log had used the
PDE residual's mean square on a uniform 201 x 201 grid, as for Bratu. On
the gravity case it ranked LiL-Q ten times worse than LiL-N at P = 1,024
(3.6 against 0.65) although LiL-Q's training loss is lower. Investigated
before the Grace runs (P = 1,024, `cos_fourier`, paper settings):

- On the collocation points the PDE residual's mean square is 0.0072
  (LiL-Q) and 0.0083 (LiL-N), in line with the training losses: the
  solves are correct.
- On the test grid, 1% of the points carry 99% of the metric. They lie
  on the t = 0 line and in the first few time levels, where the IC's
  steepness-100 front has second derivatives no 32-mode basis resolves
  (mean r^2 over x: 91 at t = 0, 0.02 by t = 0.0175, for both methods),
  and on the domain edges, outside the collocation points' extent (t up
  to 0.1725 of 0.175; LiL-Q's r^2 reaches 1.6e4 at the corner x = 1,
  t = T). The metric measured behaviour where no method is trained, and
  it did not converge with the grid (3.6, 2.5, 2.0 at 201, 401, 801).
- Against a finite-difference reference solution the two methods are
  about equally accurate: relative L2 error of S 1.16e-2 (LiL-Q) and
  1.05e-2 (LiL-N); max error 0.045 and 0.080.

Change: `eps_u` is now the relative L2 error of S, and `maxerr_u` the
maximum absolute error, against `reference_solution(config)` on the same
201 x 201 grid. The reference solves the same PDE with the same data by
second-order central differences on 4,000 intervals (method of lines, BDF,
rtol 1e-9); 2,000 against 4,000 intervals differ by 1.3e-6 relative, and
it takes 0.3 s. It is saved under `reference/bl*.npz` (field `S`). At the
paper sizes, LiL-Q's eps_u falls with P in both cases (viscous 9.9e-2,
3.8e-2, 9.0e-4, 1.9e-4; gravity 1.2e-1, 3.2e-2, 2.1e-2, 1.2e-2). Bratu and
Burgers keep the residual metric: there it tracks the training loss at
every size (checked).

Affects only the logged test-error columns of Buckley-Leverett runs; the
manuscript reports no test metric for this benchmark (Tables 5, 6: target
loss, iterations, time).

---

## 2026-09-25 -- Everything on the cluster; one set of job scripts for Grace or FASTER

The user's rule: all runs on one platform. The untimed work the addendum
allowed on the laptop now runs on the cluster too: B8 (`41_b8_initial_guess`,
one task per case and guess, networks on a shared A100 as in the
four-method GPU pass, so its rows are comparable with Tables 4 and 5),
Component C (`42_component_c`, CPU) and B6 (`43_basis_study`, CPU).
`90_finalize` merges B8's four CSVs.

`scripts/grace/` became `scripts/cluster/`: job scripts carry only their name,
time and array plus a `# lilq-resources:` class (timed / shared-gpu / cpu);
`sbatch.sh` applies the class's resources from `profiles/grace.sh` or
`profiles/faster.sh` (account, partitions, A100 gres, cores per node, module
names), and `submit_all.sh` checks the profile before submitting anything.
Deleted as stale: `scripts/hprc/` (the FASTER shakedown scripts, superseded)
and `experiments/validate_pre_hprc.py` (built configurations by hand with
pre-decision budgets).

---

## 2026-09-25 -- F2 (Levenberg-Marquardt) made faster; same algorithm

Addendum v2.1 Section 6 allows speeding up the reference provided the
finite-difference and A2 checks pass and the step equation, damping
schedule, acceptance rule and stopping rule are unchanged. They are. Why it
matters: the budget is wall-clock, so a faster step means more LM steps and
a stronger F2 baseline. Changes in `baselines/lm_kovasznay.py`:

1. **Batched Taylor-mode residual.** The reference evaluated each interior
   point separately, with three nested forward-mode passes through
   `functional_call` (thousands of tiny kernels). `taylor_forward` pushes the
   value, first derivatives and the two second derivatives the Laplacian
   needs through the network for a whole block of points at once, exactly
   (the Fourier embedding and Linear layers are linear maps; tanh' = 1 -
   tanh^2, tanh'' = -2 tanh tanh'). The Coons interpolant and the distance
   factor do not depend on the parameters, so their derivatives are
   computed once (`boundary_geometry`). Agreement with the reference's
   evaluation: 3-4e-16 relative.
2. **Normal equations by row block.** J^T J and J^T r are accumulated block
   by block; the full Jacobian is never stored (the reference formed J, a
   concatenated copy, then J^T J).
3. **Damping in place** on a copy of J^T J instead of adding a dense
   `diag(d)`.

Measured on this laptop's GPU (weak in float64), per LM step, reference ->
optimized: 1.62 -> 0.49 s (4,291 parameters, 3,001 rows), 4.47 -> 2.03 s
(7,395; 6,001), 5.08 -> 3.45 s (12,675; 3,001); the steps agree to 1e-11
relative (rounding, amplified by the conditioning of J^T J). The Jacobian
itself went from 1.74 to 0.61 s at 12,675 parameters; what remains is
dense float64 linear algebra (J^T J 1.66 s, Cholesky 1.20 s), which an
A100 runs 10-20x faster, so on Grace the step becomes Jacobian-bound.

End to end, the advisor's smoke run reproduced with the optimized code (this
laptop's CPU): the same 60 steps and 115 function evaluations (the same
accept/reject sequence), final loss 1.8783366e-9 against his 1.8783366e-9,
eps_u 9.0925575e-5 against 9.0925571e-5; loss history within 9e-8 relative
throughout. Tests: the batched residual and the block normal equations
against the reference forms; the finite-difference and A2 checks.

---

## 2026-09-25 -- Code review before the Grace runs: fixes

A full read-through against Package 1 v2.0, Addendum v2.1 and the decisions
above. The core monitors (chi_k, o_k, stall flag with tau = 0.1, round-off
comparison, SVD/pivoted-QR conditioning and numerical rank) match the
spec's formulas. Fixed:

1. **F1 could overrun its budget by a whole L-BFGS call.** The clock was
   checked only between calls, and a call runs up to 500 iterations: for
   the largest configurations (three 256-wide networks, 32,000 points) that
   could be minutes against a 10-minute screening budget, unequal across
   configurations. The budget is now checked at every evaluation inside a
   call; an interrupted call keeps the lowest-loss point L-BFGS evaluated.
   Overshoot: at most one evaluation.
2. **Section 4.6 "best within budget" counted rows logged after the budget**
   (an LM step can end just past it). Only rows with t <= budget count, and
   the end-of-run values only if the run did not overrun.
3. **Race in Component A's search.** Both screening array tasks saved the
   search at the same moment; one could read a half-written file. The
   preflight saves it once. The tuning-log header is created atomically.
4. **An F1 run failing before training** (e.g. out of memory building a
   network) would have stopped the stage; it is now a logged failure, as
   for F2.
5. **Component C mislabelled a Kovasznay run that met its tolerance on its
   last allowed iteration** as K_max; the label now comes from the final
   coefficient change.
6. **B1 called round-off a violation.** Elasticity's relative errors (1e-16
   against the paper's 1e-15) failed the factor-2 test on the FASTER run.
   Error entries at or below 1e-13 on both sides are now `round-off`
   (still reported, like B3's amended rule): 8 of the FASTER run's 23
   violations; the 15 left are all known.

Checked, not changed: the FASTER run's Burgers P = 625 LiL-Q time (6.07 s
against the paper's 1.26 s) was the diagnostics on the clock, fixed on
2026-09-24; locally, with full logging, 0.94 s (0.91 s without logging).

---

## 2026-09-24 -- Component C: oversampling sweep; CGL and scattered collocation

`experiments/component_c.py` runs Section 5's 196 LiL-Q runs (Kovasznay P =
300, 1,200; Bratu P = 100, 225; N/P in {1, 1.5, 2, 3, 5, 10, 20}; paper
construction, CGL tensor grid, uniform random points with seeds 0-4) to the
paper's stopping rule with the full Section 3.1 log, and writes
`results/oversampling.csv` and, per benchmark and P, final ||R_lin||_h and
kappa_2 against N/P (random: median with min-max band). Resumable; a
failing run is logged with its traceback.

Choices where the package is open:
- **Hitting N/P.** For each target, the density factor `k_ratio` whose
  actual row count is closest (proportions of row types as in the paper;
  exact counts recorded). Actual N/P lands within 3% of every target
  (0.97-1.01 at N/P = 1).
- **Minimum point counts lowered to 1 in the sweep.** The paper's
  generators enforce at least 10 points per direction and per edge
  (Kovasznay) and 5 and 10 (Bratu). At Kovasznay P = 300 those floors alone
  give ~381 rows, so N/P = 1 would be unreachable, and at small N/P they
  distort the row-type proportions. `KovasznayConfig.collocation_floor`
  (default 10) and `BratuConfig.collocation_floor` (default None: the
  paper's) make them settable; the sweep uses 1 for every run.
- **Distributions.** `lilq.collocation.points_1d`: 'cgl' is the
  Chebyshev-Gauss-Lobatto family (endpoints included; interior grids inset
  by 1e-6 like the paper's), and 'scattered' draws the same number of
  interior points uniformly at random (not a tensor grid). Boundary points
  follow the same family along each edge. The paper's constructions are
  unchanged: bit-identical point sets for 'random' and 'uniform' (checked
  against the previous code).

---

## 2026-09-24 -- Component A driver; F2's test errors off the clock

`experiments/component_a.py` runs Section 4.3 as resumable stages (search,
screen, select, full, cpu, float32, a1, f2-jacobian) under the package's
`A_calibration/` layout, appending every run to `tuning_log.md` in the
order tried (and a "manual intervention" line whenever a stage runs with a
non-package budget). Screening: every configuration, seed 0, 600 s.
Selection: final training loss only. Full: top 3 x seeds 0-4, 3,600 s;
representative = lowest median final training loss; the best test errors
over the 15 runs are reported separately. Test errors are logged at every
log row (every 100 Adam iterations, every L-BFGS call, every LM step), for
time-to-accuracy.

F2 change, not to the algorithm: the reference computed its test errors
inside the clock, so they counted against its budget, while F1's do not.
`baselines/lm_kovasznay.py` now excludes their time from `t_cum_s`, the
budget and `wall_s` (recorded in its docstring). Check A2 still passes.

---

## 2026-09-24 -- Component A, F1 (modern first-order PINN) and the random search

`baselines/search.py`: 24 configurations per family, generator seed 12345
(one `numpy.random.default_rng(12345)` per family), F2 draws over
n_theta = 20,000 redrawn (15 of them), saved once to `search/F*_configs.json`
and never silently replaced. Choice where the package is open: F1's
"hard or soft (lambda_bc in {1, 10, 100})" is drawn as hard/soft with equal
probability, then lambda_bc if soft.

`baselines/f1_pinn.py` implements Section 4.2's F1. Choices where the
package is open, to go into the tuning log:
- Loss balancing: the gradient-norm form of the cited guide (Wang, Sankaran,
  Wang, Perdikaris, arXiv:2308.08468): lambda^_i = sum_k ||grad L_k|| /
  ||grad L_i||, lambda_i <- 0.9 lambda_i + 0.1 lambda^_i, every 100 Adam
  iterations, over the momentum, continuity and (soft) boundary terms;
  the weights are frozen during L-BFGS.
- Learning rate: warm-up factor min(1, (it+1)/1000) times 0.9^(it // 2000),
  both counted from the first Adam iteration.
- No pressure pin in F1's loss (the package lists none); baselines are
  compared with the mean-free pressure error.
- The run ends on the budget, on an L-BFGS call that does not lower the
  loss (the family's criterion), or on a non-finite loss. An L-BFGS call is
  never interrupted (up to 500 iterations), so a run can overshoot the
  budget by one call; the logged time says by how much.
- Test errors (301 x 401 grid) are computed off the clock, every 10th log
  row and at the end.
- Networks are initialized in the run's own precision.
- Check A1's plain PINN (Fourier features off, balancing off, soft with
  lambda_bc = 10; the rest unspecified): width 128, depth 4, shared trunk,
  eta = 1e-3, T_Adam = 20,000, N_int = 8,000, no resampling
  (`f1_pinn.A1_CONFIG`).
Tests check F1's residual against the F2 reference's on the same weights
(1e-12), the exact solution's residual, hard boundary values, the schedule
and balancing formulas, determinism, logging, and the float32-Adam switch.

---

## 2026-09-24 -- Component A, F2: the advisor's Levenberg-Marquardt reference in the repo

`baselines/lm_kovasznay.py` is `lm_reference/lm_kovasznay_reference.py`
with one change, recorded in its docstring: `torch.set_default_dtype
(torch.float64)` moved from import time into `lm_train`/`main`, so
importing the module does not change PyTorch's default precision for the
rest of the process (the same side effect removed from `lilq` on
2026-09-22). The algorithm is untouched. `tests/test_lm_kovasznay.py`
reproduces the advisor's checks: boundary values (< 1e-14), residual
against an independent autograd evaluation (< 1e-12), the exact solution
satisfying the equations (< 1e-12), the Jacobian against central
differences (< 1e-7 relative; advisor 6e-10), and check A2 (two seed-0
runs, identical loss histories over 20 steps). A2 needs a problem the
network cannot fit exactly: with 40 interior points (121 rows, 273
parameters) the loss reaches 1e-31 and the run ends when the damping
overflows, at a step that depends on rounding; 150 points is used.

---

## 2026-09-24 -- Task B8: Buckley-Leverett initial-guess sensitivity

`BLConfig.initial_guess` ('zero' or 'ic'; None keeps the paper's choice:
zero with gravity, the profile without) sets the starting point of all four
methods through `BLPhysics.initial_guess`: the least-squares fit of the
coefficients (LiL-Q, LiL-N; `pretrain_lil`) and the network pretraining
(NiL-N, NiL-Q; `pretrain_nn`). 'ic' is the initial saturation profile
extended in time (constant in t). `run.json` records which.

`experiments/b8_initial_guess.py`: the 128 runs (2 cases x 2 guesses x 4
sizes x LiL-Q, LiL-N, NiL-N and NiL-Q with seeds 0-2), from
`run_bl.paper_setup` (paper budgets and targets; gravity: `cos_fourier`
and the retargeted losses). LiL-Q stops only on its target or at K_max and
keeps its full log in `lilq_logs/`; its CSV row adds the first stall
iteration, whether the stall flag ever fired, and the chi_k history.
Resumable; failures logged with tracebacks. CPU by default (untimed).

LiL-Q at full budget, P = 64 and 256, this laptop (iterations; * = K_max
reached without the target):

| Case, guess | P = 64 | P = 256 |
|---|---|---|
| viscous, zero | 6 | 5 |
| viscous, ic (paper) | 10 | 4 |
| gravity, zero (paper) | 20* (loss 0.951, stall flag from k = 4) | 12 |
| gravity, ic | 14 | 20* (loss 0.099) |

---

## 2026-09-24 -- Four-method harness used the wrong NiL pretraining budget

`experiments/four_method_tables.py` rebuilt each benchmark's configuration
by hand and dropped two settings that the paper scripts
(`experiments/run_*.py`) pass: `pretrain_epochs` (1,000 for Bratu and
Buckley-Leverett; the dataclass default is 500) and `max_quasi_iters_lil`
(20 for gravity; default 50). The first changes the NiL-N/NiL-Q starting
network, so every four-method NiL row for Bratu and Buckley-Leverett so far
used half the paper's pretraining (Burgers uses 500 in both, unaffected).
Those rows (the FASTER run) were already superseded. The second did not
matter there (no LiL-Q in the four-method tables) but would have in B8.
The Section 3.3 driver and the residual-band figures only run LiL-Q, which
uses neither setting.

Fix: each paper script has `paper_setup(N, ...)` returning its
`(config, opt)`; the script itself, the four-method harness and the
residual-band builders all use it, and B8 will too. A test asserts the
harness's configs equal `paper_setup`'s for every size and benchmark.

---

## 2026-09-24 -- Task B9: Darcy pressures against the FVM solution; NiL in float64

`experiments/darcy_fv_comparison.py` (Addendum fix 3.4, second script):
per field and method, delta_FV = ||p_h - p_FV|| / ||p_FV - 3,000 psi|| on
the 60 x 220 cell centres, the TPFA system's residual at the FVM solution,
and `fvm_rel_L2` (normalized by ||p_FV||) kept for comparison only. No
`np.gradient` divergence. `solve_fvm` is split into `assemble_fvm` +
solve (same numbers); `delta_fv` and `tpfa_residual` in `problems/darcy.py`;
`solve_lilq_darcy` now returns `P_lil` and `metrics['delta_fv']`.

LiL, this laptop, order 32 (the paper's), confirming the advisor's values:

| Field | delta_FV | advisor | TPFA residual (relative) |
|---|---|---|---|
| S1 | 1.373e-4 | 1.4e-4 | 5.7e-15 |
| S2 | 2.331e-4 | 2.3e-4 | 1.6e-14 |
| S3 | 6.736e-4 | 6.7e-4 | 6.8e-16 |
| SPE10 | 3.421e-2 | 3.4e-2 | 8.8e-14 |

**NiL (`DarcyPINN`) is now float64 by default**, per Package 1 v2.0
Section 2 ("float64 everywhere"); it had been forced to float32, and the
2026-09-22 entry left it so only because no package task ran it. B9 does.
`dtype=torch.float32` remains available, and a `seed` argument serves the
three-seed runs. Measured cost per Adam epoch on this laptop (the default
3 networks x 200 wide x 8 layers, 13,200 collocation points):

| Device, precision | ms/epoch | 150,000 epochs |
|---|---|---|
| RTX 5080, float64 | 561 | 23.4 h |
| RTX 5080, float32 | 76 | 3.2 h |
| CPU, float64 | 1,132 | 47 h |
| CPU, float32 | 704 | 29 h |

B9 needs 4 fields x 3 seeds = 12 NiL runs: about 280 h in float64 on this
laptop. Open: where and in what precision to run them (the user's call).

---

## 2026-09-24 -- Addendum fixes 3.1 and 3.3; diagnostics taken off the solver clock

**Fix 3.1 (Buckley-Leverett LiL-Q iteration count).** The shared
`solve_lil_q` already counts solves. Tests added for what the addendum
asks: a run that stops after k solves reports k (k = 2, 3, 7), and a real
Buckley-Leverett LiL-Q run whose outer least-squares solves are counted
independently reports exactly that count.

**Fix 3.3 (no PyTorch/CUDA setup in CPU-only LiL-Q drivers).** The
LiL-Q-only drivers (Beltrami, Beltrami pinned, Kovasznay, elasticity;
Darcy when its NiL part is skipped) and the `run_lil_q` of Bratu, Burgers
and Buckley-Leverett now seed NumPy only, instead of `set_seed`. LiL-N
keeps `set_seed`: it trains with PyTorch. Measured whether PyTorch's
presence slows the CPU solve at all: a 24,000 x 5,000 `gelsy` solve
(Beltrami's dominant operation), three repeats per fresh process, two
rounds, 24 threads on this laptop:

| Process | min / 3 (round 1) | min / 3 (round 2) |
|---|---|---|
| NumPy/SciPy only | 36.2 s | 37.8 s |
| + `import torch` | 38.1 s | 37.2 s |
| + CUDA probe (`is_available`, cuDNN version) | 36.0 s | 38.4 s |
| + full CUDA context | 37.7 s | 38.4 s |

No effect beyond run-to-run noise (about 5%), so PyTorch is still
imported by these modules; removing it would touch every problem module
for no measured gain. The 2026-09-22 entries already found that
`set_seed`'s CUDA settings cost ~20 s of ~540 s and that the real
Beltrami regression was the per-iteration SVD.

**Diagnostics taken off the solver clock.** The Section 3.1 diagnostics
(per-iterate test errors, check B2, SVD or pivoted-QR conditioning) are
passive, but their time was inside the reported solver times:
`solve_time_total` of Kovasznay, Beltrami and elasticity, Darcy's
`total_time`, `solve_lil_q`'s `training_time`, and the Section 3.7
"wall-clock". B1 compares Kovasznay's `solve_time_total` with Table 9,
and with logging on the tracker adds an SVD every iteration. Each solver
now times its diagnostics block and subtracts it (`diagnostics_time` is
reported alongside; `QuasilinearMetrics.exclude_time` does it for
`solve_lil_q`). `t_cum_s` in `iterations.csv` (assembly + solve) was
already clean and is unchanged. Section 3.7 now reports the solver time,
the diagnostics time and the process wall-clock separately.

Beltrami on this laptop, paper configuration, full logging, nothing else
running: 4 iterations, **solver time 308.1 s** (assembly 2.2-2.4 s and solve
73.6-75.5 s per iteration; `t_cum_s` 306.6 s), diagnostics 75.8 s, process
wall-clock 384.1 s; errors u 3.445e-4, p 2.614e-3 (Table 11's 0.0345% and
0.261%). The paper says about 300 s; the pre-GitHub code measured 302 s on
2026-09-22. The Section 3.7 figure of 363 s recorded earlier included the
diagnostics.

---

## 2026-09-24 -- Addendum v2.1: reference codebase, budgets, line-search caps, BL-gravity

**Reference codebase.** Addendum v2.1 Decision 2 names the GitHub version
plus four fixes. We use this branch (v3-dev) instead, and will say so to
the advisor. It fixes bugs the GitHub version has (the BL LiL-N gradient,
the ELM class and its memory use, per-iteration SVDs in Kovasznay and
Beltrami, seeds that moved the collocation grid, unpinned threads, an
import-time dtype side effect) and adds the instrumentation the package
requires. Where GitHub made a *choice* rather than a mistake, it did
better than the pre-GitHub values restored here on 2026-09-22 (which
were restored only to reproduce the old manuscript tables, now replaced),
so those choices are GitHub's again:

1. **Pretraining grid** (`cabdc71`): one fixed 50 x 50 grid for every
   problem and size, for the NN fit and the LiL coefficient fit alike
   (`pretrain_grid = 50`; LiL uses max(50, ceil(sqrt(2P))) per direction,
   50 at every P <= 1,250). The restored per-problem grids were sparser at
   every package size (7 x 7 at Bratu P = 25; about 2 points per
   coefficient for the LiL fit) and used floors of 50/100/50 for no stated
   reason. The 2026-09-22 entry called GitHub's LiL formula a structural
   bug; it is the denser fit, and that label was wrong.
2. **Bratu budgets** (`a0b70ed`): P = 100 back to 10,000 iterations
   (NiL-N/LiL-N) and 25 x 400 (NiL-Q).
3. **Line-search caps, standardized** (this commit): the cap on function
   evaluations is 3x each method's own iteration budget, for every
   problem -- `max_iterations` for NiL-N and LiL-N,
   `max_quasi_iters_nn * max_inner_iters_nn` for NiL-Q
   (`lilq.solvers.line_search_cap`; an explicit `max_line_searches`
   overrides it). GitHub's caps, as multiples of the iteration cap, were
   Bratu 3 / 2.5 / 2.5, Burgers 30 / 15 / 15 / 15 / 15 and BL 3, with
   NiL-Q sharing NiL-N's cap whatever its own budget. L-BFGS makes at most
   15 evaluations per iteration, so Burgers' caps could never bind; with
   3x they can. A typical iteration uses about 2 evaluations. This is a
   deliberate deviation from GitHub, to be disclosed: it gives every
   method the same evaluation budget per allowed iteration. The addendum
   asks for runs to end on a target, the iteration cap, the line-search
   cap or failure, so a cap that can bind is part of the protocol.
   Replaces the 2026-09-22 "made inert" BL entry and the Bratu
   `MAX_LINE_SEARCHES` revert.

**BL-gravity: `cos_fourier` and the retargeted losses stay.** The user's
point: the paper's targets were also set a margin above what LiL-Q
reached, so the retargeted values follow the same rule, measured against
LiL-Q's true floor. The remaining question was LiL-Q at P = 64
(`cos_fourier` does not reach the target from zero within K_max = 20).
Diagnostic (scratchpad, not in the repo): with J = A(beta) (the verified
Jacobian) and S = sum_i r_i Hess(r_i), the term Gauss-Newton drops (built
by central differences of J^T r), rho = max|eig((J^T J)^{-1} S)| --
the local GN rate at a solution:

| Case | final loss | rho at the solution | rho along the path from zero (k = 1..12) |
|---|---|---|---|
| `cos_fourier` P=64 | 0.2283 | 0.78 | 9.4, 1.5, 1.07, 1.25, 1.41, 1.64, 1.86, 1.94, 2.32, 2.07, 2.73, 2.00 (loss 0.51 -> 0.82) |
| `cos_fourier` P=256 | 0.0706 | 0.65 | 31, 17, 2.6, then 0.96-1.08, then 0.6 by k = 11 |
| `fourier` P=64 | 0.1968 | 0.51 | 4.0, 3.9, 4.4, 2.1, 7.9, then 0.5 from k = 6 |
| `fourier` P=256 | 0.1230 | 0.63 | 16, 6.8, 1.3, then 0.6-0.8 from k = 4 |

Every solution attracts the undamped Gauss-Newton iteration (rho < 1;
linear, not quadratic, convergence -- this problem is large-residual at
every size). What separates `cos_fourier` P = 64 is the path: the other
three reach a region with rho < 1 within 4-6 iterations; this one settles
near loss 0.5, where the dropped term outweighs the kept one (rho
1.1-2.7), the steps raise the loss, and the iteration never reaches the
basin. Consistent with: one step to the floor from LiL-N's solution; 14
iterations from the fitted initial profile (B8); backtracking along the GN
direction stalling at 0.47. Not explained: why this path lands there for
this basis and size. For the report: LiL-Q is an undamped Gauss-Newton
iteration, it converges from inside the basin, and at P = 64 from zero it
is caught where its own linearization is poor -- the regime the phase
indicator chi_k monitors.

---

## 2026-09-24 -- First HPRC run (FASTER): ELM memory fix; check B3 at P = 1,875

**ELM derivatives are now closed-form.** Job `lilq-b36-b37` was killed
for exceeding 32 GB in the Section 3.6 basis study, on the ELM row.
`ELMBasis2D_Xavier.derivative` built each of the 625 columns with
autograd under `create_graph=True`, keeping every column's graph alive
(peak 11.5 GB on the laptop, over 32 GB on FASTER's Linux nodes). For tanh
and sigmoid the derivatives have closed forms,
$s^{(k)}(z)\,(2\alpha/L_x)^{d_x}(2\beta/L_y)^{d_y}$, now used directly:
1.05 GB peak, 31 s for the ELM row. A test checks them against autograd to
1e-12 for every derivative order and both activations. The ELM row moves
at the rounding level only -- final $\|R\|_h^2$ 0.049469 vs 0.049472,
ranks unchanged (SVD 43, gelsy 70); the matrix has $\kappa \approx
1.3\times10^{18}$, so rounding-level column changes shift the
rank-truncated solution slightly. Paper: 5.0e-2. The ELM basis is used
only by the basis study, so no other finished run is affected. Section
3.6/3.7 is rerun on this commit.

**Check B3 fails at P = 1,875 on the $\|R_{\rm lin}\|$ criterion alone.**
On the A30: $\beta$ agrees to 6.6e-14 (criterion 1e-8), but the final
$\|R_{\rm lin}\|_h$ differs by a relative 1.3e-3 (criterion 5e-7, "six
significant figures"). Reproduced on the laptop GPU (8.9e-4), so it is not
the A30. There $\|R_{\rm lin}\|_h \approx 6.3\times10^{-13}$: the system is
solved to the rounding floor, and a residual that small is rounding
noise, whose leading digits no two machines agree on. P <= 1,200 pass.
The earlier local validation ran B3 only at the smoke size, which is why
this wasn't caught before the cluster run.

Decision (user): both. (a) The report states the spec-rule result as a
failure at P = 1,875, with the explanation above. (b) An amended rule is
recorded alongside it and decides whether the job stops: the six-figure
test is waived when **both** runs' final residuals are at or below
Algorithm 1's round-off floor, $\|R_{\rm lin}\|_h \le
\kappa(A)\,\varepsilon_{\rm mach}\,\|f\|_h$ (the paper's own floor, already
logged per iteration as `roundoff_ratio` vs `kappa_eps`); $\beta$ must
still agree to 1e-8 at every size, and a missing $\kappa$ never waives
it. Values (laptop GPU; the floor is the CPU run's):

| P | $\|R_{\rm lin}\|_h$ | floor $\kappa\varepsilon\|f\|_h$ | spec rule | amended |
|---|---|---|---|---|
| 75 | 2.5 | 1.8e-13 | pass | pass |
| 300 | 1.3e-1 | 1.9e-12 | pass | pass |
| 675 | 6.2e-5 | 1.4e-11 | pass | pass |
| 1,200 | 4.7e-8 | 1.0e-10 | pass | pass (spec rule decides) |
| 1,875 | 6.3e-13 | 9.2e-10 | **fail** | pass (waived: at floor) |

`gpu_cpu_equivalence.csv` gains `rlin_floor_cpu`, `rlin_floor_gpu`,
`rlin_at_floor`, `rlin_ok_amended`, `equivalent_amended`; `equivalent`
stays the spec's verdict. `component_b.py --gpu-equivalence` prints both
and exits non-zero only if the amended rule fails. Revisit: if the spec
is amended differently, change `verify_gpu_cpu_equivalence` only.

---

## 2026-09-23 -- `reference/` and `code/` folders (Sections 2 and 6)

`component_b.py --save-reference` writes every test grid, and the
reference field where one exists, to `<package>/reference/` (Section 2:
"Save the grids and reference fields under reference/"): Kovasznay
301 x 401 with exact u, v, p; the Bratu, Burgers and BL residual grids
(no closed-form reference); elasticity 200 x 200 with exact u_x, u_y;
Beltrami 21^3 x 11 with exact u, v, w, p; the four Darcy fields' FVM
pressure at the cell centres; a README describing each (~4.5 MB). The
grids come from the same constants the error code uses; a test recomputes
a real run's logged error from the saved field to 1e-12.
`--save-code` writes Section 6's `code/`: `PROVENANCE.json` (commit,
branch, diff -- from git locally, from the bundle's record on the
cluster), `uncommitted.diff` if any, and the source directories plus
`DECISIONS.md`. Both run in `40_finalize.slurm`.

---

## 2026-09-23 -- BL-gravity: known pathologies (kept as is; for the report)

Decision: the `cos_fourier` basis, retargeted losses and K_max = 20 stay
as they are. What the runs will show, and why:

1. **LiL-Q at P=64 does not reach its target (0.24) within K_max = 20.**
   From the zero initial coefficients the loss is 0.951 at iteration 20;
   it reaches 0.24 at iteration 43 and levels off at 0.2283. It is not a
   code error: the system matrix matches a finite-difference Jacobian of
   the residual to 1.7e-10 (a true Gauss-Newton step). The quasilinear
   iteration is Gauss-Newton on the least-squares loss, which is fast only
   when the achievable residual is small; at P=64 the basis cannot go
   below ~0.23, and from zero the iterates wander (loss 0.43-0.72 for
   dozens of iterations). A monotone backtracking step on the same
   direction stalls at 0.47 -- the step barely reduces the loss there.
   Started from LiL-N's solution (loss 0.240), LiL-Q reaches 0.2283 in
   one iteration, so near the solution it behaves as elsewhere in the
   paper. LiL-N (L-BFGS with line search) reaches 0.24 in ~80 iterations
   from the same start.
2. **The starting point matters, with no uniformly better choice.**
   Starting LiL-Q from the least-squares fit of the IC profile (the
   viscous case's start) reaches the targets in 14 / 24 / 21 / 11
   iterations at P = 64 / 256 / 576 / 1024, against 43 / 12 / 5 / 5 from
   zero -- better only at P=64, as Q9 found for the `fourier` basis.
3. **At P=64, `fourier` is the better basis for LiL-Q** (6 iterations,
   plateau 0.197 vs `cos_fourier`'s 43 and 0.228). `cos_fourier` wins at
   P >= 256 (plateaus 0.071 / 0.043 / 0.010 vs 0.123 / 0.072 / 0.034) and
   makes LiL-N converge at every size (with `fourier` it stalls at
   P = 576 and 1024 under the paper's own targets).
4. **The test-grid residual (`eps_u`, PDE-only mean square on a uniform
   201 x 201 grid) does not track the training loss here**: at P=1024
   LiL-Q scores 3.6 (`cos_fourier`) and 55 (`fourier`) against LiL-N's
   0.33, although LiL-Q's training loss is lower. Unresolved: candidate
   causes are residual growth between collocation points near the steep
   front, the grid's t=0 line (IC slope ~25), and the metric excluding
   the IC/BC terms the training loss weights. Do not use this column for
   BL-gravity comparisons until it is understood.

Full comparison, 2026-09-23 (this machine; iterations / seconds / final
loss; * = target not reached; budgets from `experiments/run_bl.py`):

| Targets | Method, basis | P=64 | P=256 | P=576 | P=1024 |
|---|---|---|---|---|---|
| paper | LiL-N `fourier` | 94 / 3.5 / 0.247 | 5,748 / 121 / 0.150 | 15,000* / 338 / 0.101 | 20,000* / 477 / 0.061 |
| paper | LiL-N `cos_fourier` | 77 / 1.2 / 0.249 | 125 / 2.2 / 0.150 | 401 / 9.2 / 0.075 | 869 / 21 / 0.035 |
| paper | LiL-Q `fourier` | 6 / 0.0 / 0.219 | 7 / 0.4 / 0.139 | 10 / 2.3 / 0.073 | 8 / 8.3 / 0.035 |
| paper | LiL-Q `cos_fourier` | 20* / 0.1 / 0.951 | 5 / 0.3 / 0.125 | 4 / 0.9 / 0.074 | 5 / 5.0 / 0.010 |
| current | LiL-N `fourier` | 96 / 1.6 / 0.234 | 10,000* / 234 / 0.146 | 15,000* / 361 / 0.101 | 20,000* / 467 / 0.061 |
| current | LiL-N `cos_fourier` | 81 / 1.2 / 0.240 | 859 / 19 / 0.075 | 1,223 / 29 / 0.045 | 20,000* / 448 / 0.013 |
| current | LiL-Q `fourier` | 6 / 0.0 / 0.219 | 20* / 0.9 / 0.123 | 20* / 4.9 / 0.072 | 20* / 19 / 0.034 |
| current | LiL-Q `cos_fourier` | 20* / 0.1 / 0.951 | 12 / 0.6 / 0.074 | 5 / 1.2 / 0.043 | 5 / 5.1 / 0.010 |

Paper targets 0.25 / 0.15 / 0.075 / 0.035; current 0.24 / 0.075 / 0.045 / 0.011.

---

## 2026-09-23 -- SLURM job scripts for Component B (`scripts/hprc/`)

Seven jobs chained by `submit_all.sh`: a pre-flight (test suite + smoke
run of everything on the cluster's own library versions; everything else
waits for it to pass), the Section 3.3 CPU runs, the Kovasznay GPU runs
plus check B3, the four-method sweep as two arrays of four (GPU pass per
benchmark; CPU pass at the largest sizes on a CPU node rather than idling
a GPU), Sections 3.6/3.7, and a finalize job (merge the four-method CSVs,
check B1, Section 3.5 figures). Every job: one task, `--cpus-per-task`
cores and exactly that many threads (24 for timed CPU work = the paper's
count), A100s for float64 GPU work (T4 only for the pre-flight),
resumable after a walltime kill. `--exclusive` is left commented out in
the timed CPU jobs (Section 2's "nothing else runs" vs. being charged for
a whole node) -- the user's call; `scripts/hprc/README.md` explains it.

Supporting changes: `four_method_tables.py --merge-from` combines the
per-job CSVs (a run appearing in two inputs is an error; a job directory
without a CSV is reported and skipped); `run_beltrami_pinned.py
--out-dir`; the upload bundle now writes text files with LF endings (a
Windows checkout has CRLF, which bash on the cluster rejects -- 360 of the
tracked files had it) and marks `.sh`/`.slurm` executable.

---

## 2026-09-23 -- Residual-band figures from the Section 3.3 logs

Section 3.5 asks for the figures "from the LiL-Q logs of 3.3";
`residual_band_figures.py` reran the solves itself.
`--from-logs <package root>` now reads the driver's `iterations.csv`
files instead (`--pass paper|kmax`; the K_max pass shows the plateau)
and copies each panel's CSV next to the figure. Panels plot
$\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$ and $\|\mathbf{R}^{(k+1)}\|_h$ at
integer $k$. Checked on the validation pass: every problem shows the band
closing within a few iterations and $\chi_k$ falling to round-off.

---

## 2026-09-23 -- Check B1 (`reproduction_check.csv`)

`experiments/paper_values.py` transcribes every LiL-Q entry of the
manuscript's tables from `Post-JCP/main.pdf` (Tables 2, 3, 5, 6, 7, 9, 11,
14, 15 in that document's numbering -- one ahead of the package's for
some tables), each tagged with its comparison rule: iterations and ranks
equal, errors and residuals within a factor of 2, condition numbers
within 10, timings reported without tolerance (hardware and load differ).
`component_b.py --reproduction-check` compares them with the CPU
paper-pass reruns and writes `reproduction_check.csv`, discrepancies
first. Two transcription notes: Table 5's "8 x 10^-2" target at P=64 is
the code's 8.5e-2 rounded (the pre-v2 code and the reference runs both
used 8.5e-2 -- unchanged); Table 14's residuals are the code's
physical-unit `mse_*_phys` (SPE10 matches to every digit).

On a full validation pass (all but Beltrami, this machine, not a record
run) the violations were: Bratu P=225 4 iterations vs 3 (Q2); BL P=64
10 vs 11 (Q1's off-by-one); Kovasznay P=300 9 vs 10 and its pressure
errors 2.6-7x *better* than Table 9 (see the Kovasznay entry); Darcy S3
Darcy-x MSE 3.0e-3 vs 1.97e-1 (66x; S1, S2, SPE10 match); BL P=1024
numerical rank 936 vs 934; and BL-gravity throughout, by design.

---

## 2026-09-23 -- Check B3 and the rest of Section 3.2

`experiments/component_b.py --gpu-equivalence` runs check B3 at every
Kovasznay size and writes `gpu_cpu_equivalence.csv`: the GPU run against
the CPU `gelsy` run (`||beta_GPU - beta_CPU|| / ||beta_CPU|| <= 1e-8`,
`||R_lin||_h` to six significant figures), plus the solve time of CPU
`gelsy` (the paper's driver), CPU `gels` (the same algorithm as the GPU
QR -- `_lstsq_cpu_gels` existed but was never called) and the GPU QR, all
on the same final-iterate system, median of 3 after a warm-up. It exits
non-zero if any size fails ("if it fails, stop and report"). Also:
- The Kovasznay GPU path now synchronizes before both clock reads around
  the solve (Section 2); previously only the result copy synchronized
  implicitly. The GPU solve time includes the host-to-device copy of A,
  which is assembled on the CPU.
- `run.json` gains `gpu_qr`: the $3\times8NP$ memory estimate (previously
  only printed), the smallest $\min|R_{pp}|/\max|R_{pp}|$ seen, and the
  iterations flagged below $10^{-13}$ (previously printed, not recorded).
- `solve_kovasznay(..., return_final_system=True)` returns the last
  weighted system for the same-system timing.

---

## 2026-09-23 -- Component B driver (`experiments/component_b.py`)

The Section 3.3 logged reruns had instrumentation but no script that ran
them. `experiments/component_b.py` is that script: 62 runs --
Bratu (3 sizes), Burgers (5), BL viscous and gravity (4 each), each in a
`paper` pass (the paper's stopping rule) and a `kmax` pass (stopping rule
disabled: a zero loss target / zero coefficient-change tolerance can't be
met, so the loop runs to K_max); Kovasznay (5 sizes x CPU/GPU x both
passes); elasticity (5 sizes), Beltrami ($P=7{,}984$), Darcy (S1, S2, S3,
SPE10), `paper` pass only, as Section 3.3 asks. Each writes
`B_instrumentation/<benchmark>_<config>_<device>_<pass>/` with `run.json`,
`iterations.csv`, `summary.json` (the problem's own final metrics plus
the log's), and that run's `hardware.json`/`environment.txt`;
`runs_index.csv` summarizes them. Settings come from each problem's
`experiments/run_*.py` constants (for BL gravity, the `cos_fourier` basis
and retargeted losses -- a documented deviation, kept by decision).

Same operational behavior as the four-method sweep: a folder with
`summary.json` is complete and skipped on rerun (resume after a walltime
kill); an exception writes `error.txt` with the traceback and the run is
retried next time; one untimed warm-up per device; runs are sequential;
`--benchmarks/--passes/--devices/--configs` split the plan into jobs,
`--list` prints it, `--smoke` runs the smallest size of everything in
seconds.

---

## 2026-09-23 -- Per-iterate test errors in `iterations.csv`; Beltrami's final-error memory spike removed

Section 3.1 item 10 (test errors of every iterate, "for the log only,
never for stopping") was the last column group left empty. Row `k` now
carries the errors of $\boldsymbol\beta^{(k)}$ and the terminal row those of
the returned coefficients, on these test grids (never used for
collocation):

| Problem | Grid | Columns |
|---|---|---|
| Kovasznay | 301 x 401 uniform (Section 2) | `eps_u/v/p`, `eps_p_meanfree`, `maxerr_u/v/p`; `eps_p` in the paper's corner-pin gauge, `eps_p_meanfree` after subtracting the grid mean from both fields (Section 2) |
| Bratu | 201 x 201 uniform (Section 2) | `eps_u` = mean square of $\Delta u + \lambda e^u$ (no closed form) |
| Beltrami | 21^3 x 11, the paper's own (Section 2: "as in the paper") | `eps_u/v/p`, `maxerr_*`; pressure shifted to the exact mean per time level, as `compute_errors` does. No `eps_w` column exists; w's error equals u's by symmetry |
| Burgers, BL | 201 x 201 uniform over space x [0,T] -- **our choice**, Section 2 names none | `eps_u` = mean square of the PDE residual (residual-MSE benchmarks, per the spec's note). At a zero start the Burgers row 0 is exactly 0: u=0 satisfies the interior PDE and misses only the IC |
| Elasticity | 200 x 200, the grid behind Table 7 -- **our choice**, Section 2 names none | `eps_u` ($u_x$), `eps_v` ($u_y$), `maxerr_*` |
| Darcy | the 60 x 220 cell centres against the FVM reference | `eps_p`, `maxerr_p` (psi) |

Evaluation goes through the tensor-product structure
(`lilq.test_errors.tensor_grid_values`: $u=\Phi_x\Theta\Phi_y^\top$ and its
N-D analogue) instead of full basis matrices -- a few milliseconds per
iterate even at Kovasznay $P=1{,}875$ and Beltrami $P=7{,}984$, outside the
timed assembly/solve phases. Tests check it against full evaluation for
every basis type and derivative order in use, and check every problem's
terminal-row errors against that problem's own final-error code
(identical to 1e-10).

**Beltrami's final errors now use the same evaluator.** `compute_errors`
built full basis matrices on its 101,871-point grid (4,096 pressure
columns): a 3.8 GB peak and 3.7 s on top of the solve's own memory, just
to report errors. Now 13 MB and 0.02 s, with identical values (0.0
relative difference; snapshots 2e-16). This lowers the memory a
cluster job for Beltrami needs.

---

## 2026-09-23 -- Elasticity gets Section 3.1 logging

Section 3.3 asks for logged elasticity runs at all five sizes; elasticity
was the one problem with no `iteration_logger`/`run_json_path`.
`solve_elasticity` now takes both and logs the same two-row layout as
Darcy (the other linear problem): `k=0` for the single solve and the
terminal `k=1`. Its `lstsq` call now passes `cond=EPS_MACH` explicitly
like every other solver -- scipy's own default for `gelsy`, confirmed
bit-identical against the committed solver in both BC modes (a test does
this against git history; it skips on the cluster copy, which has no
`.git`, as do the two provenance tests that need git).

---

## 2026-09-23 -- `iterations.csv` follows the spec's k numbering; check B2 recorded in real runs

**Numbering.** Rows were labeled by solve count (`k = 1..K`), which put
each row one index off the spec's row k: row "k" held $\|\mathbf{R}_{\mathrm{lin}}^{(k-1)}\|_h$,
$\chi_{k-1}$, $o_{k-1}$, $\kappa(\mathbf{A}^{(k-1)})$, $\delta\boldsymbol\beta^{(k-1)}$
next to $\|\mathbf{R}^{(k)}\|_h$ (the post-solve residual). Now row `k`
(from 0) is outer iteration $k$ exactly as the spec defines it: the system
assembled at $\boldsymbol\beta^{(k)}$ and solved for $\boldsymbol\beta^{(k+1)}$,
with `norm_R_h` $=\|\mathbf{R}^{(k)}\|_h$ computed as the spec's one
mat-vec $\mathbf{A}^{(k)}\boldsymbol\beta^{(k)}-\mathbf{f}^{(k)}$ (and its
interior part from the same vector), alongside
$\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$, $\chi_k$, $o_k$, the stall flag,
$\kappa(\mathbf{A}^{(k)})$, $\delta\boldsymbol\beta^{(k)}$. A terminal row
`k = K` carries $\|\mathbf{R}^{(K)}\|_h$ at the returned coefficients
(nothing is assembled there, so its other columns are empty) -- a run of
$K$ solves has $K+1$ rows. `first_stall_iteration` is therefore in the
spec's indexing. Helpers `solve_rows`/`last_solve_row` (in
`lilq.iteration_log`) separate the solve rows from the terminal row;
the Section 3.5 figures plot $\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$ and
$\|\mathbf{R}^{(k+1)}\|_h$ (the next row's `norm_R_h`) at $k$. The
Section 3.6 CSV (`first_stall_iteration`) and the Section 3.7 run use this
layout once rerun; their existing outputs predate it.

**B2 in real runs.** Until now B2 was checked only in unit tests, and as
direct-evaluation vs direct-evaluation at the final iterate. The tracker
now compares, on every iteration, the mat-vec $\mathbf{R}^{(k)}$ with the
nonlinear operator evaluated directly at $\boldsymbol\beta^{(k)}$ (the
spec's identity, which only holds if the quasilinearization is right) and
`run.json` records `b2_check = {k, rel_err, max_rel_err_over_run}` with
`rel_err` at `k = 1`. Not the run maximum: near convergence
$\mathbf{R}^{(k)}$ is a small difference of O(1) quantities and
cancellation dominates the relative difference -- measured on Kovasznay
$P=1{,}875$: 4.0e-15 at $k=1$, 1.1e-2 at the converged iterate. The
maximum stays in the record for transparency. Verified across all six
nonlinear problems before this change (2.4e-14 or better at every
iterate before convergence; the scratch check is described in the
pre-HPRC audit).

**Also:** `run.json` gains `kappa_qr_raw_ratio`, the $|R_{11}|/|R_{PP}|$
over all pivoted-QR diagonal entries that Section 3.1 item 8 asks for
beside the retained-part ratio (previously computed and discarded).

---

## 2026-09-23 -- Four-method sweep: failures logged, incremental writes, resume, warm-up

`experiments/four_method_tables.py` had no exception handling (one
failing run ended a multi-hour sweep, and Section 3.4's `failure`
stopping reason could never be recorded) and wrote its CSV only at the
end (a walltime kill lost every row). Now: each run is wrapped, and an
exception becomes a `failure` row carrying the traceback (new `error`
column; Section 9: "log it with the traceback"); the CSV is rewritten
atomically after every row; rerunning with the same `--out-dir` resumes,
skipping completed rows and retrying failures (`--fresh` to start over);
one untimed warm-up per device precedes the timed runs (Section 2);
`--P`/`--methods`/`--seeds`/`--passes` select a subset so the sweep can
be split across concurrent jobs, each with its own `--out-dir`;
`hardware.json`/`environment.txt` are written into the output folder.
New columns: `collocation_seed` (always the config's 42; `seed` is the
network-init seed) and `wall_total_s` (setup + pretraining + training;
`training_time_s` remains the optimizer loop alone).

---

## 2026-09-23 -- NiL multi-seed runs no longer change the collocation set

`config.seed` set both the NiL network initialization (`set_seed`) and
the random collocation abscissae (`generate_collocation_points_2d(...,
seed=config.seed)`). The multi-seed harness (`lilq.multiseed`) and
`experiments/four_method_tables.py` replaced `config.seed` with 0/1/2,
so NiL-N/NiL-Q seeds 0/1/2 were trained on three different collocation
grids, none of them the seed-42 grid the spec keeps fixed and the LiL
methods use (confirmed: Bratu's interior points move by up to 0.35
between seeds 0 and 42). Their final losses were therefore measured on a
different point set than the other methods' against the same target.

Fix: `init_seed` on Bratu/Burgers/BL configs (default `None` = `seed`,
so a default single run is unchanged); `run_nil_n`/`run_nil_q` seed from
`lilq.utils.nn_init_seed(config)`; the multi-seed paths vary `init_seed`
only. The LiL runners are untouched (deterministic; `seed` only fixes
their collocation). Nothing in `results/` from before this change used
multi-seed runs, so no stored result is affected.

---

## 2026-09-23 -- Cluster readiness: thread count bounded by the allocation; provenance survives without `.git`

**Thread count.** `lilq/blas_threads.py` defaulted to
`SLURM_CPUS_PER_TASK`, else `os.cpu_count()`. Under an `--ntasks=N`
request (the form in `HPRC_FASTER_Workflow_Guide.md`) `SLURM_CPUS_PER_TASK`
is unset, and `os.cpu_count()` reports the whole node (64 on FASTER), so
BLAS and PyTorch would each start 64 threads on an N-core allocation.
Now: `SLURM_CPUS_PER_TASK`, else `SLURM_CPUS_ON_NODE` (the job's own
cores on the node), else the process's CPU affinity
(`os.sched_getaffinity`, which honors the scheduler's cpuset); only a
machine with no affinity API falls back to `os.cpu_count()`. On the
workstation the result is unchanged (all 24 logical cores).

**Provenance.** The cluster copy is uploaded without `.git`, so
`capture_git_info` would have recorded no commit. New
`scripts/make_hprc_bundle.py` packs the git-tracked files (minus
`reference_results/`, plus `tests/`) under a single `lilq-pinn/` folder
and writes `PROVENANCE.json` (commit, branch, uncommitted diff, time,
host); `capture_git_info` falls back to it, marked `"source": "bundle"`.
`hardware.json` also gains `scheduler` (SLURM job id, partition, node,
CPUs/GPUs/memory actually allocated) and `cpu.usable_cores`/`hostname`,
since a cluster node's hardware alone doesn't say what the job had.

---

## 2026-09-23 -- Kovasznay: diagnostic SVD gated; current configuration confirmed as the paper's

**Found during the pre-HPRC audit.** `solve_kovasznay` called
`np.linalg.cond(A_sys)` (a full SVD) on every outer iteration, recording
it in `history['cond_number']`, which nothing reads. The pre-GitHub code
gated the same call off by default; the v2 consolidation made it
unconditional, and the Phase 1 slowdown fix (`319987f`) only gated
Beltrami's copy. Now opt-in via `analyze_conditioning=False` (same
pattern as Beltrami). Same machine, warm-up excluded, best of 3:

| P | with SVD (s) | without (s) | manuscript Table 9 (s) |
|---|---|---|---|
| 75 | 0.09 | 0.06 | 0.06 |
| 300 | 0.63 | 0.21 | 0.22 |
| 675 | 1.72 | 0.68 | 0.67 |
| 1,200 | 6.52 | 1.92 | 1.87 |
| 1,875 | 17.07 | 5.49 | 5.58 |

**This, not thread contention, explains the reference run's slower
Kovasznay timings** (0.07/0.52/2.0/4.5/12.8 s). The Q5 answer in
`reponse_Package1_v2.md` attributes them to unpinned BLAS threads and says
there was no Kovasznay code difference between versions; both need
correcting.

**Configuration.** The manuscript text (Section 6.6) says $n_d =
\lceil\sqrt{2P}\rceil$, $\lceil P/6\rceil$ points per edge (14,037 rows at
$P=1{,}875$) and a $10^{-12}$ coefficient-change tolerance. That is
`k_ratio=10`, ratios (0.6, 0.2, 0.2). `experiments/run_kovasznay.py` uses
`k_ratio=4` with the same ratios (5,564 rows at $P=1{,}875$) and
tolerance $10^{-9}$. All three candidate configurations were run at all
five sizes:

| Config | rows @1,875 | iterations | time @1,875 | E_u / E_v / E_p @1,875 | E_p @675 |
|---|---|---|---|---|---|
| Manuscript Table 9 | ? | 16/10/6/6/6 | 5.58 s | 7.2e-13 / 5.3e-12 / 1.3e-11 | 8.1e-4 |
| Current script (k=4, .6/.2/.2, 1e-9) | 5,564 | 16/9/6/6/6 | 5.5 s | 7.0e-13 / 5.3e-12 / 1.8e-12 | 1.1e-4 |
| Pre-v2 script (k=4, .8/.1/.1, 1e-9) | 6,580 | 16/10/6/6/6 | 7.5 s | 3.9e-13 / 3.5e-12 / 1.1e-11 | 4.4e-4 |
| Manuscript text (k=10, .6/.2/.2, 1e-12) | 14,037 | 20/11/7/6/6 | 18.2 s | 1.2e-13 / 1.1e-12 / 7.3e-12 | 3.3e-4 |

The current script reproduces Table 9's timings, its velocity errors at
$P \ge 1{,}200$ to two digits, and every iteration count except one
(9 vs 10 at $P=300$); its pressure errors are 2.6-7x *better* than the
table's. The manuscript-text configuration is 3x slower, needs more
iterations than the table reports, and is not uniformly more accurate
(pressure worse than the current script). The tolerance check agrees:
the current configuration with $10^{-12}$ gives 21/11/7/6/6 iterations,
not the table's 16/10/6/6/6. **Decision: keep the current configuration.**
The manuscript's Section 6.6 description (row counts, 14,037, and the
$10^{-12}$ tolerance) does not describe the runs behind Table 9 and should
be corrected, as should the Q5 answer that confirms it. Report the
Table 9 pressure-error difference as an improvement, not a reproduction
failure.

---

## 2026-09-23 -- Section 3.7 (Beltrami pressure pinned at every temporal level) completed

**Same "pulled forward from HPRC, run locally" batch as Section 3.6 above.**
Section 6.9.1 / Section 3.7 of the spec: the Beltrami linear system has
an exact null space (the pressure temporal modes $T_j(t)\cdot 1$, $j=0..N_p-1$,
enter the momentum/continuity residuals only through their *gradient* in
x/y/z -- which is identically zero for a spatially-constant mode -- so
they're constrained only by the single pin row at $t=t_{\mathrm{domain}[0]}$,
leaving $N_p-1$ free directions). Spec asks for a rerun with the pin
generalized to one pin row per temporal collocation level ($N_p=8$ rows
instead of 1), expecting: full column rank, a meaningful $\kappa$ from
the pivoted QR, and a smaller $t=1$ pressure error than the paper's
published 0.752%.

**Implementation** (`problems/beltrami.py`): added
`BeltramiConfig.n_pressure_pin_levels` (default `1`, bit-identical to
the pre-3.7 single pin -- verified both by a direct pre/post-change
bit-identical rerun and by every existing Beltrami test still passing
unmodified). The `n_pin` pin times are Chebyshev-Gauss-Lobatto nodes in
`t_domain` (`_cgl_temporal_pin_nodes`) -- the natural collocation set for
an $N_p$-term (degree $0..N_p-1$) Chebyshev temporal expansion, matching
the null-space argument exactly (`n_pin == N_p` removes exactly enough
directions to leave none free). The pin block's weight generalizes the
existing one-scalar-per-block convention: `sqrt(lambda_bc / n_pin)`
(reduces to the existing `sqrt(lambda_bc)` at `n_pin=1`). Both the
scalar-loss and residual-vector Section 3.1 instrumentation closures
were generalized the same way (mean-of-squares / weighted-vector over
`n_pin` rows instead of the single hardcoded row).

**Verified the null-space claim empirically before the real run**, at a
small config (`N_vel=N_p=3`, so 2 free null-space directions expected):
measured SVD rank directly at `n_pin=1,2,3` and got `P_total-2`,
`P_total-1`, `P_total` respectively -- an exact match to the "$n$ pin
rows close $n$ null-space dimensions, up to $N_p-1$" claim, not just a
plausible-sounding result.

**Real run** (`experiments/run_beltrami_pinned.py`, paper config
$N_{\mathrm{vel}}=6$, $N_p=8$, `chebyshev` basis, $P_{\mathrm{total}}=7984$,
matching the manuscript exactly): converged in 4 iterations. Wall-clock
**363 s** (about 72 s per QR solve), from a later rerun with nothing else
running on the machine; the 531 s first recorded here was invalid -- that
run shared the CPU with the Section 3.6 study and the test suite. Every
other number below is identical between the two runs. (The spec estimated
"about ten minutes".)

| | Result |
|---|---|
| Full column rank | **True** -- `num_rank_gelsy=7984/7984` every iteration (the SVD-based rank isn't computed at this $P$ -- above `DEFAULT_SVD_CONDITIONING_THRESHOLD`, uses pivoted QR instead, by design) |
| $\kappa$ (pivoted QR, final iterate) | 18166.1 -- finite, real, not NaN |
| $t=1$ pressure error | 0.7515% vs. paper's published 0.752% |

**Honest characterization of the last row**: technically confirms the
spec's expectation ("a smaller pressure error at $t=1$") but the margin
is razor-thin -- 0.0005 percentage points, about 0.07% relative, not a
dramatic drop. Resolving the null space to full rank clearly delivers
the two structural guarantees (full rank, meaningful $\kappa$) cleanly,
but apparently wasn't the dominant source of the paper's 0.752% error to
begin with -- most of that error is presumably ordinary Chebyshev
approximation error at $N_p=8$, not null-space ambiguity. Reporting this
as-is rather than as a bigger win than it is.

Five-snapshot table ($t=0,0.25,0.5,0.75,1.0$; $u=v=w$ errors identical
by the problem's own symmetry):

| t | u=v=w | p |
|---|---|---|
| 0.00 | 1.356e-4 | 1.463e-3 |
| 0.25 | 4.141e-4 | 2.822e-3 |
| 0.50 | 3.737e-4 | 3.153e-3 |
| 0.75 | 3.819e-4 | 4.511e-3 |
| 1.00 | 3.430e-4 | 7.515e-3 |

Output: `results/beltrami_pinned/{run.json,iterations.csv,report.json,hardware.json,environment.txt}`.

8 new tests added to `tests/test_beltrami_instrumentation.py` (CGL node
generation incl. the `n_pin=1` backward-compat case, the null-space
rank-closure claim at small scale, pin-row weight formula via a real
`run.json` readback, `N_composition`/Check-B2 residual identity at
`n_pin>1`). Full suite (218 tests, including these and the Section 3.6
tests below): all passing.

---

## 2026-09-23 -- Section 3.6 (Table 3 Burgers basis study) completed

**Not the full Package 1 spec rerun -- these two items (3.6 and 3.7) were
pulled forward and run locally rather than deferred to the HPRC pass,
since both are short enough to run outside that budget** (Section 3.6:
one ~625-coefficient study per basis, 9 bases; Section 3.7: one ~10-minute
Beltrami solve, per the spec's own note that it "is included here rather
than in Package 2 for that reason").

`experiments/run_burgers_basis_comparison.py`'s existing 8-basis study
(pre-existing script, independent of the other 6 problems' shared
`solve_lil_q`/instrumentation wiring) was missing: (a) the "Sin x
{Cos,Sin}" basis row from Table 2/3 -- oddly absent even though it's
`problems/burgers.py`'s actual production `DEFAULT_BASIS`, (b) a protocol
run with the stopping rule disabled to `K_max=50` so each row reports a
stagnation *level* rather than a stopping value, (c) the Section 3.1
instrumentation columns (`kappa`, numerical rank, first stall iteration)
the spec's Table 3 asks for.

**Added:**
- `'sin_fourier'` basis key (`sin(1..N_x)(x) x cos+sin(N_t)(t)`) to
  `BASIS_CONFIGS`/`create_comparison_basis`.
- `ComparisonConfig.disable_stopping_rule` (default `False`, unchanged
  behavior) -- when `True`, both the `R_tol` check and the
  stagnation-window early break are skipped inside
  `solve_lilq_burgers_comparison`'s loop, so every basis always runs the
  full `max_quasi_iters` budget.
- `LilQDiagnosticsTracker`/`IterationLogger` wiring inside
  `solve_lilq_burgers_comparison` (an `iteration_logger` parameter,
  optional, `None` by default -- omitted, behavior is unchanged), via an
  independently-coded `residual_vector(b)` closure (not shared code with
  the existing scalar loss tracking, so the cross-check is real) and the
  same `cond=EPS_MACH` explicit LAPACK `gelsy` pattern used everywhere
  else in this codebase, now also capturing `rank_gelsy` here.
- `run_table3_study()` (defaults: `N_x=N_t=25`, `disable_stopping_rule=True`,
  `max_quasi_iters=50`, all 9 basis keys -- the study's own grid, matching
  the spec's `73x73` interior / 313 IC / 313 BC exactly, confirmed
  directly, no grid change needed) and `write_table3_csv()`, plus a new
  `--table3` CLI flag writing `table3_basis_study.csv` to the script's
  existing `results/burgers_basis_comparison/` output directory (its
  `save_provenance` call included, same as the script's default path).

**Verified before running at full scale:** a small-scale smoke test
(`N_x=N_t=6`, `max_quasi_iters=10`) confirmed real, sensible
per-iteration `chi`/`stall_flag`/`order_obs`/`kappa` values (not NaN or
placeholder), and that `stall_flag` correctly flips to `True` once `chi`
drops below its threshold -- confirming the new `residual_vector_fn`
wiring is correctly connected end-to-end, not just executing without
crashing.

**Full-scale run** (`N_x=N_t=25`, `K_max=50`, all 9 bases, the study's own
grid: 5329 interior collocation points at this density, 313 IC, 313 BC --
`results/burgers_basis_comparison/table3_basis_study.csv`):

| basis | final $\|R\|_h^2$ | $\kappa$ | rank (svd / gelsy) |
|---|---|---|---|
| Cheb x Cheb | 7.82e-05 | 2.5e3 | 625 / 625 |
| Sin x Cheb | 4.91e-09 | 3.4e2 | 625 / 625 |
| Sin x Sin | 4.98 | 5.5e1 | 625 / 625 |
| Cos x Cheb | 1.71e-01 | 3.4e2 | 625 / 625 |
| AugSin x Cheb | 3.14e-08 | 5.2e2 | 625 / 625 |
| Fourier x Cheb | 1.27e-05 | 4.3e9 | 625 / 625 |
| Fourier x Fourier | 1.27e-05 | 1.3e16 | 610 / 624 |
| ELM | 4.95e-02 | 1.3e18 | 43 / 70 |
| **Sin x {Cos,Sin} (new)** | 4.91e-09 | 9.6e8 | 625 / 625 |

All consistent with what's already understood about these bases from
earlier in this engagement: Sin x Sin fails outright (4.98 -- a sine
temporal factor vanishes at $t=0$, structurally unable to represent the
nonzero IC, same degeneracy documented for `cheb_sin`/`sin_cheb`-family
bases elsewhere); Cos x Cheb is poor (cos(0)=1 can't satisfy the
homogeneous Dirichlet BC exactly); ELM is both inaccurate and severely
rank-deficient (43/625 -- most of its 625 random features are
numerically redundant at this collocation density) with the worst
conditioning by 14 orders of magnitude. The new Sin x {Cos,Sin} row
tracks Sin x Cheb almost exactly (4.9085e-09 vs 4.9085e-09) -- both pair
a sin-in-x factor (satisfies the BC) with a *complete* temporal basis
(full Fourier vs. Chebyshev), so both can represent the IC; matches the
production default's (`problems/burgers.py`) good behavior elsewhere in
this codebase.

Noted but not investigated further (out of scope for this item): SVD
rank and gelsy rank disagree at the two worst-conditioned bases
(Fourier x Fourier: 610 vs 624; ELM: 43 vs 70) -- expected when singular
values decay gradually near the rank cutoff, since the two methods
threshold differently (SVD's direct singular-value cutoff vs. gelsy's
pivoted-QR diagonal-decay heuristic); not a bug, just a reminder that
"numerical rank" is method-dependent exactly where it matters least
(these bases are unusable regardless of which rank number is trusted).

9 new tests added to `tests/test_burgers_basis_comparison.py` covering:
the new basis key's structure, `disable_stopping_rule` actually
overriding both the `R_tol` and stagnation checks (not just one),
`run_table3_study`'s config-validation guard, its default 9-basis
coverage, and `write_table3_csv`'s CSV roundtrip. Full suite (218 tests,
including these and the Section 3.7 tests below): all passing.

---

## 2026-09-23 -- BL-gravity `GRAVITY_TARGET_LOSSES` retargeted to LiL-Q's actual floor

**Follow-up to the `cos_fourier` basis entry immediately below, same
investigation.** Prompted by a direct question: since LiL-Q was
consistently reaching the shared target early and stopping (`R_tol` is a
stopping rule, not a ceiling), could it actually do meaningfully better
than what the target lets it show?

**Measured LiL-Q's true floor** by rerunning with `R_tol` effectively
disabled (`1e-12`) and `max_quasi_iters_lil=300` (vs. the default 50),
confirming full plateau (loss identical to 6+ decimal places, `update_norm`
at floating-point noise) before trusting the number -- N=8 needed the
extended budget (still visibly descending at iteration 50, settled by
~100); N=16/24/32 were already exactly flat by iteration 50, so the
default 50-iteration budget was already sufficient there:

| N | Old target | LiL-Q's actual floor | New target (~5% margin) |
|---|---|---|---|
| 8 | 0.25 | 0.2283 | 0.24 |
| 16 | 0.15 | 0.0706 | 0.075 |
| 24 | 0.075 | 0.0427 | 0.045 |
| 32 | 0.035 | 0.0103 | 0.011 |

**Verified the retargeted values against all three other methods**
(NiL-N, NiL-Q, LiL-N) before finalizing, using their real, unmodified
iteration/line-search budgets and the `cos_fourier` basis now in place:

- N=8/16/24: all three converge cleanly under the new targets (N=24 in
  particular lands within 0.00002 of 0.045 for every method -- strong
  confirmation the ~5% margin was well-chosen, not just for LiL-Q).
- N=32: **none of the three reach 0.011**, even run to their full
  existing caps -- NiL-N reaches 0.0296 (20,000-iter cap, 1,464s),
  NiL-Q reaches 0.0378 (10,000-iter cap, 637s), LiL-N reaches 0.0129
  (20,000-iter cap, 425s), closest but still short. This isn't a
  budget shortfall fixable by waiting longer within any reasonable
  margin -- LiL-Q's quasilinearization is genuinely more powerful than
  L-BFGS-based optimization at this scale, and a real capability gap
  opens up between the methods at the largest size that doesn't exist
  at the smaller three.

**Decision (asked and confirmed): tighten N=32 to 0.011 as well**,
accepting that NiL-N/NiL-Q/LiL-N will show as non-converged there. The
alternative (leaving N=32 at its old 0.035, initially proposed as the
safer default) was explicitly rejected -- the target should track what
LiL-Q can actually do, not be loosened to keep every method's row
looking converged. N=32 becomes the size where the four-method
comparison most clearly shows LiL-Q's advantage, which is presumably
closer to the point of the comparison than uniform convergence.

`GRAVITY_TARGET_LOSSES` updated in `experiments/run_bl.py`
(`four_method_tables.py`/`residual_band_figures.py`/`validate_pre_hprc.py`
all import it from there, so no other file needed a change). No test
hard-coded the old literal values (checked directly). 203/203 tests
passing, unchanged from the `cos_fourier` entry -- this is a target-value
change with no code-path/schema implications, so no new tests were
needed beyond what already exercises `GRAVITY_TARGET_LOSSES`.

---

## 2026-09-23 -- BL-gravity switched to `cos_fourier` basis (LiL-N convergence)

**Not Phase 1 work -- found during a pre-HPRC validation pass (small
network-size sanity sweep across all 7 problems, requested to catch
bugs/check results/gauge timing before committing to full HPRC runs;
see `experiments/validate_pre_hprc.py`).** That sweep flagged
Buckley-Leverett-gravity's LiL-N as an outlier: 144s at N=16 versus
NiL-N/NiL-Q's 24-39s at the same size, where LiL-N is normally the
*fast* method.

**Investigation.**

1. **`reference_results/bl_gravity_experiments_fourier/` doesn't help as
   a baseline.** Every stored N (8/16/24/32) shows `converged: False`
   with `total_line_searches` exactly `3 * max_iterations` -- the exact
   signature of the `max_line_searches = max_iterations * 3` bug
   documented in this log's "line-search cap actually binds" entry,
   found and fixed before this session. These reference runs were cut
   off by that bug, not by exhausting a correct budget, so they don't
   establish whether LiL-N converges under the current, fixed codebase.

2. **Real reruns at N=16 and N=24 (current codebase, correct budgets)
   show a genuine stall, not noise.** N=16: 5,748 of a 10,000-iteration
   budget, loss trajectory drops fast to ~0.18 by iteration 300 then
   crawls near-linearly for the remaining ~5,400 iterations to just
   barely cross the 0.15 target. N=24: hits the full 15,000-iteration
   cap without converging at all, plateauing at loss≈0.101 against a
   0.075 target -- the same fast-drop-then-flatline shape, just landing
   short instead of barely crossing.

3. **Root cause: basis representation, not optimizer or gradient.** The
   gravity initial condition is a steepness-100 smooth step
   (`1 - 1/(1+exp(-100(x-0.4)))`), essentially monotonic in $x$.
   `BLConfig`'s default `basis_type='fourier'` (`mode_x='both'`) splits
   its $x$-modes evenly between sine and cosine -- confirmed directly
   (`Fourier1D(N, mode='both').evaluate(...)` has exactly $N$ columns,
   same as `mode='cos'` alone, i.e. `'both'` gets *half* as many pure
   cosine modes as a cosine-only basis at the same $N$). Sine modes
   contribute almost nothing to representing a monotonic profile, so
   half the spatial resolution is wasted. Direct least-squares fit of
   the real IC against each basis, independent of the solver entirely:

   | $N$ | `fourier` (current) | `cos_fourier` |
   |---|---|---|
   | 16 | 7.46e-2 rel. L2 err | 5.27e-2 |
   | 24 | 4.29e-2 | 2.31e-2 |
   | 32 | 2.39e-2 | **9.81e-3** |

   (`cos_fourier`/`cos_cos`/`cheb_cos` all score identically here --
   the win is specifically "full cosine resolution in $x$", not
   Chebyshev per se; `cheb_sin`/`sin_cheb`/`cos_sin`/`sin_cos` are
   degenerate for this check, since a `sin`-mode temporal component
   vanishes at $t=0$ and can't represent anything at the IC regardless
   of the spatial basis.) The non-gravity IC (`exp(-10x)`, smooth) fits
   to 1.3e-5 relative error by $N=16$ under the *same* `fourier` basis
   currently in use -- confirming the representation gap is specific to
   gravity's steep IC, not a general problem with the basis choice.

   This creates an irreducible floor in the loss landscape that L-BFGS
   (LiL-N's optimizer) stalls near -- gradients go small and
   uninformative approaching a floor the basis genuinely can't get
   below, producing exactly the observed crawl. LiL-Q isn't exposed the
   same way because each outer iteration solves the linearized
   least-squares system exactly (Bellman-Kalaba), not via
   gradient-descent creeping -- consistent with LiL-Q converging in
   0.42s at N=16 on the *same* basis where LiL-N needed 144s. NiL-N/
   NiL-Q are less exposed because a neural network isn't capacity-limited
   by a fixed truncated Fourier expansion the same way.

4. **`cos_fourier` (full cosine resolution in $x$, unchanged full Fourier
   in $t$) verified end-to-end with real LiL-N reruns at every paper
   size:**

   | $N$ | `fourier` (current) | `cos_fourier` |
   |---|---|---|
   | 8 | 94 iters, 1.5s, converged | 77 iters, 3.9s, converged |
   | 16 | 5,748 iters, 144s, converged (barely) | **125 iters, 3.9s** |
   | 24 | 15,000 iters (capped), 368s, **not converged** (0.101 vs 0.075) | **401 iters, 8.8s**, converged |
   | 32 | not rerun (extrapolated to fail) | **869 iters, 22.5s**, converged (0.0349 vs 0.035) |

**Decision: `experiments/run_bl.py` gains a `GRAVITY_BASIS = 'cos_fourier'`
constant**, used as the default basis for every BL-gravity config this
script builds (CLI `--basis`, left unset by default, now resolves to
`'cos_fourier'` for `--gravity` and `'fourier'` otherwise; an explicit
`--basis` always overrides). `experiments/four_method_tables.py`,
`experiments/residual_band_figures.py`, and
`experiments/validate_pre_hprc.py` all import `GRAVITY_BASIS` from
`run_bl.py` (not duplicated) and use it identically, so every
"real experiment" pathway in this codebase now agrees. Applied to the
*config* (affects all four methods sharing it), not just LiL-N -- a
better representation of the same physics should not depend on which
solver is asked to use it, and the fit-quality numbers above are
solver-independent. `BLConfig.with_gravity()`'s own dataclass default is
**not** touched -- every call site that builds a real experiment config
passes `basis_type` explicitly, so changing the factory's own default
would not have propagated anyway, and leaving it alone keeps ad hoc/
example callers (`examples/run_bl_lilq.py`, `experiments/run_all_dry.py`)
unaffected unless they ask for this basis explicitly.

**This is a methodology decision, not a bug fix** -- `'fourier'` is the
paper's own established default for BL, both variants; this is a
deliberate, documented deviation from that for the gravity case only,
because the evidence shows it removes a stall that otherwise consumes
disproportionate iteration budget (and, at $N \ge 24$, prevents
convergence entirely) for a reason unrelated to the physics or the
gradient (both already verified correct) -- purely a fixed-basis
representation-capacity limit. Non-gravity BL is untouched; it doesn't
show this problem (`exp(-10x)` fits the current basis to near machine
precision already).

**Not done here, worth knowing:** the pre-existing reference results
never converged either (same non-convergent basis, plus the separate
line-search-cap bug on top), so there's no clean "does this match the
published table" comparison available locally -- only the manuscript
itself would say whether BL-gravity LiL-N's published entries are
already asterisked (non-converged) at $N=24$/$32$, which this change may
now turn into converged entries with different (likely lower/better)
final losses than whatever was previously reported.

Regression tests: `tests/test_bl_experiment_runner_config.py` (the
`GRAVITY_BASIS` constant, and `run_bl.py --gravity`'s CLI default
resolution -- verified functionally via a stubbed `main()` run, not
just source inspection, including that an explicit `--basis` still
overrides), plus a `basis_type` assertion added to the existing
viscous-vs-gravity tests in `tests/test_four_method_tables.py` and
`tests/test_residual_band_figures.py`. 198/198 -> 203/203 tests passing.

---

## 2026-09-22 -- Kovasznay GPU solve path (Section 3.2)

**Phase 1, sub-batch 11 -- the last piece of the Component B
instrumentation rollout.** Section 3.2's GPU full-rank solve path,
Kovasznay-only per the spec. This machine has a real CUDA device
(confirmed directly, `torch.cuda.is_available() == True`), so every
piece of this batch was verified against a real GPU run, not skipped or
mocked.

**`problems/kovasznay.py`**: new `config.use_gpu: bool = False` field
(default off, same backward-compatible pattern as Beltrami's). When
True, each outer iteration's solve runs via `_lstsq_gpu_qr` --
`torch.linalg.qr(A, mode='reduced')` + `solve_triangular`, float64 --
chosen over the spec's other listed option
(`torch.linalg.lstsq(driver='gels')`) because the QR factorization
already produces R's diagonal as a byproduct, which the rank-degeneracy
flag needs anyway. No rank-revealing step (a genuine limitation, not an
oversight -- the spec explicitly says one isn't required in this
package; the docstring states what it would take: a pivoted/randomized
QR, since no `torch.linalg` primitive currently exposes column
pivoting, or a GPU SVD, which would reintroduce the same per-iteration
cost problem already documented for Beltrami's `analyze_conditioning`).

Before each GPU solve, `gpu_memory_estimate_bytes(N, P)` ($3 \times 8NP$)
is printed when `verbose=True` -- this is a prospective estimate, not a
new `iterations.csv` column (the schema is fixed since sub-batch 1;
`gpu_mem_peak_bytes`, the *measured* value, already had a column,
unused by every problem until now). After each solve,
`torch.cuda.max_memory_allocated()` is read (peak stats reset
immediately beforehand, so this is the factorization's own footprint,
not accumulated across the whole run) and logged into that column.

Rank-degeneracy flag: $\min_p|R_{pp}|/\max_p|R_{pp}| < 10^{-13}$ triggers
a real CPU `gelsy` solve for that iteration, purely to get a rank
estimate for the log (`num_rank_gelsy` is otherwise left empty on the
GPU path -- an honest "not computed", not a fabricated value). The GPU
iterate itself still drives the quasilinearization forward when this
fires; substituting the CPU result would silently change what "the GPU
run" actually measures, which the spec doesn't ask for. Verified via a
monkeypatch that forces a near-singular R diagonal on one call and
confirms the fallback rank appears in that row's log, without needing a
genuinely ill-conditioned physical config to hit naturally
(`test_degeneracy_flag_triggers_cpu_gelsy_cross_check`).

`_lstsq_cpu_gels` uses the real LAPACK `gels` routine via
`scipy.linalg.lapack.dgels` directly -- scipy's high-level `lstsq`
wrapper only exposes `gelsd`/`gelsy`/`gelss`, not `gels` -- for the
spec's "time both gelsy... and gels" comparison. Verified against
`gelsy` to ~1e-14 on both a synthetic system and a realistic
Kovasznay-scale one (P=48, N=2000) before being trusted.

`verify_gpu_cpu_equivalence(config)`: runs the same config once on each
device and checks `||beta_GPU - beta_CPU||_2/||beta_CPU||_2 <= 1e-8` and
`||R_lin||_h` agreement to six significant figures (implemented as
relative difference $\le 5\times10^{-7}$, the standard meaning of "equal
to six significant figures" -- not a literal string-formatting
comparison). Returns a dict rather than raising on failure itself --
"if it fails, stop and report" is an experiment-script-level decision,
not something a reusable comparison function should hard-code. Run for
real on an actual small Kovasznay config: `beta_rel_diff ~1.2e-15`,
`R_lin` relative difference `~1.2e-16` -- both dramatically inside
tolerance, essentially machine-precision agreement between the two
algorithms/devices.

**`lilq/iteration_log.py`**: `LilQDiagnosticsTracker.step()` extended
with optional `solver_path`/`gpu_mem_peak_bytes` parameters (defaulting
to the CPU-only values every existing caller relied on implicitly --
`"cpu_gelsy"`/`None` -- so this is additive, not a behavior change for
Bratu/Burgers/BL/Beltrami/Darcy, all still CPU-only) and `rank_gelsy`
now accepts `None` (previously `int(rank_gelsy)` would have crashed --
the GPU path's full-rank solve genuinely has no rank to report most of
the time). Verified the existing tracker/instrumentation test suite
still passes unchanged before building anything on top of it.

Verified `config.use_gpu=False` (the default) bit-identical against the
pre-change implementation loaded from git HEAD, same pattern as every
earlier solver change in this log.

198/198 tests passing.

---

## 2026-09-22 -- Four-method tables (Section 3.4), new lilq/four_method_log.py + experiments/four_method_tables.py

**Phase 1, sub-batch 10.** Section 3.4's rerun of NiL-N, NiL-Q, and LiL-N
(LiL-Q already has its own full Section 3.1 log from sub-batches 3-6/9)
across Bratu/Burgers/BL-viscous/BL-gravity, one CSV row per (benchmark,
P, method, seed, device), logging iterations, function evaluations,
wall-clock time, final loss, whether the target was reached, the
stopping reason, and the loss history every 10 iterations.

New `lilq/four_method_log.py`: `FOUR_METHOD_CSV_COLUMNS` schema +
`FourMethodLogger` (same reject-unknown-column, fill-omitted-with-None
contract as `IterationLogger`); `classify_stopping_reason(converged,
iterations_used, iterations_cap, line_searches_used, line_searches_cap)`
-- checks `iteration_cap` before `line_search_cap` on a tie, a genuine
judgment call (the summary dict a completed run returns doesn't preserve
which cap was hit *first* in wall-clock time, only both final counts),
made in the direction this codebase's own prior empirical finding
already points: DECISIONS.md's existing MAX_LINE_SEARCHES entries note
the line-search cap has never actually bound for Bratu in stored
results (iteration cap always binds first); `subsample_loss_history`
takes every 10th recorded row of a `MetricsTracker.to_dict()` (index
stride, not a re-filter -- `record()` is called once per optimizer step
with a contiguous per-call iteration counter, so stride 10 *is* "every
10 iterations", not an approximation) and always keeps the final row so
the last loss is never missing just because it fell off-stride.

New `experiments/four_method_tables.py`. Reuses every P/target-loss/
iteration-and-line-search-budget value directly from each problem's own
`experiments/run_*.py`, same DRY principle as `residual_band_figures.py`
(sub-batch 7) -- including BL's `max_line_searches` staying
**unset**, letting `BLOptConfig` derive it, matching the existing
`test_bl_experiment_runner_config.py` regression guard (an explicit
override previously truncated LiL-N before convergence). NiL-Q's
`iterations_cap` is `opt.max_quasi_iters_nn` (its outer-loop budget),
not `opt.max_iterations`/`total_iterations` -- NiL-Q has no single named
cap on its inner L-BFGS loop, only the outer quasi-iteration count and
the global line-search count -- confirmed directly against
`lilq.solvers.solve_nil_q`'s actual loop structure, not assumed from
NiL-N's shape; verified with a real run that `iterations_cap` and
`total_iterations` are genuinely different numbers
(`test_run_and_log_nil_q_uses_quasi_iter_cap_not_total_iterations`).
Seeds 0/1/2 for NiL-N/NiL-Q; LiL-N runs once (`seed` logged as empty),
per the spec's own exemption (same reasoning as `lilq.multiseed`).
Every benchmark's *largest* P also gets an additional CPU run alongside
the GPU one (this machine has a real CUDA device, confirmed directly,
not assumed) -- smaller sizes run on GPU only, matching "as in the
paper" plus the one-CPU-comparison-size requirement.

`--quick` (1 size, 1 seed, tiny iteration/line-search caps, real
tolerance `R_tol` left untouched so the run genuinely exercises "hit a
cap" rather than trivially converging) verified end-to-end for real
across all four problems -- 24 rows, ~9s total, CPU vs. GPU losses
matching to ~10 significant figures (real float non-determinism between
devices, not a bug, consistent with the spec's own GPU/CPU-equivalence
framing elsewhere being "six significant figures", not exact).

**The full (non-`--quick`) sweep was not run in this batch.** Unlike
`residual_band_figures.py`'s LiL-Q reruns (direct linear solves, cheap
even at the paper's largest sizes), this is real L-BFGS training with
iteration caps up to 10,000-20,000, at up to 3 seeds, across ~15
(benchmark, P) combinations, plus a duplicate CPU pass for the largest
size per benchmark -- a substantially larger compute commitment,
closer in kind to Component A's baseline search (which the spec itself
says takes "About 19 GPU-hours per family" and is explicitly not
something to run without deliberate intent). Flagged to the user rather
than launched automatically.

185/185 tests passing.

---

## 2026-09-22 -- run.json retrofitted onto Bratu/Burgers/BL/Kovasznay

**Phase 1, sub-batch 9.** Sub-batch 6 built `lilq/run_metadata.py` and
wired `run.json` into Beltrami/Darcy only, deliberately excluding the
other four problems and flagging the retrofit as separate, unscoped
work. This closes that out: `run_json_path` (optional, requires
`iteration_logger`, raises `ValueError` otherwise -- same contract as
Beltrami/Darcy) added to `problems.bratu.run_lil_q`,
`problems.burgers.run_lil_q`, `problems.buckley_leverett.run_lil_q`
(covers both viscous and gravity), and `problems.kovasznay.solve_kovasznay`.

Field values are genuinely derived per problem, not copy-pasted with the
names changed:

- **`initial_coefficients`** actually differs by problem, checked
  against each `initial_guess`/`_ic_no_gravity` function's real source
  rather than assumed: Bratu and Burgers are `"zero"` (confirmed
  `initial_guess` returns `zeros_like` unconditionally in both); BL is
  `"zero"` for gravity but `"fitted initial profile (least-squares
  pretrain of exp(-10*x))"` for viscous -- `BLPhysics.initial_guess`
  dispatches on `self._has_gravity`, and the non-gravity branch returns
  `exp(-10*x)`, not zero. This is exactly the spec's own callout
  ("zero, or the fitted initial profile for viscous Buckley-Leverett")
  verified against the actual code, not taken on faith; locked in by
  `test_run_json_initial_coefficients_distinguishes_viscous_and_gravity`.
  Kovasznay is `"zero"` (no pretraining step exists at all --
  `theta_u`/`theta_v`/`theta_p` are literally `np.zeros(...)`).
- **`N_composition`**/**`row_weights`** match each problem's actual row
  blocks: Bratu (pde, bc), Burgers/BL (pde, ic, bc_left, bc_right),
  Kovasznay (x/y-momentum, continuity, bc_u, bc_v, pressure_pin) --
  reusing the exact same weighted-block structure each problem's
  `_make_lil_q_system_fn`/assembly loop already uses (confirmed
  directly, not re-derived independently), so `N_total` sums to exactly
  what each problem's `A_stacked`/`A_sys` actually has.
- **`collocation_construction`** reports `"random-tensor"` for Bratu/
  Burgers/BL (all three pass `sampling='random'` through
  `generate_collocation_points_2d`, which places points via
  `np.random.uniform`) and `"equispaced tensor grid"` for Kovasznay
  (`_generate_collocation` calls `np.random.seed` but actually builds
  points via `np.linspace`/`meshgrid` -- the seed call is vestigial for
  this problem, confirmed by re-reading the function rather than
  assumed from its name).
- **`stopping_reason`** derives from each problem's own convergence
  signal: Bratu/Burgers/BL from `summary['converged']` (already computed
  by the shared `solve_lil_q`); Kovasznay has no such field in its
  return dict, so this reads `history['coeff_change'][-1] < config.tol`
  directly, matching `solve_kovasznay`'s own loop-exit condition exactly.

Verified bit-identical (two calls without `run_json_path`, matching the
established pattern) for all four; each also smoke-tested with a real
small solve and its `run.json` inspected by hand (N_total/P_total sums
checked against `N_composition`/`P_composition`, BL's viscous vs.
gravity `initial_coefficients` value both printed and confirmed
correct) before writing the automated tests.

164/164 tests passing.

---

## 2026-09-22 -- Multi-seed harness wired into run_bratu.py/run_burgers.py/run_bl.py

**Phase 1, sub-batch 8.** `lilq.multiseed.run_multiseed` was built (and
fully unit-tested) back in Phase 1 batch 2, but deliberately left
unwired into any experiment script at the time -- a decision the user
explicitly flagged and asked not to be forgotten. This closes that out,
scoped narrowly to *wiring the harness in* (Section 2/3.4's "seeds 0, 1,
2 for NiL-N and NiL-Q"), not the full Section 3.4 `four_method_tables.csv`
logging apparatus (stopping reason, function-evaluation counts, loss
history every 10 iterations) -- that remains separate, unscoped work.

New `experiments.exp_utils.run_stochastic_with_seeds(runner, config, opt,
seeds, device, verbose)`: with `seeds` falsy, calls `runner` once exactly
as before (unchanged default behavior, verified bit-identical against a
real Bratu NiL-N run at the same seed); with `seeds` given, runs
`lilq.multiseed.run_multiseed` across all of them and returns the
**median-`final_loss`-seed's own** `(model, metrics, summary)` tuple --
not an average of three trained networks, which has no principled
meaning -- with `summary` additionally carrying the full
`multiseed_aggregate` (median/min/max per field, convergence rate across
all seeds, from `lilq.multiseed.aggregate_summaries`), `multiseed_seeds`,
and `multiseed_representative_seed`. This keeps the *shape* of what every
downstream consumer (checkpoint saving, solution-field plotting,
`save_summary_json`) already expects unchanged, so multi-seed mode
slots into the existing single-run pipeline without touching it.

Wired into all three scripts' `run_experiment_for_N` (Bratu, Burgers,
BL -- both viscous and gravity share BL's one `run_experiment_for_N`):
a new `seeds=None` parameter, threaded only into each script's non-LiL
(NiL-N/NiL-Q) branch via `run_stochastic_with_seeds`; the LiL-N/LiL-Q
branch is untouched and never goes through it, matching
`lilq.multiseed`'s own documented exemption (deterministic from zero, no
random initialization to average over -- confirmed this reasoning
still holds, not just asserted). Each script gained a `--seeds` CLI flag
(e.g. `--seeds 0 1 2`); omitted, every method runs exactly as before this
change -- verified bit-identical against a real Bratu NiL-N run, and
locked in by a regression test that would fail if a future edit routed
the LiL branch through `run_stochastic_with_seeds` by mistake or dropped
it from the NiL branch. `seeds` is also recorded in each script's
`*_master_results.json` `config_info` for provenance.

Verified with a real (not mocked) 3-seed Bratu NiL-N run
(`run_stochastic_with_seeds` called directly against the actual
`run_nil_n`): all three seeds genuinely ran, the representative seed
selected was the one whose `final_loss` was the true median of the
three, and the aggregate's median/min/max matched a hand check.

**Not done here, deliberately out of scope**: the four-method-table
CSV/stopping-reason/function-eval-count logging Section 3.4 also asks
for; Kovasznay/Beltrami/Darcy don't have NiL-N/NiL-Q at all (Kovasznay
and Beltrami are LiL-Q only per their own module docstrings, Darcy's
`DarcyPINN`/`run_nil_n_darcy` is a separate, still-stubbed path per
earlier DECISIONS.md entries) so they're untouched by this batch, not
silently included or silently skipped -- there was nothing for this
harness to wire into there.

140/140 -> 156/156 tests passing.

---

## 2026-09-22 -- Residual-band figures (Section 3.5), new experiments/residual_band_figures.py

**Phase 1, sub-batch 7 of the iterations.csv work.** The last piece of
Component B that consumes the instrumentation built in sub-batches 3-6
rather than adding more of it: Section 3.5 asks for four figures
comparing $\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$ and
$\|\mathbf{R}^{(k+1)}\|_h$ against $k$, one panel per $P$, for Bratu,
Burgers, and both Buckley-Leverett variants, with the y-axis correctly
labeled $\|\cdot\|_h$ ("the present figures say MSE, which is wrong" --
moot here since the repository had no such figures at all, correct or
not, to begin with).

New `experiments/residual_band_figures.py`. Design choices:

- **Every P/target-loss/quasi-iter-budget value is imported directly**
  from that problem's own `experiments/run_*.py` (`DEFAULT_N_VALUES`,
  `TARGET_LOSSES`, `MAX_QUASI_ITERS`, etc.) rather than duplicated as
  separate constants -- guarantees this script uses the exact same
  "paper settings" (Section 3.3) already established elsewhere in this
  codebase, with no risk of silently drifting out of sync with them.
  `pretrain_epochs` is deliberately *not* imported from any of them:
  confirmed by re-reading each problem's `run_lil_q` that the shared
  `pretrain_lil` step (used identically by all three) is a least-squares
  fit, not an epoch-based optimizer, so that config value has no effect
  on the LiL-Q path this script exercises.
- $\chi_k$ goes in **a second row of panels**, not a secondary axis --
  the spec explicitly offers both; a secondary axis was rejected because
  $\chi_k$'s own dynamic range (seen directly in the real reruns below:
  span of 15+ orders of magnitude within a single BL-gravity P) needs
  its own log-scale axis, and forcing it to share space with two
  already-log-scale residual curves would be unreadable.
  `test_plot_residual_bands_produces_pdf_and_png` covers this indirectly
  via a real run; the panel layout itself was verified by eye against
  the actual PDFs (see below) before committing, not merely inferred
  from the code.
- Each (problem, $P$) run's full `iterations.csv` is saved alongside the
  figure -- literally "the CSV behind each panel" the spec asks for,
  reusing `IterationLogger.to_csv` rather than inventing a
  plot-data-only format; a reader gets the complete Section 3.1 row set
  for that run, not just the four columns the figure itself plots.
- `--quick` flag (first 2 $P$ per problem) for fast iteration/testing
  without committing to the full paper-scale sweep; `main()`'s default
  (no flag) is the real full sweep matching Section 3.3 exactly.

**Verified against real output, not just "the code runs":** ran the
full (non-quick) sweep for real -- all 4 figures + 15 CSVs
(3+5+4+4 = 16 total (problem, $P$) reruns) generated in well under a
minute total (LiL-Q's direct-solve-per-iteration cost stays cheap even
at $P=1024$; nothing here resembles Beltrami's cost profile). Visually
inspected all four PDFs: axis labels, log scaling, and the two-curve
comparison all match the spec; Bratu/Burgers converge in as few as 2-4
iterations at their tight paper tolerances (consistent with LiL-Q's
already-documented fast convergence elsewhere in this project, not a
bug); Buckley-Leverett (both variants) shows richer 6-10-iteration
trajectories with a visible transient bump in $\|\mathbf{R}_{\mathrm{lin}}\|_h$
before it settles, which is real algorithm behavior on the more
nonlinear problem, not a plotting artifact.

**Not done here, deliberately out of scope**: the generated
`results/residual_band_figures/` output itself is not committed
(`results/` is gitignored, matching this repository's existing
convention of tracking scripts, not their generated artifacts, with
`reference_results/` the sole documented exception). Re-running this
script is how a reader reproduces the figures, not a stored copy in git.

140/140 tests passing.

---

## 2026-09-22 -- Beltrami and Darcy wired to Section 3.1 instrumentation; run.json added

**Phase 1, sub-batch 6 of the iterations.csv work** -- the last two
problems, plus the "once per run" `run.json` metadata file
(Section 3.1's own list, separate from `iterations.csv`'s per-iteration
columns) that Sections 3.3 and 6 require alongside it but which no
problem had until now.

**Beltrami**: same self-contained-loop situation as Kovasznay (sub-batch
5) -- `solve_beltrami` never calls `solve_lil_q`, so this manually drives
a `LilQDiagnosticsTracker` again. New `_make_beltrami_nonlinear_loss_fn`/
`_make_beltrami_residual_vector_fn`, generalizing the one-scalar-per-block
convention to Beltrami's 26 row-blocks (momentum x3, continuity, BC-u/v/w
x6 faces, IC-u/v/w, pressure pin) -- same "independently re-derived, not
shared code with the assembly loop" discipline as every prior residual-
vector function, so check B2 is a real cross-check (verified `rel_err`
down to exact `0.0` on one real run). Interior-row unweighting is
conditional on `lambda_mom == lambda_cont`, same reasoning and same
guard-with-regression-test discipline as Kovasznay. `iteration_logger` is
CPU-only (`config.use_gpu=False`) -- raises `NotImplementedError`
otherwise, since the GPU path uses `torch.linalg.lstsq(driver='gelsd')`,
not `gelsy`, and a GPU instrumentation path is Section 3.2's concern,
explicitly Kovasznay-only in the spec. At the manuscript's own Beltrami
scale (P_total=7,984, confirmed by direct calculation:
`3*6**4 + 8**4 == 7984`), the tracker automatically uses pivoted-QR at
the final iterate only (never a full per-iteration SVD) purely because
P_total exceeds `DEFAULT_SVD_CONDITIONING_THRESHOLD` -- this is Section
3.1 item 8's Beltrami-specific requirement, already satisfied by the
generic tracker design from sub-batch 1 with no Beltrami-specific code
needed; locked in by `test_large_P_uses_pivoted_qr_at_final_iterate_only`.
Also made the CPU `_lstsq` branch's `cond=EPS_MACH` explicit (was
`cond=None`, same LAPACK `gelsy` RCOND=-1 convention as every earlier
instance of this change) to capture `rank_gelsy`, which the original code
discarded -- verified bit-identical against the pre-change implementation
loaded from git HEAD.

**Darcy**: a genuinely different case from every other problem in this
codebase -- `solve_lilq_darcy`'s system (Darcy-x, Darcy-y, continuity) is
**linear** in h_tilde*/u*/v*, so there is no Bellman-Kalaba
quasilinearization loop at all, just one direct `lstsq` solve. This
produces exactly one logged row (`k=1`) rather than driving an iteration
loop. Consequences, each locked in by
`test_single_row_reflects_linear_system_structure`: `order_obs` is NaN
and `stall_flag` is False (no k-1 history, same degenerate case every
other problem's first row already exercises); `chi` evaluates to exactly
`0.0`, not NaN -- because the system is linear, the nonlinear residual
and the linearized residual are literally the same operator
(`A @ beta - b`), so `compute_residual_vector_fn` needed no separate
formula, and a chi of 0 is the mathematically correct statement that
there is no linearization error for a linear problem; `norm_R_interior`
equals `norm_R_h` exactly, because Darcy's boundary conditions are
satisfied exactly by construction (the lifting function and augmented
bases bake them into the basis functions themselves -- see the module
docstring), so there is no separate BC row block and *every* row is a
PDE/interior row (`n_interior_rows = A.shape[0]`, `interior_weight=1.0`).
Darcy also has no `lambda_pde`/`lambda_bc`-style row-weighting scheme at
all (the only row scaling is the `sqrt(K*)` physics normalization,
applied per-row rather than as a single block scalar) -- `run.json`'s
`row_weights` field says this explicitly rather than forcing it into the
other problems' single-scalar-per-block shape. `iteration_logger` is only
supported for `config.solver_method == 'qr'` (the `gelsy` driver) --
raises `NotImplementedError` for `'lstsq'`, which calls
`numpy.linalg.lstsq` (a different driver; the schema's `num_rank_gelsy`
column specifically means the scipy `gelsy` rank). Also made the `qr`
branch's `cond=EPS_MACH` explicit to capture rank, same pattern, verified
bit-identical.

**`lilq/run_metadata.py` (new module)**: `run.json` schema/writer for
Section 3.1's "once per run" metadata list (N/P composition, row
weights, collocation construction, basis description, initial
coefficients, solver driver/rcond, the stopping rule actually used and
K_max, stopping reason, first stall iteration, device/thread count).
Unlike `iterations.csv` (one fixed column schema every problem fills in
identically), every problem's row/field composition genuinely differs
(Darcy's 3 unweighted blocks vs. Beltrami's 26 lambda-weighted ones), so
`N_composition`/`P_composition`/`basis_description`/`row_weights` are
free-form dicts rather than fixed columns -- only the top-level field
*names* are fixed (`build_run_metadata` rejects an unrecognized one, same
"typo should fail loudly" discipline as `IterationLogger.record`).
`first_stall_iteration(rows)` scans an `IterationLogger.rows` list for
the first `stall_flag=True` (or `None` if the run never stalled),
reusable by any problem. Wired into `solve_beltrami`/`solve_lilq_darcy`
via a new optional `run_json_path` parameter, requiring `iteration_logger`
be given alongside it (raises `ValueError` otherwise) since
`first_stall_iteration` needs the logged rows. `thread_count` reads
`OMP_NUM_THREADS` via the existing `lilq.provenance.capture_blas_thread_env`,
falling back to `os.cpu_count()` (not `1`) when unset -- an unset BLAS
thread env var means "use every core available", and reporting `1` would
have been actively wrong, not just imprecise; caught and fixed before
committing, not left as a plausible-looking bug.

**Not done here, out of scope, flagged rather than silently assumed**:
Bratu/Burgers/BL/Kovasznay (sub-batches 3-5) do **not** yet have
`run.json` wired in -- this sub-batch was explicitly scoped to
"Beltrami/Darcy + run.json" by the user, and retrofitting the other four
problems is a separate decision, not assumed as included here. Whether
`experiments/run_*.py` should be changed to always produce `run.json`
alongside `iterations.csv` for real experiment runs is also undecided,
same as the equivalent `iterations.csv` question flagged in sub-batch 3.

123/123 -> 134/134 tests passing.

---

## 2026-09-22 -- Kovasznay wired to Section 3.1 instrumentation (own solver, manually driven)

**Phase 1, sub-batch 5 of the iterations.csv work** -- the Kovasznay
follow-up flagged as out-of-scope for sub-batch 4. Unlike Bratu/Burgers/
BL, `problems/kovasznay.py`'s `solve_kovasznay` never calls
`lilq.solvers.solve_lil_q`; it is a self-contained three-field
(u, v, p) quasilinearization loop with its own hand-rolled `history`
dict and its own `rel_delta`-based convergence check. Wiring it required
manually driving a `LilQDiagnosticsTracker` inside that loop rather than
just adding parameters to a thin `run_lil_q` wrapper.

New pieces:

- `_make_kovasznay_nonlinear_loss_fn` / `_make_kovasznay_residual_vector_fn`
  (both new -- Kovasznay had no pre-existing loss/residual helpers to
  extend): the scalar-MSE and weighted-vector forms of the same total
  residual, generalized to Kovasznay's 12 row-blocks (x-momentum,
  y-momentum, continuity, BC-u/BC-v for 4 edges, pressure pin) with the
  same one-scalar-per-block weighting (`w_mom=sqrt(lambda_mom/n_pde)`,
  `w_cont=sqrt(lambda_cont/n_pde)`, `w_bc=sqrt(lambda_bc/n_edge)` per
  edge, `w_pin=sqrt(lambda_bc)` for the single pin row -- confirmed by
  direct derivation that `norm(vector)**2 == total` exactly). Each
  function independently re-derives the raw nonlinear residuals rather
  than sharing code with the assembly loop or with each other, so check
  B2 is a real cross-check.
- `solve_kovasznay` gained the same optional `iteration_logger` parameter
  (default `None`, `history` and all existing behavior unchanged when
  omitted -- verified bit-identical against the pre-change implementation,
  loaded from git HEAD and run side-by-side, not just asserted).
- Split `t_assemble_s`/`t_solve_s` timing via `time.perf_counter()`
  around the existing `A_sys`/`b_sys` build and `lstsq` call respectively
  (additive -- the pre-existing `time.time()`-based combined `dt` and
  `history['solve_time']` are untouched). Made the `lstsq` call's `cond`
  explicit (`cond=EPS_MACH`, same as the `solve_lil_q` change earlier in
  this log) to capture `rank_gelsy`, which the original code discarded
  entirely (`...lstsq(...)[0]`) -- verified bit-identical before/after,
  same LAPACK `gelsy` RCOND=-1 convention as before.

**Interior-row unweighting is conditional, not universal.** Kovasznay's
leading (PDE) rows are actually two sub-blocks weighted independently --
momentum (`w_mom`, 2*n_pde rows) and continuity (`w_cont`, n_pde rows) --
which only collapse to the tracker's required single leading scalar
weight when `lambda_mom == lambda_cont` (true for `KovasznayConfig`'s
defaults, and for every config touched so far, but not guaranteed in
general). `solve_kovasznay` checks this explicitly and only passes
`n_interior_rows`/`interior_weight` to the tracker when it holds; falls
back to the tracker's existing documented NaN behavior otherwise --
locked in by `test_interior_norms_nan_when_lambda_mom_and_cont_differ`
in the new `tests/test_kovasznay_instrumentation.py`, not left as an
unverified assumption.

Check B2 verified against a real solve (`rel_err ~1e-16`, both via the
automated test and manually against the actual returned coefficients).
Manually inspected a real-scale `iterations.csv` (default config:
Re=40, chebyshev N_x=N_y=15, P_total=675) -- 7 iterations to
convergence, `stall_flag` correctly latching `True` once at the
round-off floor, `num_rank_svd == num_rank_gelsy == P_total` throughout,
`chi` decreasing then noisy at the floor -- consistent with the pattern
already seen on Bratu.

Not done here, out of scope: `eps_u`/`eps_v`/`eps_p`/`eps_p_meanfree`
(per-iteration test error against the known Kovasznay exact solution)
are left unpopulated, same as Bratu/Burgers/BL -- `solve_kovasznay`
already computes final-iterate `rel_l2_u/v/p` once at the end, but
wiring per-iteration error columns (and deciding what `eps_p_meanfree`
should mean given the pressure pin, rather than a true mean-free
projection) is a separate piece of work, not assumed as part of this
batch.

108/108 -> 112/112 tests passing.

---

## 2026-09-22 -- Burgers and Buckley-Leverett (viscous + gravity) wired to Section 3.1 instrumentation

**Phase 1, sub-batch 4 of the iterations.csv work.** Rolled out the exact
pattern sub-batch 3 proved on Bratu to the two other problems that share
`lilq.solvers.solve_lil_q`:

- `problems/burgers.py`: `run_lil_q` gained the same optional
  `iteration_logger` parameter, plus a new `_make_lil_residual_vector_fn`
  (vector form of `_make_lil_nonlinear_loss_fn`, stacked
  PDE-then-IC-then-BC-left-then-BC-right, matching
  `_make_lil_q_system_fn`'s exact weighting). `n_interior_rows=n_pde`,
  `interior_weight=sqrt(lambda_pde/n_pde)` -- confirmed by direct
  derivation that the weighted vector's squared norm equals `total_loss`
  even though the BC block is itself split across two sub-blocks
  (`bc_left`/`bc_right`) with independently-computed weights
  (`w_bl=sqrt(lb/n_bc_l)`, `w_br=sqrt(lb/n_bc_r)`) rather than one shared
  scalar -- summing their squared contributions still reduces to
  `lb*(mean_left + mean_right)`, matching `compute_loss`'s `bc` term
  exactly.
- `problems/buckley_leverett.py`: same pattern, covering **both**
  configurations (`BLConfig()` viscous and `BLConfig.with_gravity()`)
  since they share one `run_lil_q` -- gravity vs. viscous is fully
  internal to `physics.flux`/`flux_derivative`'s dispatch on `config.N_g`,
  invisible to the instrumentation wiring. The residual-vector function
  uses the same detached `physics.flux_derivative` the existing
  `_make_lil_nonlinear_loss_fn` already used (a forward-only evaluation
  for logging, not a gradient path) -- not the graph-preserving
  `flux_derivative_differentiable` needed only for LiL-N's `.backward()`
  (see the LiL-N gradient fix entry further down this log). BL's BC
  targets are non-zero (`physics.bc_left`/`bc_right`, unlike Bratu's/
  Burgers' homogeneous Dirichlet BCs), so the residual vector subtracts
  them explicitly, matching `_make_lil_q_system_fn`'s `b_stacked`.

Both are backward-compatible (bit-identical coefficients/summary with
`iteration_logger=None`, the default) and both pass check B2
(`test_check_b2_*` in the new `tests/test_burgers_instrumentation.py` /
`tests/test_bl_instrumentation.py`, same independent-reconstruction-via-
public-API method as Bratu's check) at `rel_err < 1e-10`. 108/108 tests
passing.

**Kovasznay deliberately excluded from this batch, not silently
dropped.** Unlike Bratu/Burgers/BL, `problems/kovasznay.py`'s
`solve_kovasznay` is a wholly self-contained quasilinearization loop --
it does not call `lilq.solvers.solve_lil_q` at all, has no
`_make_lil_q_system_fn`/`_make_lil_nonlinear_loss_fn`-style helpers to
reuse, and already computes its own condition number and test errors
(`rel_l2_u/v/p`) inline every iteration via a hand-rolled `history` dict
rather than `IterationLogger`. Wiring it to the shared Section 3.1 schema
means manually replicating the tracker-calling pattern inside that loop
(three coupled fields stacked into one system, `eps_p_meanfree` needing a
real decision for a pressure field pinned at one point rather than the
scalar problems' single-field error columns) -- structurally different,
larger work than the three pattern-repeats above. Flagging this now
rather than assuming it belongs in "sub-batch 4" as originally scoped;
proposed as its own follow-up sub-batch.

---

## 2026-09-22 -- Bratu wired end-to-end to Section 3.1 instrumentation; check B2 passes

**Phase 1, sub-batch 3 of the iterations.csv work ("proof of concept":
wire one real problem all the way through and verify before rolling out
further).** Two parts:

1. `LilQDiagnosticsTracker`/`solve_lil_q` extended with `n_interior_rows`/
   `interior_weight` constructor/call parameters to actually compute
   `norm_R_interior`/`norm_Rlin_interior` (sub-batch 2 left these
   permanently `NaN`, documented as deferred). Both were `NaN`-by-default
   and additive, so this needed no new test of the "does this break
   anything" kind beyond the two new interior-row-specific tests added to
   `tests/test_lil_q_diagnostics_tracker.py`.
2. `problems/bratu.py`'s `run_lil_q` gained an optional `iteration_logger`
   parameter (default `None`, existing behavior unchanged when omitted --
   confirmed bit-identical coefficients/summary with and without it in
   `tests/test_bratu_instrumentation.py`). When given, it builds a new
   `_make_lil_residual_vector_fn` (the vector form of
   `_make_lil_nonlinear_loss_fn` -- same weighted interior-then-boundary
   stacking as `_make_lil_q_system_fn`'s `A_stacked`/`b_stacked`, verified
   by construction: `norm(vector)**2 == total_loss`) and passes
   `n_interior_rows=n_pde`, `interior_weight=sqrt(lambda_pde/n_pde)`
   through to `solve_lil_q`, matching the one-scalar-per-block convention
   confirmed directly in Bratu's own `assemble_system_fn` (documented in
   `lilq/iteration_log.py`'s `LilQDiagnosticsTracker` docstring).

**Check B2** (Computational_Package_1_v2.md Section 3.1: the logged
$\|\mathbf{R}^{(k)}\|_h$ must match the nonlinear operator evaluated
directly at the collocation points, to $10^{-10}$ relative) implemented
as an automated test
(`test_check_b2_residual_identity_against_direct_evaluation`), run
against a real (not synthetic) Bratu solve: reconstructs
`A_u`/`A_uxx`/`A_uyy`/`A_bc` from scratch via the public
`create_basis_2d`/`generate_collocation_points_2d`/`basis.evaluate`/
`basis.derivative` API (same config/seed as `run_lil_q`'s own internal
setup -> deterministic, same collocation points -- confirmed
`generate_collocation_points_2d` reseeds `np.random` internally
regardless of prior RNG state consumed during pretraining), independent
of any of `problems.bratu`'s private helper functions. Passed at
`rel_err < 1e-10` (this codebase's own float64 near-machine-precision
scale, not a loosened tolerance).

Manually inspected two real `iterations.csv` outputs (not committed --
scratch verification, not fixtures) to sanity-check the full 30-column
row at scale: a P=100 Bratu solve converging in 1 iteration under the
default `R_tol=1e-4` (matches the P=100 `final_loss` value already
recorded in `tests/test_solve_lil_q_instrumentation.py`'s bit-identical
check), and the same problem forced to run 25 iterations at a much
tighter `R_tol=1e-12` past its actual round-off floor -- `stall_flag`
correctly latches `True` once `chi`/`norm_Rlin_h` stop moving, and
`order_obs` oscillates (sign flips, occasional large magnitudes) once in
that floor regime, which is the expected behavior of a quadratic-rate
estimator applied to noise rather than a bug.

Not yet done, deliberately out of scope for this batch (per the
established "one step at a time" discipline): rolling this same pattern
out to Burgers/BL-viscous/BL-gravity/Kovasznay (sub-batch 4), and
deciding whether `experiments/run_bratu.py` itself should be changed to
always produce `iterations.csv` for real experiment runs -- that changes
a script's default output and is flagged here rather than assumed, the
same way the multi-seed harness's non-wiring was flagged rather than
silently deferred.

---

## 2026-09-22 -- solve_lil_q's lstsq call: cond=None made explicit as cond=EPS_MACH

**Phase 1, sub-batch 2 of the iterations.csv work.** While wiring the
new `rcond` log column, noticed `solve_lil_q`'s `scipy.linalg.lstsq`
call never passed `cond` explicitly (relying on scipy's `cond=None`
default). Per LAPACK's own `gelsy` convention, `cond=None` maps to
`RCOND=-1`, which means "use machine precision" -- i.e. already
numerically identical to explicitly passing `cond=EPS_MACH`. Verified
directly before changing anything: ran the same `lstsq` call both ways
on a random matrix and confirmed bit-identical output
(`np.array_equal(x1, x2) == True`, max diff `0.0`). Made explicit so the
logged `rcond` value matches what the code actually does rather than
relying on an unstated LAPACK convention -- a documentation/clarity
change, confirmed not a numeric one.

---

## 2026-09-22 -- BLAS thread counts now actually set, not just recorded

**Phase 1, batch 2 follow-up.** The provenance-capture entry below
noted this as a "not yet acted on" gap -- correctly challenged as having
no real reason to wait for a later phase, so closed out the same day.

`lilq/blas_threads.py`: sets `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`MKL_NUM_THREADS` (all three, since which one actually governs depends on
which BLAS backend numpy/scipy link against -- setting the irrelevant
ones is harmless) as its own import-time side effect -- the one place in
this codebase where that pattern is the *correct* choice rather than the
antipattern fixed elsewhere in this log: there is no non-environment-
variable way to configure a native BLAS library's thread pool before it
loads, and BLAS reads these at load time, not dynamically. Default value:
`SLURM_CPUS_PER_TASK` when running under sbatch/salloc (correctly matches
the actual allocation rather than the whole node's core count on shared
HPRC nodes), else every logical core on the machine. Never overrides a
value the environment already set (`os.environ.setdefault`, not
assignment) -- an explicit shell/job-script choice always wins.

Because of the load-time requirement, `import lilq.blas_threads` has to
be the first import in every experiment script's entry point, before
`import numpy`/`import scipy` and before any `from lilq...`/
`from problems...` that would pull numpy in transitively. Wired into all
8 real experiment scripts at exactly that position (one script,
`run_burgers_basis_comparison.py`, had numpy imported *before* its own
`sys.path` setup -- reordered so blas_threads and numpy both come after).

Verified end-to-end, tying both provenance-capture pieces together: ran
`run_kovasznay.py` with all three variables explicitly unset in the
calling shell, and confirmed the resulting `hardware.json` shows them all
set to this machine's core count (24) -- the two mechanisms working
together exactly as intended, not just independently unit-tested.

---

## 2026-09-22 -- Provenance capture (hardware.json/environment.txt) and multi-seed harness added

**Phase 1, batch 2.** Two new, independent, tested building blocks
required by Computational_Package_1_v2.md Section 2:

**`lilq/provenance.py`** -- captures everything Section 2 asks for:
CPU/GPU info (`nvidia-smi` for GPU name/driver/memory, `/proc/cpuinfo` on
Linux or `platform.processor()` elsewhere for CPU model), the
`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` environment
variables actually in effect, numpy/scipy/torch versions, verbatim
`numpy.show_config()`/`scipy.show_config()` text, and the git commit hash
+ branch + any uncommitted diff. Every capture function degrades to
`None`/`"available": False` on a missing tool rather than raising --
provenance capture must never be why a multi-hour run crashes.

This directly retires the "which run/code/machine produced this number"
class of question that most of the earlier Q1/Q2/Q5/Q6 investigation was
spent answering after the fact (Codebase_v3_Proposal.md S2.5). Confirmed
live: the first real capture on this machine shows
`blas_thread_env: {OMP_NUM_THREADS: null, OPENBLAS_NUM_THREADS: null,
MKL_NUM_THREADS: null}` -- directly, immediately visible confirmation of
Q5's finding (no thread-pinning is set anywhere), instead of something
that took code-reading to establish.

Wired into all 8 real experiment scripts (`run_bratu.py`, `run_burgers.py`,
`run_bl.py`, `run_kovasznay.py`, `run_elasticity.py`, `run_beltrami.py`,
`run_darcy.py`, `run_burgers_basis_comparison.py`) via a new
`save_run_provenance()` helper in `exp_utils.py`, called once per script
invocation right alongside the existing `save_master_results()` call --
not `run_all_dry.py`, which is a fast pipeline smoke test, not a real
timed run. Verified end-to-end: a real Kovasznay run now writes
`hardware.json`/`environment.txt` into its results directory automatically.

**Note:** the spec says these thread-count variables must be *set*, not
just recorded -- this module only ever reported what was already in
effect. See the "BLAS thread counts now actually set" entry above
(newer, listed first) for `lilq/blas_threads.py`, which closes that gap.

**`lilq/multiseed.py`** -- implements the spec's "every stochastic method
runs at seeds {0,1,2} (Component B) / {0-4} (Component A), report
median/min/max, never a single run" requirement. `run_multiseed(runner,
config, seeds, *args, **kwargs)` calls a `run_nil_n`/`run_nil_q`-style
function once per seed via `dataclasses.replace` (never mutates the
config passed in), collects each run's summary dict (always the last
element of the returned tuple, true for every `run_*` function in
`problems/*.py`), and aggregates numeric fields into median/min/max plus
a convergence *rate* for the boolean `converged` field (a median of
booleans isn't meaningful; a rate is).

Deliberately scoped to NiL-N/NiL-Q only -- LiL-N/LiL-Q have no random
initialization of their own (coefficients come from a deterministic
least-squares pretrain fit) and the spec explicitly exempts them
("LiL-N from zero is deterministic; one run", Section 3.4); sweeping
seeds for them would only vary the collocation set, which the spec asks
to keep fixed at seed 42 instead. The utility itself is runner-agnostic
and doesn't special-case this -- it's a caller decision which methods to
sweep.

**Not yet wired into the existing experiment scripts.** Wiring this into
`run_bratu.py`/`run_burgers.py`/`run_bl.py`'s NiL-N/NiL-Q calls would
meaningfully change those scripts' output format and roughly triple their
NiL-method runtime -- a bigger, more disruptive change than provenance
capture's "add two files, nothing else changes." Since the real target
for this requirement is Section 3.4's instrumented reruns (which will
restructure these scripts substantially anyway -- the `iterations.csv`
logger, stall detection, etc.), wiring it in now and restructuring again
later would be duplicate work. Verified instead via a real integration
test: three actual seeds against Bratu's `run_nil_n` at a tiny size,
confirming genuinely different results per seed (final loss 0.31 / 0.25 /
0.063) and correct aggregation.

---

## 2026-09-22 -- torch.set_default_dtype() import-time side effect removed

**Phase 1, batch 1 (second half).** `lilq/solvers.py` used to call
`torch.set_default_dtype(torch.float64)` at module import time -- a
process-wide mutation that silently made every later `nn.Module`
construction anywhere in the process default to float64, regardless of
which file did the constructing, based on import order rather than an
explicit choice. Same class of problem as `set_seed`'s old unconditional
CUDA touching (previous entries): a shared setup step mutating global
state instead of taking explicit configuration -- and, per
`Codebase_v3_Proposal.md` S3, the two were always meant to be fixed
together.

**Fix:** `MLP` (`lilq/nn.py`) now takes its own `dtype: torch.dtype =
torch.float64` parameter, passed directly to each `nn.Linear(...)`
construction. The default preserves every current NiL-N/NiL-Q call
site's existing behavior with zero call-site changes needed (Bratu,
Burgers, and BL all construct `MLP(...)` with no explicit dtype and
correctly keep getting float64). The `torch.set_default_dtype` call
itself was deleted from `lilq/solvers.py`.

Audited every bare (no explicit dtype) `torch.tensor`/`linspace`/`zeros`/
`full`/`empty` construction across `problems/`, `lilq/`, and
`experiments/` (grep, cross-checked against multi-line calls to avoid the
false negatives a naive single-line grep would give) and made each one
explicit:
- `problems/bratu.py`: the IC-loss placeholder and the evaluation-grid
  `linspace` calls -> `dtype=torch.float64`.
- `problems/burgers.py`, `problems/buckley_leverett.py`: same
  evaluation-grid `linspace` pattern -> `dtype=torch.float64`.
- `problems/darcy.py`: `DarcyPINN`'s three `MLP(...)` calls now pass
  `dtype=torch.float32` directly instead of constructing at float64 and
  immediately downcasting via `.float()`.

**Bonus find while auditing Darcy:** `DarcyPINN.xleft`/`.xright` (two
bare `torch.zeros`/`torch.full` calls) had **no explicit dtype at all**,
silently inheriting float64 from the global default while every other
tensor in the same class (`xbot`, `xtop`, `xpde`, etc., all built via an
explicit-float32 `_t()` helper) was float32 -- a real, if currently
harmless (no failure observed, likely masked by implicit type promotion
somewhere downstream), latent inconsistency that would have become a
silent float32 gap once the global default was removed. Fixed to
`dtype=torch.float32`, matching the rest of the class's clear intent.
Darcy's PINN path remains out of scope for this package (Phase 0), but
since this codebase-wide dtype audit touched it anyway, worth fixing
while here rather than leaving a known inconsistency for later.

**Verified:**
- `tests/test_dtype_explicit.py`: importing `lilq.solvers` in a fresh
  subprocess leaves `torch.get_default_dtype()` unchanged; `MLP()`
  defaults to float64 even under deliberately hostile global state
  (`torch.set_default_dtype(torch.float32)` set immediately before
  construction); `MLP(dtype=torch.float32)` still works for Darcy.
- End-to-end: ran real NiL-N solves for Bratu, Burgers, and BL and
  confirmed float64 parameters and float64 evaluation-grid output;
  constructed a real `DarcyPINN` and confirmed float32 parameters and
  (now-fixed) float32 `xleft`/`xright`.

---

## 2026-09-22 -- Beltrami's real slowdown cause: unconditional per-iteration SVD, not set_seed

**Phase 1, batch 1.** While verifying the `set_seed` fix below actually
resolved Beltrami's known slowdown (Q6: paper ~297s, GitHub/v3-dev
~513-539s), a same-machine, same-moment, three-way controlled comparison
told a different story than the original Q6 investigation had concluded:

| Run (today, same machine, back to back) | Time |
|---|---|
| pre-GitHub codebase, actual code, run fresh | **302.0s** |
| v3-dev, `set_seed` new default (`deterministic_cuda=False`) | 539.0s |
| v3-dev, `set_seed` old behavior forced (`deterministic_cuda=True`) | 558.8s |

The `set_seed` fix only accounts for ~20s (539 vs. 559) -- nowhere near
the ~1.7x originally attributed to it. That original comparison was made
hours apart in this same session and wasn't a clean, isolated A/B test;
running pre-GitHub's *actual* code right now (302.0s, matching its
original ~297s closely) ruled out "the machine is just slower today" and
confirmed a real, current, code-level gap the `set_seed` fix doesn't explain.

**Real cause, found by reading `problems/beltrami.py`'s solve loop
directly:** `solve_beltrami` computes `np.linalg.cond(A_sys)` -- a full
SVD on the system matrix, P_total=7,984 columns -- **unconditionally,
every outer iteration**, and nothing downstream ever reads the result
(`history['cond_number']` is appended to and never consumed anywhere).
Pre-GitHub's `beltrami_core.py` has the identical computation but gates
it behind `analyze_svd=False` (default off) -- confirmed by reading that
file directly, not by assumption.

This also isn't just an efficiency gap -- it's the *wrong* algorithm per
the computational package spec: Section 3.1 item 8 explicitly requires
full-SVD conditioning only for $P \le 3{,}200$, and prescribes a cheaper
pivoted-QR check **at the final iterate only** for Beltrami specifically
(P=7,984, over that threshold), precisely because per-iteration SVD isn't
practical at this scale. The unconditional call was doing the
spec-prohibited expensive thing by default.

**Fix (stopgap):** added `analyze_conditioning: bool = False` to
`solve_beltrami`, matching pre-GitHub's `analyze_svd` pattern exactly --
gated the `np.linalg.cond` call behind it, logging `nan` when disabled
rather than silently shortening the history list. Verified:
`experiments/run_beltrami.py` never requested it, so no caller needed
updating. Reran at full scale after the fix: **324.0s** -- matching
pre-GitHub's 302.0s within normal run-to-run variance.

**Not the final fix.** This flag is a stopgap that restores correct
default performance now. The real, spec-correct replacement -- SVD for
$P\le3{,}200$, pivoted QR at the final iterate for Beltrami -- belongs to
Phase 1's `iterations.csv` instrumentation work (Codebase_v3_Proposal.md
S2.1), where conditioning logging is being built properly for every
problem anyway. Revisit this flag when that lands; it should likely be
subsumed rather than kept as a separate toggle.

**Process note, worth keeping in mind for the rest of Phase 1:** the
original Q6 diagnosis (below) wasn't wrong that a regression existed, but
it misattributed the *cause* without ever running a controlled, same-
session A/B test -- it reasoned from plausible mechanism (CUDA context
init cost) rather than measuring the actual isolated effect. This entry
exists because re-verifying an old finding before building on it (as
asked) caught that. Worth treating other not-yet-re-verified claims in
this log with the same scrutiny before leaning on them.

---

## 2026-09-22 -- set_seed(): CUDA/cuDNN determinism made opt-in (real but smaller effect than Q6 claimed)

**Phase 1, batch 1.** `set_seed()` used to unconditionally call
`torch.cuda.manual_seed_all()` and force
`cudnn.deterministic=True`/`benchmark=False`, applied identically at
every one of its ~19 call sites regardless of whether that particular
solve path uses CUDA. Q6 (this project's earlier investigation) attributed
Beltrami's full slowdown to this; re-verified this session with a
controlled A/B test and found the real effect is much smaller (~20s of
~540s) -- see the entry above for the actual dominant cause and the
corrected numbers.

The fix stands on its own merits regardless of the corrected magnitude:
confirmed via direct grep that no architecture in this codebase has
Dropout, BatchNorm, or Conv layers (only `nn.Linear` + activation) --
meaning there is no GPU-side random operation and no convolution for
cuDNN to benchmark, so CUDA-level determinism has zero observable effect
on any current result, for a real (if now-modest) cost. Made opt-in via a
new `deterministic_cuda: bool = False` parameter rather than reclassifying
all 19 call sites individually -- since nothing currently needs it, no
call site needed updating, only the shared function's default. `torch.manual_seed`
(CPU-side, covers this codebase's weight init since parameters are
constructed before any `.to(device)` call) is unaffected and still always
set. Pass `deterministic_cuda=True` explicitly if a future addition
introduces GPU-side randomness or convolutions that need it.

---

## 2026-09-22 -- BL's LiL-N gradient was silently wrong (root cause of the plateau above)

**This is a real correctness bug, confirmed and fixed, not a hyperparameter
or config drift like everything else in this log.**

Root cause of the previous entry's open finding: `_make_lil_n_loss_fn`
computed the PDE residual's flux-divergence term as

```python
f_p = physics.flux_derivative(S)   # detached from the autograd graph
f_x = f_p * S_x
```

`BLPhysics.flux_derivative` deliberately detaches its output -- correct
and required for `_make_lil_q_system_fn` (LiL-Q), which is quasilinear and
is *supposed* to freeze this Jacobian coefficient at the current iterate.
It is not correct for `_make_lil_n_loss_fn`: `solve_lil_n` calls
`total.backward()` directly on this loss to get the true nonlinear
gradient w.r.t. `beta`, and the detached path silently drops the
contribution of `f'(S)`'s own dependence on `beta` through `S = A_u @
beta`. The computed "gradient" was missing a term -- not numerically
imprecise, structurally incomplete. L-BFGS, which relies entirely on
accurate gradients to build its quasi-Newton approximation, then
converges to a stationary point of the wrong effective objective, or
simply plateaus.

Confirmed by comparison: the pre-GitHub `BuckleyLeverettPhysics` had two
parallel implementations for exactly this reason -- `flux_derivative`
(detached, for the quasilinear solver) and
`flux_derivative_differentiable` (graph-preserving, for gradient-based
solvers) -- and `solve_nonlinear_lil` (pre-GitHub's LiL-N) correctly used
the `_differentiable` one. The June 2026 consolidation's `BLPhysics` only
kept the detached version, and `_make_lil_n_loss_fn` ended up wired to it.
This also explains why only LiL-N was affected: `_make_lil_q_system_fn`
correctly wants the detached coefficient (unaffected), and NiL-N's
`_compute_pde_residual_nn` computes its derivative a structurally
different way (autograd directly on `f(S(x,t))` w.r.t. `x,t` with
`create_graph=True`, never calling this helper at all -- also unaffected).

**Fix:** added `BLPhysics.flux_derivative_differentiable(S)` -- computed
via `torch.autograd.grad(f.sum(), S, create_graph=True)` on the
*undetached* `S`, rather than a hand-derived closed-form analytic
expression (more robust: it can't drift out of sync with `flux()` the way
two independently-maintained formulas could) -- and switched
`_make_lil_n_loss_fn` to use it. Every other call site of the detached
`flux_derivative` (LiL-Q's system assembly, LiL-Q's convergence-check-only
nonlinear loss, NiL-Q's linearization coefficients) was checked directly
and confirmed to be a genuinely quasilinear/frozen-coefficient context
where the detached version remains correct -- left unchanged.

**Verification, in order of rigor:**
1. `tests/test_bl_lil_n_gradient.py` uses `torch.autograd.gradcheck`
   (numerical finite-difference gradient checking) on the actual loss
   function, both gravity and no-gravity. Passes with the fix.
2. Confirmed the test is meaningful, not a tautology: temporarily reverted
   the one-line fix and reran the same test -- it fails, with the
   analytical and numerical gradients disagreeing by up to ~100 in
   magnitude on individual components (not floating-point noise). Restored
   the fix immediately after confirming.
3. Reran BL viscous and gravity at N=8/16 end-to-end (see
   `Phase0_Empirical_Verification.md` for the before/after numbers) to
   confirm the practical effect on real training, not just the unit gradient.

This was found by taking the "digging into why" request seriously rather
than assuming a config/hyperparameter explanation (everything else in
this log so far) -- worth remembering that not every regression in this
codebase is a caps/tolerances drift; this one was a genuine math bug
hiding behind code that runs without error and produces plausible-looking
(if wrong) numbers.

---

## 2026-09-22 -- Empirically: has the line-search cap ever actually ended training?

Checked directly against every stored `*_summary.json` in both
`reference_results/` (GitHub) and `pre-v2-local-codebase/*/`'s own results,
for every problem, by comparing `total_iterations` against each run's
`max_iterations` and `total_line_searches` against its line-search cap:

- **Bratu, both codebases: no.** Every non-converged run has
  `total_iterations == max_iterations` exactly (the iteration cap bound),
  with `total_line_searches` comfortably below its cap every time (e.g.
  GitHub N=15: 19,818-21,282 evals against a 25,000 cap).
- **Burgers, GitHub (no pre-v2 stored results found to check): no.** Same
  pattern -- every non-converged run hits its iteration cap with evals well
  under the line-search cap (e.g. N=25: ~20,800 evals against a 225,000 cap).
- **Buckley-Leverett LiL-N, GitHub: yes, every time.** Checked
  `reference_results/bl_experiments_fourier/` and
  `bl_gravity_experiments_fourier/` at every N (8/16/24/32): LiL-N's
  `total_iterations` is always well below `max_iterations`, while
  `total_line_searches` sits at exactly `3 * max_iterations` (e.g. N=16:
  iterations=3938, line_searches=30001, against max_iterations=10000) --
  the signature of a cap that bound and cut the run off. NiL-N/NiL-Q at the
  same sizes converge comfortably within the same budget; only LiL-N needs
  enough steps to hit it.

**This traces to something the previous BL line-search entry below didn't
catch: `experiments/run_bl.py` was passing its own explicit
`max_line_searches=MAX_LBFGS_ITERS.get(N, 10000) * 3` at the call site,
which bypasses `BLOptConfig`'s `__post_init__`-derived default entirely**
(the derived value only applies when the field is left `None`; an explicit
value always wins). The `BLOptConfig` fix made in the previous entry was
real and correct, but had **no effect on actual experiment runs** through
`run_bl.py`, since that script never relied on the dataclass default in
the first place. Confirmed pre-GitHub `run_bl_experiments.py` /
`run_bl_gravity_experiments.py` never set anything like this (only
`n_epochs_lbfgs`, a pure iteration count) -- this override is entirely new
to the GitHub consolidation.

**Fix:** removed the explicit override from `run_bl.py`; it now falls
through to `BLOptConfig`'s derived worst case (`max_iterations * 15`),
which is both correct and, per the instruction below, the more generous
choice anyway.

---

## 2026-09-22 -- Bratu's MAX_LINE_SEARCHES: reverted to the higher (pre-GitHub) values

Every entry differs from pre-GitHub, not just N=10's iteration cap:
pre-GitHub `{5:24000, 10:30000, 15:30000}` vs. the GitHub values that had
been in place, `{5:15000, 10:25000, 15:25000}`. Per the empirical check
above, this cap has never actually bound for Bratu in either codebase, so
there's no correctness question here -- reverted to the pre-GitHub
(higher, more generous) numbers on instruction, documented for the record
rather than because evidence favored one value over the other.

---

## 2026-09-22 -- Bratu's N=10 iteration cap: reverted 10,000 -> 7,500

`experiments/run_bratu.py`'s `MAX_ITERATIONS` schedule is per-size, not one
flat number; only the N=10 (P=100) entry changed during the consolidation.
Confirmed against `pre-v2-local-codebase/Bratu/run_bratu_experiments.py`:
`{5: 5000, 10: 7500, 15: 10000}` there vs. `{5: 5000, 10: 10000, 15: 10000}`
before this change. N=5 and N=15 were already unchanged; only N=10 reverted.

**Flag raised here, resolved same day:** while checking this,
`MAX_LINE_SEARCHES` turned out to *also* differ for Bratu, on every entry
-- something Q1's original writeup claimed was "unchanged" (true for
Burgers, checked and confirmed identical in both versions, but evidently
not checked carefully enough for Bratu at the time). See the "Bratu's
MAX_LINE_SEARCHES" entry above (newer, listed first) for the resolution
-- reverted to the pre-GitHub values on instruction.

---

## 2026-09-22 -- Buckley-Leverett's line-search cap: made inert instead of removed

Pre-GitHub BL had no evaluation-based termination condition at all -- a
pure iteration-count loop, `func_eval_counter` tracked for logging only.
The consolidation added `max_line_searches=100,000` as a second
loop-termination condition. Since each `.step()` is separately capped at
`max_eval=15` (hardcoded in `lilq.solvers`, shared by all problems), the
true worst-case eval count for a never-converging run was always
`max_iterations * 15` regardless -- meaning the added 100,000 cap could
actually cut a run short *before* `max_iterations` did in some
configurations, which pre-GitHub BL would never have done.

Rather than hardcoding a literal "big enough" number, `BLOptConfig` now
derives `max_line_searches = max_iterations * 15` in `__post_init__` when
not explicitly overridden -- it can never bind first by construction,
including if `max_iterations` is changed later, and an explicit override
is still honored if anyone wants a genuinely tighter cap for a specific run.

**Addendum, same day:** this fix alone turned out to be insufficient --
see the "Empirically: has the line-search cap ever actually ended
training?" entry above (newer, listed first) for the follow-up fix this
one needed in `experiments/run_bl.py` itself.

---

## 2026-09-22 -- NN/LiL pretraining fit-grid density: reverted to pre-GitHub, per-problem formulas

**What changed.** Two separate fixes:

1. **`pretrain_nn`'s callers** (Bratu/Burgers/BL's NiL-N and NiL-Q pretrain
   calls): were passing a flat `n_grid=50` (2,500 fit points) for every
   problem size. Now each problem computes its own historical value via the
   new `nn_pretrain_grid_side(target_dof, floor)` helper in
   `lilq/pretraining.py`:

   | Problem | floor | target_dof |
   |---|---|---|
   | Bratu | 50 | `N_x * N_y` |
   | Burgers | 100 | `N_x * N_t` |
   | Buckley-Leverett | 50 | `N_x * N_t` |

2. **`pretrain_lil`'s internal formula**: was `n_side = max(n_grid, ceil(sqrt(2*n_basis)))`
   (a structural bug -- comparing a floor meant for a *total point count*
   against an already-square-rooted quantity). Now correctly
   `n_side = sqrt(max(100, 2*n_basis))`, floor=100, matching all three
   problems' identical pre-GitHub formula exactly (this one *was* shared
   across Bratu/Burgers/BL historically, unlike the NN-pretrain floors above).

**Why.** The August-June 2026 consolidation flattened three separate
per-problem pretraining implementations into shared `lilq/pretraining.py`
functions, and lost each problem's own grid-density formula in the process
(Q1, item 1). Verified by reading the pre-GitHub source for all three
problems directly (`pre-v2-local-codebase/{Bratu,Burgers,BL}/*_core.py`).

**What it affects.** The exact starting point of NiL-N/NiL-Q's neural
network (via `pretrain_nn`) and LiL-N/LiL-Q's initial coefficients (via
`pretrain_lil`) before main training begins -- a coarser or finer pretraining
fit changes the initial loss landscape position, which can change downstream
iteration counts even though it never changes the converged answer's
correctness.

**Note on magnitude, since it's counterintuitive:** at every P actually used
in this package's Component B (Bratu <=225, Burgers 625, BL <=1,024), the
*reverted* (historical) grids are **smaller** than the flat 2,500-point grid
that was there before this change -- e.g. Bratu P=25 now fits on a 7x7=49
point grid, not 50x50=2,500. The original Q1 write-up characterized this
backwards ("the GitHub version uses a fixed, smaller grid... for training
speed"); it's actually the fixed grid that was larger at these sizes.

**Revisit later, if:**
- We want one *consistent* formula across all three problems instead of
  three different ones (would deviate from paper-faithful reproduction, but
  is simpler to maintain and reason about).
- Profiling shows pretraining grid density measurably affects final
  convergence quality (not just iteration count) at some P we care about,
  in which case a deliberately denser grid might be worth choosing over
  historical fidelity.
- We want a single shared floor/multiplier instead of two different NN
  floors (50 vs 100) that have no obvious reason to differ beyond "that's
  what each problem's author happened to pick."

---

## 2026-09-22 -- Burgers basis comparison: ELM class corrected to Xavier-scaled

`experiments/run_burgers_basis_comparison.py`'s `'elm'` branch instantiated
`ELMBasis2D` (fixed bound, +/-1.2247 regardless of size) instead of
`ELMBasis2D_Xavier` (bound scales with size, ~+/-0.098 at n_hidden=625).
This is a plain bug, not a judgment call -- the paper's Table 3/4 ELM row
(final loss 5.0e-2) was produced with the Xavier-scaled class (Q8); the
un-Xavier'd class produces a different, much lower loss (~1.7e-5), matching
the repository's own stored (wrong-class) reference result almost exactly.
One-line fix, no future revisit needed.

---

## 2026-09-22 -- BL's LiL-Q off-by-one iteration count: found already fixed, no action taken

Q1 described a real bug in the pre-GitHub Buckley-Leverett code
(`solve_quasilinear_lil` reported `len(metrics.data[...])`, which included a
pre-loop initial-state snapshot, as the iteration count -- one higher than
solves actually performed). Checked the current shared
`lilq.solvers.solve_lil_q` (used identically by Bratu/Burgers/BL) directly:
it counts via `n_quasi_iters = quasi_iter + 1` set inside the loop, which
does **not** have this bug. The June 2026 consolidation fixed it as a side
effect of merging three separate implementations into one correct shared
one. No code change made; a regression test
(`tests/test_solve_lil_q_iteration_count.py`) locks in the correct behavior
so it can't silently regress in a future refactor.

---

## 2026-09-22 -- BL's `tolerance_grad`: kept at 1e-8 (no revert)

Pre-GitHub BL's L-BFGS calls never set `tolerance_grad` explicitly, so they
silently ran on PyTorch's own default (1e-7). The current shared solver
explicitly sets 1e-8 for every problem including BL. Considered reverting
BL specifically back to the looser implicit value, but: Bratu's and
Burgers' pre-GitHub *main-solve* optimizers already explicitly set
`tolerance_grad=1e-8` (identical to today's value) -- only their
*pretraining* step used a tighter 1e-9. So 1e-8 was already the deliberate
standard for two of three problems before the consolidation; BL's 1e-7
looks like an omission, not a considered choice. Decision: keep 1e-8
uniformly. No code change was needed (current code already does this);
this entry exists purely to record the decision was made deliberately,
not left as an unexplained side effect of the consolidation.

---

## 2026-09-22 -- Darcy's float32 neural-net path: left as-is, confirmed out of scope

`problems/darcy.py`'s `DarcyPINN` class force-casts its three MLPs to
float32 (`lilq/solvers.py` sets float64 as the process-wide default for
every other problem). Checked which code path Computational_Package_1_v2.md
actually requires for Darcy (Section 3.3): `solve_lilq_darcy`, which uses
basis objects (float64, confirmed) -- not `DarcyPINN` at all. Nothing in the
current package calls `run_nil_n_darcy`/`DarcyPINN`. Decision: leave it
alone; revisit only if a future package brings Darcy's NN comparison into
scope, at which point the float64-everywhere requirement (Section 2 of the
package spec) would need to be satisfied there too.

---

## 2026-09-22 -- Bratu's NiL-Q inner-iteration cap: second, separate N=10 revert (found via a real run)

Caught while sanity-checking a fresh v3-dev run against expectations: with
the N=10 `MAX_ITERATIONS` fix already in place, NiL-N correctly stopped at
7,500 iterations, but **NiL-Q ran to 10,000** -- the fix hadn't fully taken.

Root cause: NiL-Q's actual iteration budget is `MAX_QUASI_ITERS` (flat, 25
in both codebases) times `MAX_LBFGS_PER_QUASI_ITER[N]` (a wholly separate
per-size dict from `MAX_ITERATIONS`, which only governs NiL-N/LiL-N's flat
loop). This second dict also changed at N=10 during the consolidation:
pre-GitHub `{5:300, 10:300, 15:400}` vs. the GitHub value that had been in
place, `{5:300, 10:400, 15:400}` -- worst-case NiL-Q totals of 7,500 vs.
10,000 at N=10, exactly matching what the fresh run showed. Confirmed
Burgers' and both Buckley-Leverett variants' equivalent per-quasi-iter
schedules are genuinely unchanged between codebases (checked directly),
so this second-knob issue is isolated to Bratu N=10, same as the first.

Reverted `MAX_LBFGS_PER_QUASI_ITER[10]`: 400 -> 300. This is the kind of
thing the empirical run-and-compare pass this entry belongs to exists to
catch -- two independently-named constants that both nominally describe
"Bratu's N=10 iteration cap" but govern different methods, easy to fix one
and miss the other from source-reading alone.

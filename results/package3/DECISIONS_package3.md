# DECISIONS.md: the Package 3 entries

Copied from the repository's `DECISIONS.md` at the assembly's commit (3e13572), newest first.

---

## 2026-10-09 -- Package 3, the assembly: Grace's tree, L1 at this commit, the Section 9 records

`experiments/p3_assemble.py` builds `package3_results/` from the last report's tarball. That
tarball is wave 1 and B3 level 2: 11 jobs, all COMPLETED.
- **What is Grace's.** Every run and every summary is Grace's, at the runs' commit (97e463d,
  under the source lock), except L1. L1 reads only the logged CSVs, and is re-evaluated at the
  assembly with the corrected comparison. Grace's verdict is kept beside the new one in
  `checks_item4.json`.
- **`checks.json`** gathers K0-K8, item 3's all-digit re-solves, and L1-L4, each with its
  source file.
- **`su.csv`** is the report's `su_per_job.csv` plus the last report job's own row. That row
  comes from `sacct`, since a job's tarball cannot hold its own charge.
- **`hardware.json` and `environment.txt`** come from the jobs' own logs (the node, the
  threads, the versions, the preflight) and `sacct.txt`. The Package 3 jobs did not call
  `save_provenance`, so the CPU model is not in their logs. `hardware.json` says so, and cites
  Package 2's record of a node in the same partition (Xeon Gold 6248R, 48 cores).
- **The laptop's batch 0 K1 and K2** move to `laptop_preview/P3_checks/`. `P3_checks/` is now
  Grace's.

Tests: `tests/test_p3_assemble.py`.

---

## 2026-10-09 -- Package 3, after wave 1: check L1 compares a terminal row like with like

Wave 1 (bundle 97e463d) reported L1 failed, on one value. P2-12's k = 4 `norm_R_h` was
5.91211765423539e-05 against the control's 5.9121176541457504e-05: 9e-16 absolute, round-off
for a residual whose entries are O(1).
- **Every other value at k <= 4 was identical.** That covers norm_R_h and norm_Rlin_h at
  k = 0-3, and eps_ref at k = 0-4.
- **Why k = 4 differed.** P2-12's pass stopped there at the paper's target, so its k = 4 row is
  `LilQDiagnosticsTracker.finish`'s, and its norm_R_h is sqrt(the solver's loss). The control
  goes on to K_max = 60, so its k = 4 row is a solve row, and its norm_R_h is
  ||A^(4) beta^(4) - f^(4)||. The two are algebraically equal, but computed differently.
- **The control's sqrt(loss at iterate 4)** (`losses.csv`) is 5.91211765423539e-05: P2-12's,
  bit for bit.

**The correction (one line):** on a P2-12 terminal row, L1 compares norm_R_h with
sqrt(the control's loss at that iterate). It reads only the logged CSVs. So L1 is re-evaluated
at the assembly on Grace's files, and passes; no rerun is needed.

Also: `submit_p3.sh`'s closing hint now says to run wave 2 when every wave-1 job but the report
has finished, which is what the gate requires. It used to name only B3 level 1 and K0.

---

## 2026-10-09 -- Package 3, batch 4: the Grace submission, checks K0 and K8, B3 level 2's gate

`scripts/cluster/package3/`, `experiments/p3_k0_k8.py`; the p3 stage in `env.sh`, `sbatch.sh`,
`submit_lib.sh` and `package2/stage_report.py` (`results/package3`, one commit, one lock).

**K0** (Section 7). A random 600,000 x 7,984 matrix (4.79e9 entries, above 2^32; 38 GB), with
a consistent right-hand side.
- **The two solvers:** gelsy, and scipy's column-pivoted QR (`qr_multiply`, then the
  triangular solve).
- **Pass:** relative residual and relative error both at most 1e-12.
- **The laptop smoke run,** at a small size, passed at 1.9e-15.

**K8.** It reruns two of the paper's runs:
- the pinned Beltrami run, `solve_beltrami(pinned_config())`;
- the viscous BL run at P = 576, `run_bl.paper_setup(24, False)`.

It compares their iteration counts, and `norm_R_h` and `norm_Rlin_h` as printed strings, with
`package1`'s, at package1's 48 threads. K8 is Grace's: off Grace the digits differ, since the
BL system is rank-deficient.

**The jobs** (`_generate_jobs.py` writes all ten; `su_plan.csv` is unchanged).
- **Each Beltrami run runs under the advisor's own cap** (`timeout`):
  - B1 1 h; B2 2 h;
  - B3 level 1 235 min (inside its 4 h walltime); level 2 415 min (inside 7 h).

  A run stopped by its cap keeps its logged iterates, and its `run.json` status stays
  `running` (Section 8.2). The job goes on to its next run.
- **K5 and K6 of item 2 share item 2's pool** (`--with-checks`). The memory is small: the
  largest B_h is 20,480 x 1,025 (168 MB) and K5's B_Y 36,864 x 1,025 (302 MB). Six workers
  use a few GB of the `cpu` class's 32 GB.
- **The preflight checks all 25 input files** the jobs read before any compute job starts:
  - package1: K8's two runs, the four Darcy fields, the five elasticity sizes;
  - P2-10: compatible and specified, five sizes each;
  - the BL and Burgers references;
  - P2-12's P = 625 run.

  Item 3 would otherwise skip a missing logged residual silently. Run against the downloaded
  Grace trees (symlinked into Grace's layout), all 25 were found, and a missing one stopped it.
- **Job names** are `lilq-p3-<name>`, in lower case. The report and the gate find the logs by
  them.
- **Item 1's summary runs in the report job,** before the tarball. No Beltrami job summarized,
  since B1, B2 and B3 run in separate jobs.
  - It covers every Beltrami run so far, and runs again after B3 level 2.
  - It takes about 3 min and 6.7 GB on the laptop, mostly B3's delta_P.
  - Each report is now about 2 SU (`su_plan.csv`: wave 1 about 376 expected, all about 617).
  - If it fails, the tarball is still packed.

**Runs that stop or fail.** Grace runs everything once, so one run's failure must not cost
the others, and every run stopped by its cap must still be reported.
- **Item 1's summary of a run stopped by its cap** (Section 8.2). It reads the run's ranks from
  its log, since `run.json` records them only at the end. Before this fix it read rank 0, a
  false K7 failure, and K7 is a stop condition.
  - It counts only the iterates whose errors were written. The cap can strike between an
    iterate's log row and its errors.
  - A run with no `errors.csv` yet is not summarized.
- **Item 2.** K5, as long as the largest runs, now starts first. It had been queued after all
  48 runs, which could have ended the job near 112 min of its 120 at the laptop's pace. K6
  (P = 256) goes last.
  - **A run that fails** puts its own error (its stderr) into the job's log, and the other runs
    go on.
  - **The summary is still written,** and the job then exits non-zero naming the failures.
  - **A run that did not finish** is listed in `checks_item2.json` (its
    `constants_by_iterate.csv` is written at every iterate). It is never summarized or
    compared.
- **Item 3,** likewise: a configuration that fails shows its error, the others go on, and the
  summary is written.

**B3 level 2's gate** (`b3l2_gate.py`, run by `submit_p3_b3l2.sh`). It writes nothing. The
order:
1. **Already done (exit 1):** level 2 has a log, or is queued.
2. **Not yet (exit 2):**
   - `squeue` cannot be read;
   - any Package 3 job other than the report is still queued;
   - no Package 3 job has a log yet.
3. **K0 (exit 1 if missing or failed).** A failed K0 is a stop condition (Section 8.1).
4. **B3 level 1 (exit 1 if missing, or if its status is not `complete`;** Section 8.2).
5. **The launch rule** (Section 3.4).
6. **The budget.**
   - **The charge** is from `sacct` over every Package 3 job with a log, at Grace's rates
     (`wave_report.su_rows`). A job cancelled before it started has no log and no charge.
   - **Exit 2** if `sacct` cannot be read, lacks a job, or still shows one running.
   - **The guard:** the charge plus 336 is at most 850. B3 level 2's run stops at 6 h 55 min,
     so with a minute or two to start, its job charges at most about 334 SU. That leaves room
     for the report after it (about 1 SU).

A first draft took the charge from `wave_report.su_per_job`. That function writes
`su_per_job.csv`, and returns no rows when `sacct` fails, which reads as 0 SU charged and
passes the guard. The draft also found pending jobs only by their logs, which a pending job
does not have yet. Both are fixed here.

**Line endings.** Python's `write_text` on Windows writes CRLF. Seven files committed in
batches 0-3 had CRLF blobs:
- `DECISIONS.md`;
- the four `experiments/p3_*.py`;
- `lilq/certified.py`;
- `tests/test_p3_affine_certificates.py`.

They are LF from this commit. Nothing that ran changes, for two reasons:
- **The bundle** writes every text file with LF (`source_lock.normalized_bytes`).
- **The source lock's tree hash** is over the same normalized bytes.

Many older files are CRLF in this working tree only (checked out under the global
`core.autocrlf=true`; their blobs are LF). So the bash tests run on the extracted bundle,
never on a copy of the working tree.

Tests: `tests/test_p3_submission.py`, 29 in all:
- the scripts as generated, LF, their classes, caps and inputs, and the SU plan's 828 and
  1,164;
- the stage registered, and the report's tarball;
- the gate in 17 situations, and how it reads `squeue` and `sacct`;
- six bash tests: wave 1's dry run and chain, and wave 2 submitting, refusing, waiting, and
  its dry run.

Five more tests for the stopped and failed runs:
- item 1's summary of a run stopped by its cap;
- item 2's order, a failure, a failing run's real error, and its unfinished runs;
- item 3's failure.

---

## 2026-10-08 -- Package 3, Addendum 1: item 4, Burgers LiL-Q at larger sizes; the B2 convention

The advisor's addendum of 8 October adds item 4, and leaves the rest of Package 3 as
written.
- **Why.** LM networks reach a Burgers error of 1.6e-5 at P = 625, below LiL-Q's 3.7e-5 at
  the paper's sizes. Methods are compared at matched error, so LiL-Q's cost to reach 1.6e-5
  is needed.
- **The cap** rises from 800 to 850 SU, with 50 for item 4.

`experiments/p3_4_burgers_large_P.py`.

**The logged pass** is `problems.burgers.run_lil_q` (P2-12's driver) with
`run_burgers.paper_setup(N)`, unchanged but for the size. It has R_tol = 0 (for N > 25
`paper_setup` would default it to 1e-4) and K_max = 60, with the Cole-Hopf reference. The
sizes are N = 30, 32 and 35, with the P = 625 control; 40 runs only if no iterate at the
three reaches 1.6e-5.
- **kappa** comes from the tracker's own rule for P <= 3,200: SVD at every iterate, the last
  included. That is more than the addendum's "at the last iterate", and it changes nothing
  in the solve.
- **`iterations.csv`** is written by the unchanged driver at the end of the pass, not after
  every iterate. A pass takes minutes against a 1 h cap.
- **`losses.csv`** is the solver's own record (`QuasilinearMetrics.total_loss`): entry k is
  the loss at iterate k.

**Clean timing** (the paper's protocol, Package 1 wave 4, option B). For each size and
stopping iterate there is one untimed warm-up, then three clean runs (`diagnostics=False`),
reported as their median.
- **k_e:** the first iterate with error <= 1.6e-5.
- **k_r:** the iterate the termination rule returns.
- **The P = 625 control** is timed as the paper pass itself: its target, and max_iter 4.

**Checks:**
- **L1:** compares norm_R_h, norm_Rlin_h and eps_ref for k <= 4 with P2-12's
  `iterations.csv`, as floats. P2-12's terminal row k = 4 has no norm_Rlin_h; its pass stopped
  there at the paper's target.
- **L2:** the control's median against the paper's 1.18 s, within 15%.
- **L4:** every clean run's final loss equals the logged pass's at the same iterate, bit for
  bit, and every clean run takes exactly max_iter iterations.

**The B2 identity: Package 1's convention** (K4 of items 1 and 2, L3 of item 4). The
identity is judged at k = 1, below 1e-10, and the run maximum is reported, not tested.
- **The rule** is Package 1's (the entry "B2 in real runs" below). Near convergence R is a
  small difference of O(1) quantities, so cancellation dominates the relative difference.
  Package 1's K_max runs reached maxima of 5.6e-2 (BL P = 1,024) and 1.4e-2 (Kovasznay
  P = 1,875).
- **Why it matters here.** Item 4's passes run 60 iterates with no target, far past
  convergence: the maximum reaches 2.7e-6 at P = 1,225, while k = 1 is 4.0e-15.
- **The change to items 1 and 2.** Batches 1 and 2 had tested the run maximum. It passed on
  the laptop, but on Grace a converged iterate could have tripped a false K4 failure, which is
  a stop condition.

**Budget** (`package3_results/su_plan.csv`):
- item 4 is one exclusive-node job, 1 h (48 SU requested, about 12 expected);
- wave 1 is 828 SU if every job ran to its cap, under 850;
- B3 level 2's budget guard becomes: SU charged so far + 336 <= 850.

**Laptop rehearsal** (24 threads, 522 s with the timings):

| P | smallest error (k) | k_e | k_r (class) | to k_e (s) | to k_r (s) |
|---|---|---|---|---|---|
| 625 | 3.67e-5 (3), as the paper | none | 6 (A) | -- | 1.48 |
| 900 | 1.11e-6 (4) | 3 | 6 (C) | 1.68 | 3.41 |
| 1,024 | 4.36e-7 (47) | 3 | 6 (C) | 2.42 | 4.84 |
| 1,225 | 3.52e-7 (28) | 3 | 6 (C) | 3.72 | 7.27 |

- **Every size from 900 up reaches 1.6e-5 by iterate 3,** so 40 x 40 does not run.
- **L3 and L4 passed.**
- **L1 differs off Grace:** about 1e-14 relative in the norms, 1e-8 in the 7e-10 linear
  residual. L1 is Grace's.
- **L2:** the laptop's 0.97 s; it is Grace's too.

---

## 2026-10-08 -- Package 3, batch 3: item 3, a posteriori certificates for elasticity and Darcy

`experiments/p3_3_affine_certificates.py`.

**The re-solve is the paper's own run.** `solve_elasticity` and `solve_lilq_darcy` run
unchanged, and the system they pass to `scipy.linalg.lstsq` is captured with its solution
(`captured_solve`). So A_h, b_h and beta_h are the paper's run itself, and its residual is
compared with the logged k = 0 `norm_Rlin_h`:
- **Against Package 1** (the paper's elasticity, and Darcy), wave 2 at 48 threads.
- **Against P2-10** (`compatible`, `specified`), at 24 threads.
- **Both stacks are the same:** numpy 1.26.4, scipy 1.13.1, FlexiBLAS 3.4.4.
- **Each configuration runs in its own process** at its original thread count (`THREADS`),
  so the Grace re-solves can match to all digits. `--threads` overrides it on a laptop.

**The Y-system is built by row builders that follow the paper's code operation for
operation.** At the paper's own points they reproduce the captured system bit for bit, in
all 19 configurations (`assembly_matches_paper_bitwise`).

**Decisions within the instructions:**
- **Darcy's blocks** are the instructions' reading: Darcy-x, Darcy-y and continuity at the
  13,200 cell centres, weight 1, no boundary rows. The sqrt(K*) factors are part of each
  row's operator, not a row weight.
- **Zero rows (Section 6).** A block is left out of both systems when its rows are zero to
  round-off (|A| <= 1e-10 max|A_h|) and its data are zero.
  - In elasticity these are the u_x and u_y Dirichlet blocks on edges where the basis
    vanishes (bottom u_x, top u_x, left u_y, right u_y), and the lateral sigma_xx blocks
    where the lateral traction is zero (paper, compatible).
  - `specified` keeps its lateral sigma_xx blocks: they are zero in A but not in b, P2-10's
    inconsistent rows.
  - Single zero rows inside a kept block (the x = 0 corners) are kept; they add nothing to
    any norm.
- **The Y-norm, block-matched.** Each block carries its total squared weight in the
  collocation set:
  - elasticity: lambda per block, 96 x 96 Gauss-Legendre inside and 96 per edge;
  - Darcy: 4 x 4 Gauss-Legendre per cell with that cell's K*, each cell's weights summing to
    its centre row's 1.
- **Section 6 from the R factor.** B_Y = [A_Y | b_Y] enters only through its R factor,
  accumulated a chunk of cells at a time (`lilq.certified.tsqr`), because each quantity used
  is invariant under B_Y's orthogonal factor:
  - the column norms;
  - the column-pivoted QR of the unit-scaled B_Y (B_Y D P = Q Q' R' when R D P = Q' R');
  - the least-squares minimum and the residual at beta_h;
  - kappa(A_Y), from R's leading block.

  Darcy's B_Y is 633,600 x 3,170 (16 GB) at n_q = 4 and 1.43 M x 3,170 at n_q = 6 (K5); it
  is never held whole. Checks:
  - against the dense steps (item 2's `section6_constants`) on elasticity P = 200, it agrees
    to 1e-10;
  - with one Gauss point per cell (the centre, weight 1) the Y-system is the collocation
    system, and it gives c1 = c2 = rho_r = 1 (`tests/test_p3_affine_certificates.py`).

**Laptop rehearsal** (24 threads; 22 min with K5):
- **The certificates:**
  - elasticity: c2/c1 = 1.59, 1.75, 2.32, 3.00 and 4.14 at P = 50 .. 1,250; the three
    solutions agree to 3 digits (P = 50: 1.5891, and 1.5894 for `specified`);
  - Darcy: c2/c1 = 4.67 (S1), 5.48 (S2), 3.08 (S3) and 1.82 (SPE10); on span{A} alone S1 is
    1.70;
  - rho_r between 1.0003 and 1.024.
- **Where f is numerically in the span,** its column is dropped and rho_r is flagged
  round-off, as the instructions anticipate. That covers the paper's elasticity at every P,
  and `compatible` at P = 1,250.
- **K5 passed:** Darcy SPE10 at n_q = 6 changes c1 by 1.7e-7 and rho_r by 7e-11; elasticity
  at 144 points changes c1 and c2 by 3e-13.
- **The re-solves (laptop, so not to all digits):** `specified` matches P2-10 exactly at
  P = 50, 200, 450 and 1,250. `compatible` differs by up to 1.5e-5 relative where its
  residual is 8e-12. The paper's elasticity residuals are pure round-off (about 1e-14), so
  off Grace they do not match at all. Darcy differs by 5e-16 to 2e-14. The all-digit check
  is Grace's, at each run's thread count.

---

## 2026-10-08 -- Package 3, batch 2: item 2, Buckley-Leverett on CC-CGL grids (2a and 2b)

`experiments/p3_2_bl_certified.py`. It uses the paper's two cases and its flux
derivatives (`BLPhysics`, autograd) and quasilinearization, on `lilq.certified.bl_grid`,
with a weight per row.
- **2a's trial space** is T_i(2x - 1) T_j(2t/T - 1).
- **2b's** is (1 - x) + sin(i pi x) T_j(2t/T - 1). The lifting's terms go to the
  right-hand side.

**The assembly is the paper's.** At the paper's random grid and per-block weights,
`Rows.assemble` reproduces `buckley_leverett._make_lil_q_system_fn`, for both cases and both
spaces:
- A bit for bit;
- b to 1e-15 for 2b, which carries P2-9's lifting column with coefficient 1;
- the Newton identity to 4e-16.

**Decisions within the instructions:**
- **2b's lateral rows.** They are evaluated numerically, sin(i pi) about 1e-16, as P2-9,
  and kept in the solve. They are the zero rows that Section 6 leaves out of both systems.
- **The Y-norm.** Each block of the Gauss-Legendre quadrature carries that block's total
  squared weight in the collocation set, the lateral lines' 1 - w_0 included.
- **The Section 6 constants:**
  - B_Y's columns are scaled to unit norm, and the same scaling is applied to B_h, with a
    guard for a zero column.
  - rho_r's denominator is a `gelsy` least squares.
  - The round-off flag's kappa(A_Y) is sigma_1 / sigma_r over the numerical rank
    (max(m, n) eps sigma_1).
- **kappa of the collocation system** is computed at the last iterate only, by SVD
  (P < 3,200), and the A/C class uses it, as in item 1.
- **"Does not stall by k = 60, or oscillates" (Section 8.2)** is read as: the termination
  rule never fires within the 60 iterates. An oscillating run never meets the stall
  conditions, so the one test covers both. Such a run is run once more from the other guess
  (viscous: zero; gravity: the extended profile; for 2b the lifting is subtracted), and both
  are reported.
- **"At the rule's iterate"** in `terminal.csv` means the Section 6 row linearized at the
  returned iterate beta^(k_rule); it is the last row when the rule never fires. The loss
  target compares the CC-weighted ||R(beta^(k))||_h^2 at each k.
- **`lilq.certified.block_constants`:** a datum that is zero on its block (S = 0 on the right
  line) adds nothing to the span, so it is not a column. Otherwise its zero norm would
  divide.
- **The runs go side by side,** each in its own process with its BLAS thread count fixed
  before numpy loads. A run's numbers then depend on its thread count only, not on what runs
  beside it. K5 and K6 run at the sweep's thread count, so they compare like with like.

**Laptop, single runs:**
- P = 64: 11 s each.
- P = 1,024 at N/P = 20 (viscous, 2a): 755 s, of which 619 s are the Section 6 constants,
  computed at every iterate. It converges by k = 10, with eps_ref 1.1e-3, constants
  1.0000, and rho_r = 1.

**Laptop, the whole sweep** (`all --workers 6 --threads 4`, 8 October): the 48 runs took
67 min; then K5 and K6 took 49 min, K5 being most of it. Results:
- **No Section 8.2 rerun:** the rule fired in every run, all class A.
- **K4:** 1.3e-12 at most. K7 passed.
- **K5:** c1, c2 and rho_r change by 2.4e-13 at most, against 1e-3.
- **K6:** bit for bit.
- **eps_ref / delta_P at the rule's iterate:**
  - 2a viscous: 11-66;
  - 2a gravity: 5-22;
  - 2b viscous: 1.9 at P = 1,024, with eps_ref 9.4e-5 (P2-9's plain sine on the paper's
    grid: 6.9e-5);
  - 2b gravity: 6-28.
- **The Section 6 constants:** c2/c1 is near 1 from N/P = 10 up. The worst are 2b gravity at
  N/P = 5: up to 7.9 at the rule's iterate and 10.8 over a run, with rho_r up to 2.3. They
  are reported as they are (Section 8.2). The smallest c1 over every iterate is 0.163, far
  above Section 8.2's 1e-3.
- **On Grace,** K5 and K6 run alongside the sweep, not after it (batch 4).

**Tests** (`tests/test_p3_bl_certified.py`):
- the assembly against the paper's builder in all four combinations;
- the Newton identity;
- the Y-norm's block totals;
- the Section 6 constants on known cases (equal norms, a dependent column, a halved norm);
- 2b's zero rows;
- 2a's a priori constants (1.0212 at P = 64, N/P = 10; the S = 1 datum dropped; C6 at
  N/P = 5);
- tiny runs end to end (K4, K7, a K6 rerun kept out of the results);
- the other-guess scheduling.

---

## 2026-10-08 -- Package 3, batch 1: item 1, Beltrami on CC-CGL grids

**A separate solver path, not a branch in `solve_beltrami`.**
`experiments/p3_1_beltrami_certified.py` reuses the paper's physics, bases, `gelsy` solve
and error metric. Its own `System` assembles the same quasilinearization with a square-root
weight per row. `problems/beltrami.py` is untouched, so the paper's run cannot change; K8
still runs on Grace (batch 4).
- **The assembly is the paper's linearization.** Given the paper's equispaced points and
  per-block weights, `System.assemble` reproduces `solve_beltrami`'s own A and b on a small
  configuration (N_vel = 3, N_p = 4, 499 coefficients): bit for bit at k = 0, and to 1e-14
  at k = 1.
- **Memory** (Section 3.4, optional): u, v and w share one set of basis matrices, and A is
  preallocated, zeroed and refilled every iteration, with no list of row blocks.

**Decisions within the instructions:**
- **Pins.** The pins keep the paper's squared weight lambda_bc / N_p. K3 multiplies that
  squared weight by 1e-3 and 1e3, reading "pin weight" in Section 2.2's squared-weight
  convention.
- **kappa.** It is computed at the last iterate only (Section 2.4): by SVD for P < 3,200,
  otherwise by the paper's pivoted QR. The tracker's own conditioning is switched off. The
  tracker is not given `n_interior_rows` or `interior_weight`, as Section 2.4 says. The kappa
  time is recorded for B3 level 2's launch rule.
- **The termination rule's class (A/C)** uses that last-iterate kappa, since no kappa exists
  at the rule's own iterate. Class C means the round-off ratio at the rule's iterate is
  below 10 x kappa_retained x eps (`stopping_rule_table.classify`). By then the iteration has
  stalled, so kappa barely moves.
- **"The returned iterate"** in `terminal.csv` is the first iterate whose coefficient change
  is below 1e-9, the paper's criterion; the last iterate if none is. The combined error is
  also given at the rule's iterate and the last one, and `errors.csv` has every iterate (u,
  v, w, p shifted per level, the pin gauge, t = 1, combined).
- **rho_r (B1).** The Y-system has no pins, so its rank is P - N_p (the unpinned gauge modes).
  The denominator is a `gelsy` least squares. The round-off flag's kappa(A_Y) is therefore
  kappa_retained, sigma_1 / sigma_r over the numerical rank.
- **delta_P** is a dense `gelsy` least squares on the 21^3 x 11 grid. For p, 11 columns
  constant on each time level are added (Section 3.5).
- **K6.** `iterations.csv` is compared in every column except the three timings, which no
  rerun reproduces. `errors.csv` and every `beta_<k>.npy` are compared bit for bit.
- **A stopped run.** `run.json` is written at the start with status `running`, then
  completed. `iterations.csv`, `errors.csv`, `rho_r.csv` and `beta_<k>.npy` are written after
  every iterate (Section 2.1).

**Tests** (`tests/test_p3_beltrami_certified.py`):
- the assembly against the paper's solver;
- the Newton identity on a CC grid;
- B1 level 1's 160,386 x 1,393, and the pin weight;
- delta_P's level columns, a pressure varying only in time coming out at 1e-13;
- a run stopped at its second solve keeping its first iterate;
- tiny runs end to end: K3 (2e-15 against 1e-6), K4, K6 and K7 pass.

---

## 2026-10-08 -- Package 3, batch 0: the CC-CGL grids, the constants, checks K1 and K2, the SU plan

Package 3 is the advisor's instructions of 8 October 2026: runs inside the hypotheses of
Section 5.6. Beltrami (item 1) and Buckley-Leverett (items 2a, 2b) run on tensor CGL
grids with Clenshaw-Curtis (CC) weights; elasticity and Darcy (item 3) get a posteriori
constants on the paper's own sets. Cap: 800 Grace SU. There is one report at the end;
anything that is not a stop condition of its Section 8.1 follows the rules of 8.2 and is
recorded here.

**`lilq/certified.py` (new; nothing existing changed):**
- **The grids of items 1 and 2a** (Section 2.2):
  - `beltrami_grid(M)`: the M^4 interior, six faces with M^3 in their free coordinates,
    and the initial slab with M^3. The shares are 4/40 and 8/40; |dOmega| = 40 counts both
    slabs.
  - `bl_grid(P, r, T)`: the P2-16 Burgers rule on [0, 1] x [0, T], with |dOmega| =
    2 (1 + T).
  - A block carries its CC weights (sum 1) and its share. The squared row weight is
    lambda x share x w.
- **The interior constants:**
  - `interior_constants_kronecker`: the advisor's Kronecker shortcut on the repository's
    own nodes and weights (B10's `points_1d(..., 'cgl')` and `clenshaw_curtis_weights`);
  - `interior_constants_direct`: the SVD of the weighted evaluation matrix.
- **`block_constants`:** the advisor's routine (Section 2.3(b)), extended from lines to 3D
  faces and slabs. It uses tensor Gauss-Legendre with max(4M, 32) points per direction on
  faces, and max(4n, 400) on lines.
- **`cc_exactness`:** C4 of P2-16 at any dimension. The test polynomials are sums of
  products of random Legendre series, so the exact values come from the coefficients, and
  the check runs on each rule's own flattened points and weights.

**Check K1 passed** (`experiments/p3_checks.py k1`, run on the laptop with the script):
- **Printed digits.** All 30 printed values of Sections 3.3 and 4.2 match. So do the
  script's P2-16 check and its four Burgers lines.
- **Against the script's own functions:** the largest difference is 9e-14.
- **Direct against Kronecker:** they agree to 1.6e-15 at Beltrami B1 level 1
  (28,561 x 6,561), and to 1.3e-15 for BL P = 64, N/P = 10, in both cases. The tolerance
  is 1e-10.
- **The Burgers initial line:** the datum -sin(pi x) is dropped as numerically in the trace
  space (|R_ii| < 1e-10 |R_11|), as the advisor anticipated.

**Check K2 passed** (`p3_checks.py k2`): 144 rules, worst error 3.8e-14 against the
tolerance of 1e-12. They are:
- Beltrami: 6 grids x (interior, 6 faces, slab);
- BL: 2 cases x 12 grids x (interior, initial, two lateral lines).

**Decisions:**
- **The lateral lines of BL keep P2-16's weights.** A lateral line is the (n + 1)-point rule
  on [0, T] without t = 0. As in P2-16, its weights are kept as they are, summing to
  1 - w_0, not renormalized. Section 2.2 says to follow P2-16, and the difference is
  dropped endpoint weight w_0: 0.79% at P = 64, N/P = 5, and 0.2% or less on every other
  grid. The block constants normalize them
  anyway (Section 2.3(b)). K2 checks the rule the line is cut from, and records the line's
  own sum.
- **The interior lambda of BL is 1,** as P2-16's Burgers. Section 4.1 gives only the
  auxiliary lambda = 10.
- **The budget: by construction, never over 800 SU** (`package3_results/su_plan.csv`). The
  advisor's per-run caps for item 1 alone add up to 816 SU, and the expected total is 583.
  - Wave 1 has every job except B3 level 2. Its walltimes sum to 780 SU: K0 is capped at
    2 h (it needs about 1.25 h) and item 3 at 1.5 h (about 1 h).
  - B3 level 2 (7 h, 336 SU) is launched only if both the advisor's launch rule (Section
    3.4) and a budget guard hold. The guard: the SU charged so far plus 336 is at most 800.
  - So the charged total cannot pass 800, which would be a stop condition (Section 8.1).
  - In a job holding several runs, each run gets its own timeout at the advisor's cap.

Tests: `tests/test_p3_certified.py`:
- the printed values, P2-16's constants, and direct against Kronecker;
- the Burgers lines, and a face with and without a datum;
- the grids' rows and shares (B1 level 1: 160,386 rows; B3: 348,168 and 665,331, as
  Section 3.4);
- K2 failing on wrong weights and beyond its degree, and K2 on every grid.

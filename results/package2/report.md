# Computational Package 2: Stage 2 report

**To:** R. Younis. **From:** G. Awojinrin. **Date:** 6 October 2026.
**Refers to:** the reply to the pre-submission note and the reply before submission (both
5 October).

**Code:** `awojinrin/lilq-pinn`, branch `v3-dev` at 7af8855: the 16 commits after b2b99de,
the commit of the pre-submission note (table below). `DECISIONS.md` has one entry per step,
with the details behind everything here.

**Results:** `package2_results/` (Section 12.1 layout), on the branch `package2-results` of
`awojinrin/lilq-pinn`, under `results/package2/`. The branch starts at 7af8855; its README
says what it holds.

## Summary

Stage 2 is done, and every acceptance check passes. It cost about 686 SU: about 505 on
Grace and 181 on FASTER.

**What to know before reading:**
1. **One departure from the reply: item 5's lifting.** It uses the paper's output gain,
   h* = y* + omega NN_h / 2, not h* = y* + omega NN_h. This keeps the run different from the
   paper's in your two things only, and keeps item 5 against 5b a test of the boundary
   treatment alone (item 5).
2. **Item 7's A100 series was run twice.** The first GPU job failed at Kovasznay P = 3,675
   because the cluster environment loaded a mix of two CUDA library versions. The
   environment was fixed, and the whole A100 series reran as a small addendum (e5eb447). C8'
   passes in both environments, and no other result used the affected routine.
3. **Beltrami at 22,288 coefficients does not fit on the 40 GB A100.** The limit is
   cuSOLVER's workspace for the Q^T b step. The largest that runs is 13,764 (item 7).
4. **The classical baselines beat LiL-Q on Bratu and Kovasznay:** the same error with fewer
   coefficients, in 13-44 times less time. On Burgers, LiL-Q is more accurate (item 1).
5. **Darcy:** with LM, the boundary treatment changes little; the optimizer accounts for
   almost all of the gap to LiL on S1-S3. SPE10 is still well short of LiL (item 5).

**For your reply:** the four questions of Section 5. They concern the framing of items 1,
5, 6b and 7 in the manuscript.

**Provenance.** `package2_results` was assembled at 7af8855 from three cluster tarballs:
- **The Grace stage and FASTER's item 5** are at commit 4f327f7 (`package2_stage2.tar.gz`,
  `package2_stage2_faster.tar.gz`).
- **Item 7's A100 series** is from the GPU addendum, at e5eb447 (Grace jobs 20030355-57,
  `package2_stage2_gpu.tar.gz`).
- **e5eb447 against 4f327f7:** for the runs, it changes only the jobs' library path, the
  record of the libraries loaded, and the preflight's library probe.
- **The assembly ran after the runs, with `--later-commit`.** 7af8855 against e5eb447 adds
  the `did_not_fit` record of Beltrami 22,288 and fixes the assembly script; nothing it
  changed is run code.
- **The changed files are listed in `code/`:** `FILES_CHANGED_IN_GPU_ADDENDUM.txt` and
  `FILES_CHANGED_FOR_ASSEMBLY.txt`.
- **`provenance.json`** records every commit. The copy of `DECISIONS.md` in `code/` is
  4f327f7's; the current one is on `v3-dev`.

**Checks from Stage 1** (`report_stage1.md`):
- G1 (the release);
- C4's reruns: `norm_R_h` equals `package1` exactly at every k, on all 8 configurations;
- the Bratu and Burgers references.

C1 is reported in Stage 1 and rerun here at the Stage 2 code (item 1).

## Code since the pre-submission note

The pre-submission note and the reply refer to `v3-dev` at b2b99de. Since then there are 16
commits (`git log b2b99de..7af8855`). `DECISIONS.md` on `v3-dev` has an entry for each
step, newest first.

| Commits | What | `DECISIONS.md` entry | Report |
|---|---|---|---|
| 72c81b9, ee5eeca, a722c5d, 738b879 | `CITATION.cff` for v2.0.1 (tag on `main`, merged in); `RELEASE.md` and README with both DOIs | -- (`RELEASE.md`) | -- |
| f0a6a77 | Item 8: the manufactured solution compatible with the bases (the reply, 3.2) | "after the advisor's reply: item 8" | item 8 |
| 2eaa5db | Check C8 measured like for like, and the correction to batch 1 (the reply, 1.2) | "after the advisor's reply: check C8" | item 6a |
| 629b62f | Item 2's comparison table with eps_ref; item 6b's labels; the Beltrami release-identity test (the reply, 3.2, 3.5, 3.6) | "after the advisor's reply: items 2 and 6b, and Beltrami's CPU path" | items 2, 6b, 7 |
| b86bb6a | Item 4's normal-equation control as the reply's two variants (the reply, 3.1) | "after the advisor's reply: item 4" | item 4 |
| 28eccb6 | Item 5, the Darcy network with hard Dirichlet conditions, on FASTER (the reply, 1.1) | "batch 4: item 5" | item 5 |
| ecf8d48 | Fixes from a review in another session: item 5's output gain (below); item 7 saves its timing before the SVD; item 3's caveat on Burgers' target | "batch 5" | items 3, 5, 7 |
| 97ec184 | Item 5b, the Darcy control (the reply before submission, 5) | "batch 6: item 5b" | item 5 |
| 4f327f7 | **The Stage 2 commit.** The tests clear the submission environment (a preflight failure); FASTER contributes only item 5 | "batch 7" | events below |
| 5271d8b, e5eb447 | The torch wheel's NVIDIA libraries first, and item 7's A100 addendum; the assembly's commit check | "batch 8" | item 7 |
| d697ba3, 7af8855 | Beltrami 22,288 recorded as not fitting on the A100; the assembly at a later commit; assembly-script fixes | "batch 9a", "batch 9b" | item 7, events below |

The results come from 4f327f7, except item 7's A100 series (e5eb447). The four commits after
that change only the jobs' library environment, the records and the assembly
(`code/FILES_CHANGED_IN_GPU_ADDENDUM.txt`, `code/FILES_CHANGED_FOR_ASSEMBLY.txt`).

---

## 1. One page

### Classical baselines (item 1, P2-3), side by side with LiL-Q

The error, iterations k, and time to the stopping rule, on a whole Grace CPU node.

| Problem | Method | Free coefficients | k | Error | Time |
|---|---|---|---|---|---|
| Bratu | SQ-CGL p = 12 | 100 | 6 | 3.25e-5 | 4.5 ms |
| Bratu | SQ-CGL p = 17 | 225 | 6 | 2.39e-6 | 14 ms |
| Bratu | SQ-CGL p = 24 | 484 | 6 | 5.05e-7 | 44 ms |
| Bratu | LS-hard-CGL-1.5 p = 15 | 225 | 6 | 2.31e-6 | 72 ms |
| Bratu | LS-weakMS-CGL-3 p = 15 | 225 | 6 | 4.88e-5 | 0.11 s |
| Bratu | **LiL-Q (paper) P = 100** | 100 | 3 | 3.54e-4 | 32 ms |
| Bratu | **LiL-Q (paper) P = 225** | 225 | 4 | 5.83e-5 | 0.20 s |
| Kovasznay | SQ-PN-PN-2 p_d = 15 | 619 | 6 | 1.85e-5 (u) | 47 ms |
| Kovasznay | SQ-PN-PN-2 p_d = 20 | 1,124 | 6 | 4.78e-9 (u) | 0.14 s |
| Kovasznay | SQ-PN-PN-2 p_d = 25 | 1,779 | 6 | 1.94e-13 (u) | 0.33 s |
| Kovasznay | **LiL-Q (paper) P = 1,200** | 1,200 | 6 | 7.36e-9 (u) | 1.70 s |
| Kovasznay | **LiL-Q (paper) P = 1,875** | 1,875 | 6 | 7.00e-13 (u) | 4.38 s |
| Burgers | SQ-CGL p = 22 | 420 | 5 | 9.17e-4 | 28 ms |
| Burgers | SQ-CGL p = 27 | 650 | 5 | 1.55e-4 | 65 ms |
| Burgers | **LiL-Q (paper) P = 400** | 400 | 4 | 1.93e-4 | 0.43 s |
| Burgers | **LiL-Q (paper) P = 625** | 625 | 4 | 4.56e-5 | 1.22 s |

The square Chebyshev baselines reach a given error with fewer coefficients and in 13-44 times
less time than LiL-Q on Bratu and Kovasznay. For example, Kovasznay's 1.9e-13 against
7.0e-13 is 13 times faster, and Bratu's 3.3e-5 against 5.8e-5 is 44 times faster. On Burgers, LiL-Q is more accurate at the same
number of coefficients but about 19 times slower. The full rows and the work-precision
figures are in `P2_3_classical/`.

### NiL-N trained by Levenberg-Marquardt (item 2, P2-8), beside L-BFGS

Medians over three seeds, with the reference error's range over the seeds. The full table,
every size included, is in `P2_8_lm_networks/lm_vs_lbfgs.md`. No time ratios are formed:
LM ran on Grace CPUs, L-BFGS on the tables' A100.

| Benchmark, P | LM iterations | LM loss | LM eps_ref [min, max] | L-BFGS iterations | L-BFGS loss | L-BFGS eps_ref [min, max] |
|---|---|---|---|---|---|---|
| Bratu 25 | 28 (2/3 met target) | 2.1e-1 | 1.04 [0.066, 1.50] | 232 | 2.5e-1 | 0.085 [0.064, 0.096] |
| Bratu 100 | 2,000 * | 2.3e-4 | 2.3e-3 [1.0e-3, 2.0e-2] | 7,422 dagger | 2.6e-3 | 7.1e-3 [5.2e-3, 9.5e-3] |
| Bratu 225 | 2,000 * | 5.3e-6 | 2.4e-4 [2.1e-4, 7.7e-4] | 5,484 dagger | 7.6e-4 | 3.7e-3 [1.0e-3, 4.8e-3] |
| Burgers 25 | 51 | 5.7e-2 | 0.45 [0.23, 1.35] | 256 | 6.0e-2 | 0.13 [0.13, 0.19] |
| Burgers 625 | 846 | 5.0e-9 | 1.6e-5 [9.3e-6, 1.9e-5] | 4,032 dagger | 2.2e-5 | 6.4e-4 [5.3e-4, 6.9e-4] |
| BL 576 | 44 | 7.1e-4 | 2.4e-3 [1.8e-3, 3.2e-3] | 2,339 (2/3) | 7.5e-4 | 4.8e-3 [4.6e-3, 8.1e-3] |
| BL 1,024 | 105 | 7.4e-5 | 9.1e-4 [7.4e-4, 1.1e-3] | 7,436 (2/3) | 7.5e-5 | 1.0e-3 [7.8e-4, 1.2e-3] |

### Certified grids (item 3, P2-16)

| | Result |
|---|---|
| c2/c1, N/P = 10 (m = 2) | 1.009-1.041 |
| c2/c1, N/P = 20 (m = 2) | 1 to 1e-13 |
| c2/c1, N/P = 20 (m = 3) | 1.015-1.046 |
| c2/c1, N/P = 5 (m = 2) | Bratu P <= 225 and Burgers P = 25: not computable (C6); Burgers P = 100, 225, 400, 625: 3.21, 3.60, 1.51, 1.34 |
| Realized ratio rho_r | 1.00 throughout; largest 1.06 (Burgers P = 625, N/P = 5) |
| Bratu error / delta_P | 70 (P = 25), 16 (P = 100), 15 (P = 225) |
| Burgers error / delta_P | 1.6-5.1 |

The rule stops at k = 5 on every row, and its error agrees with k = 60's to about three digits.

### ELM sweep (item 4, P2-14)

The figure is `P2_14_elm_sweep/figures/elm_sweep.png`.
- **The error against sigma** (median of 5 seeds on Burgers, P = 625): 0.15 at sigma = 0.03,
  then 5.2e-4 at 1, 4.6e-5 at 2, and **1.9e-5 at 3** (the best), rising again to 2.7e-4 at
  10. The SVD rank goes from 26 to 616.
- **The loss target** (5e-9) is never met.
- **Table 3's two ELM rows are reproduced exactly:** ||R||^2 0.04944 at rank 43, and
  4.434e-4 at rank 176.

### ELM on Kovasznay and the normal-equation control (item 4, P2-4)

1,800 coefficients, sigma = 1, five seeds, K_max = 60. The ELM basis is shared by u, v and
p.

| Solver | Error in u at k = 60 | Kept |
|---|---|---|
| QR (`gelsy`) | 1.8e-10 to 2.5e-10 | rank 723-734 |
| Shifted Cholesky (shift 1e-12 tr/P) | 2.6e-4 to 6.0e-4 | -- |
| Pseudo-inverse via `eigh` | 2.8e-3 to 9.3e-3 | 228-233 eigenvalues |

On the Chebyshev surrogate the three solvers agree, and C5 passes at 3.6e-13 and 5.6e-13.

### Darcy network with hard Dirichlet conditions (item 5, P2-15; FASTER), and its control 5b (Grace)

delta_FV against finite volumes. Every network run stopped at 2,000 LM iterations (about 5
min on an A100).

| Field | Item 5: lifting + LM (seeds 0, 1, 2) | 5b: paper's rows + LM (seed 0) | Paper's NiL: rows + Adam (seed 0) | LiL |
|---|---|---|---|---|
| S1 | 7.6e-5, 7.5e-5, 7.6e-5 | 1.0e-4 | 2.1e-2 | 1.4e-4 |
| S2 | 2.9e-4, 2.9e-4, 3.0e-4 | 4.0e-4 | 4.7e-2 | 2.3e-4 |
| S3 | 1.9e-3, 9.0e-4, 9.4e-4 | 9.2e-4 | 5.1e-2 | 6.7e-4 |
| SPE10 | 9.0e-2, 1.1e-1, 1.1e-1 | 7.9e-2 | 8.7e-2 | 3.4e-2 |

### nu-refinement (item 6a, P2-2): viscous Buckley-Leverett

eps_ref at the stop against the refined reference (C7: every reference passes at 1e-6).

| nu | P = 576 | P = 1,024 | P = 1,600 |
|---|---|---|---|
| 0.1 | 9.1e-4 (target met at k = 4) | 2.0e-4 (target met at k = 4) | -- |
| 0.05 | 3.6e-2 (K_max) | 1.6e-2 (K_max) | -- |
| 0.02 | 0.15 | 0.13 | 0.13 |
| 0.01 | 0.22 | 0.19 | 0.67 (oscillates: max S - 1 = 5.9) |

Gravity: 6.2e-2 and 3.3e-2 at nu = 0.05, and 0.10 and 0.088 at nu = 0.02 (P = 576 and 1,024).

### Scaling exponents (item 7, P2-1)

The CPU is a whole Grace node; the A100 is Grace's A100-PCIE-40GB, in the GPU addendum.

| Series | Total time | Solve | Assembly |
|---|---|---|---|
| Kovasznay, P from 1,875 to 7,500 (CPU) | P^3.25 | P^3.32 | P^2.22 |
| Kovasznay, N at P = 1,200 (CPU) | N^1.07 | N^1.07 | N^1.11 |
| Beltrami, P from 7,984 to 22,288 (CPU) | P^2.90 | P^2.91 | P^2.01 |
| Kovasznay, P from 1,875 to 7,500 (A100) | P^2.03 | P^1.71 | P^2.24 |
| Kovasznay, N at P = 1,200 (A100) | N^0.97 | N^0.69 | N^1.13 |
| Beltrami, 7,984 and 13,764 only (A100; 22,288 does not fit) | P^2.68 (2 points) | P^2.73 | P^2.66 |

---

## 2. One section per item

### Stage-wide events

- **5 October: the first submission (ecf8d48) was cancelled before any compute ran.** The
  reply requesting item 5b had not yet been read, and Stage 2 had to come from one commit
  with 5b in.
- **6 October: the 97ec184 submission's preflight failed.** A bash test inherited the
  submission's `CPU_PARTITION`. It was fixed (4f327f7: the tests clear the submission
  environment) and resubmitted. No compute ran.
- **Grace and FASTER share `$SCRATCH`.** Both wrote one results folder under one lock, and
  the assembly takes only item 5's folder from FASTER's tarball.
- **The first assembly stopped on a fault in the assembly script.** It was run as a script,
  not imported. It was fixed in 7af8855 (batch 9b), together with two faults that would have
  appeared only on a second assembly. The real partial state, reassembled, matches a one-pass
  assembly file by file. No run was repeated.

### Item 1 (P2-3): classical baselines

**What ran.** Bratu with SQ-CGL (p = 6-24), LS-hard-CGL-1.5 (p = 5-24) and
LS-weakMS-CGL-3 (p = 5, 10, 15). Kovasznay with the P_N - P_{N-2} square system (p_d =
10-25, corner pin). Burgers with SQ-CGL (p = 7-27). Each ran on a whole CPU node, with the
paper's LiL-Q configurations as the same-day reference.

**Checks:**
- **C1:** the Bratu port reproduces the pilot's `rows.json` to three digits at p = 12, 14,
  20 and 22 (relative differences 3e-12 to 7e-11). The sanity run, LS-hard with N = P = 100
  on the interior CGL points of the 12-point grid, reproduces SQ-CGL at p = 12 to 1.2e-15.
  - **First run:** Stage 1, at c1e7f4c (`P2_3_classical/check_c1.json`).
  - **Rerun:** the classical code changed after c1e7f4c, so C1 was rerun on the laptop at
    the Stage 2 code. That code is identical to 4f327f7's in every classical file. All five
    numbers came out the same as Stage 1's, bit for bit
    (`P2_3_classical/check_c1_rerun_stage2_code.json`).
- **C2:** the reference error (p = 48 against p = 64, Stage 1) is 3.6e-9, about 67 times
  below the smallest reported Bratu error (LS-hard p = 24: 2.43e-7). Stage 2 rebuilt the
  references, which agree with Stage 1's to 1.3e-14.
- **C9:** nothing else ran on the node; warm-up done; threads recorded. The same-day LiL-Q
  runs are within 15% of package1: 0.96-1.06 times (Bratu, Kovasznay, Burgers).

**Deviations:** none. Burgers' timing reading, as accepted in the reply (3.1).

**Laptop beside Grace:** errors agree to 1e-10 or better (Kovasznay p_d = 25: 1%, at the
1e-13 level). Times are not compared.

### Item 2 (P2-8): NiL-N by Levenberg-Marquardt

**What ran.** The four-method NiL-N networks, with only the optimizer changed (F2's
damping, the f1 stall rule, 2,000 iterations or 15 min), three seeds, on a whole CPU node.

**Checks:**
- **C3:** r . r at iteration 0 equals the L-BFGS objective on all 36 runs, to 6e-16.
- **`gpu-list` is empty:** no configuration hit the wall-time cap, so the A100 reruns were
  not needed.

**Findings:**
- **LM reaches a lower final loss than L-BFGS at every size, in 2.7-71 times fewer
  iterations.**
- **Its reference error is lower at most sizes.** The gain is largest on Burgers P = 625
  (39 times) and Bratu P = 225 (16 times). It is not lower everywhere: Burgers P = 100 is
  8.0e-3 against 7.6e-3, BL P = 256 is level, and at the smallest sizes (Bratu P = 25,
  Burgers P = 25, BL P = 64) LM's error is the larger.
- **At the smallest size the loss target does not guarantee the error.** Bratu P = 25,
  seed 0 meets the target (loss 0.16 < 0.25) with eps_ref 1.04. Its solution has
  u(0.5, 0.5) = -0.18 against the reference's 0.866: a spurious low-loss state of 37
  parameters on 265 rows, not the second Bratu branch. Seed 1 stalls at loss 5.3, with
  eps_ref 1.50. Seed 2 finds the solution (eps_ref 0.066). This is the loss-error
  disconnect of the reply's Section 3.2, in its plainest form.

**Deviations:** none.

**Laptop beside Grace:** the laptop ran Burgers and BL with one seed, so the medians are
not like for like.

### Item 3 (P2-16): certified-grid runs

**Checks:**
- **C4, in three parts:**
  - **Clenshaw-Curtis exactness** holds on every grid, to at most 3.5e-15 in the integral of
    degree M - 1 and 4.4e-13 in the norm. This is the exact form, with the squared norm
    exact up to degree floor((M - 1)/2).
  - **The reruns' `norm_R_h`** equals `package1` exactly at every k (Stage 1, Grace job
    19956475).
  - **The Cole-Hopf integrator** passes the Basdevant check on Stage 2's rebuilt reference:
    u_x(0, 1.6037/pi) = -152.005162 against the published -152.00516
    (`reference/burgers_cole_hopf_checks.json`).
- **C6:** a constant is computed only where dim < N_interior. N/P = 5 at P <= 225 is
  marked not computable.

**Findings:** in the one-page table.
- **The loss target, as the reply asked:** Bratu meets the paper's target at k = 2-4.
  Burgers never does, and is reported with the rule's stop and the k = 60 values; the
  targets are not redefined.
- **Caveat on the target comparison:** for Burgers, the CC-weighted loss gives the initial
  line 2/6 and each lateral line 1/6 of the boundary weight. The paper's loss weights each
  line by 1. The comparison is like for like only for Bratu. Under the paper's weighting
  Burgers' loss is larger, so the conclusion stands.

**Laptop beside Grace:** identical, except rho_r's maximum at Burgers P = 625, N/P = 5:
1.06 on Grace against 1.37 on the laptop.

### Item 4 (P2-14, P2-4): ELM sweep, ELM on Kovasznay, normal-equation control

**Checks:** C5 passes for both variants on the surrogate (3.6e-13 shifted, 5.6e-13
`eigh`, all 300 eigenvalues kept). The harness reproduces Table 3's ELM rows exactly.

**The normal-equation control (the reply's two variants).** Shifted Cholesky needs the
shift 1e-12 tr(A^T A)/P on every seed; 1e-16 and 1e-14 do not factor, since kappa(A^T A)
is about 1e21. The pseudo-inverse keeps 228-233 of 1,800 eigenvalues.
- **What sets the stall levels: the cutoffs.**
  - `eigh` drops singular values of A below sqrt(P eps) sigma_max, about 6.3e-7 sigma_max.
  - The shift damps those below about 1e-6 sigma_max.
  - QR resolves down to about eps sigma_max.
- **What follows from forming A^T A in double precision.** It leaves nothing below about
  sqrt(eps) sigma_max. So the stall levels (1e-4 to 1e-2, where QR reaches 2e-10) show the
  squaring of kappa conjectured in Section 7, through these cutoffs.

**Deviations:** the control was respecified by the reply (Section 3.1), and the job's
walltime went from 2 h to 3 h.

**Laptop beside Grace:** the normal-equation errors are identical to three digits. The QR
floor varies between 2e-10 and 3e-10 (round-off).

### Item 5 (P2-15): Darcy network with hard Dirichlet conditions (FASTER), and 5b

**What ran.** The run differs from the paper's NiL baseline in two things only: the hard
Dirichlet lifting h* = y* + omega NN_h / 2 (the paper's output gain) and the optimizer
(Levenberg-Marquardt with F2's damping). Everything else is the paper's:
- the three SiLU networks;
- the seeds and initialization;
- the residuals, weights and lateral rows.

It covers the four fields and seeds 0, 1, 2, on a FASTER shared A100.

**Deviation: the lifting's gain.** The reply (1.1) writes h* = y* + omega NN_h. We used
h* = y* + omega NN_h / 2.
- **The gain.** The paper's network gives P = NN P_HALF + P_MID, that is h* = 1/2 + NN/2.
  With the factor 1/2, one unit of NN_h moves the pressure by P_HALF in both runs, and the
  starting correction is at the paper's scale. Read literally, the formula doubles the
  network's output scale at mid-height.
- **What else holds.** omega = 4 y*(1 - y*) still has unit maximum, and the Dirichlet data
  hold exactly either way.
- **Why the paper's gain.** It was chosen on the design, before its smoke run existed, for
  two reasons:
  - **Your two things only.** The reply asks for the run to differ from the paper's NiL in
    two things only. The literal gain would add a third, the output scale.
  - **The comparison with 5b.** 5b keeps the paper's network and gain. With the same gain,
    item 5 against 5b changes only the boundary treatment; with the literal gain it would
    change the scale too.
- **History.** A review before submission found this (`DECISIONS.md`, batch 4). The first
  version (28eccb6) used the literal form; ecf8d48 changed it.
- **The smoke runs (S1, seed 0, 5 min, about 150 of the 2,000 iterations;
  `laptop_preview/P2_15_darcy_hardbc/`).**
  - The literal gain gave delta_FV 1.4e-4 and the paper's gain 2.2e-4; the paper's NiL gave
    2.1e-2.
  - This is one seed after 150 iterations. It says nothing about the 2,000-iteration
    result.
  - Switching to the gain with the lower delta_FV would choose a configuration by its test
    error, which Section 12.4 rules out.
  - Only the paper's gain ran on FASTER.

**Checks:**
- **C3 analogue:** r . r equals the paper's loss on the lifted network to 1e-12 on every
  run.
- **C7 (item 5):** the paper's top and bottom Dirichlet terms are exactly 0 at
  initialization on every run, so the lifted network meets the data to round-off.

**Item 5b, the control** (requested in the reply before submission). The paper's network
and its Dirichlet rows, trained by the same LM: seed 0, four fields, on a Grace shared A100.
Its time is not quoted.
- **The boundary treatment (5 against 5b) changes little.** On S1-S3 the two are within a
  factor of 2, with the lifting ahead on S1 and S2 and behind on S3 seed 0.
- **The optimizer (5b against the paper's NiL) accounts for the S1-S3 gap.** LM's error is
  56-210 times below Adam's.
- **On SPE10 no network approaches LiL** (0.034): all are 0.08-0.11.

**Times.** FASTER and Grace A100 times are reported with the hardware and are not compared
with the manuscript's Grace times.

**Laptop:** smoke runs only (5-min cap). Not comparable with the 2,000-iteration runs.

### Item 6a (P2-2): nu-refinement

**Checks:**
- **C7:** every refined reference passes the two-refinement agreement at 1e-6.
- **C8 passes at 1e-10.** The nu = 0.1 reruns at P = 576 and 1,024 equal package1 bit for
  bit at every k (`norm_R_h`, final loss and eps_u: relative difference 0; the same
  iterations; `gelsy` keeps the same columns, 576/576 and 1,005/1,006). So Grace reproduces
  these rank-deficient runs exactly, as the corrected C8 reading said.
- **The criteria (reply, 1.2), under the like-for-like rule accepted before
  submission, all hold:**
  - the stopping iteration agrees;
  - `norm_R_h` agrees at every k;
  - the final loss and the reference error agree.

  Each difference is 0, so within the 1e-10 level. It is also within the original
  mixed-row figures recorded beside it (6.3e-5 for P = 576, 1.1e-4 for P = 1,024).
  `P2_2_nu_refinement/check_c8.json` holds both, run by run, with the columns `gelsy`
  keeps.

**For the manuscript's reproducibility sentence** (the reply before submission, item 2).
These are relative differences, laptop against Grace (package1). The iterations agree in
every run.

| Run | norm_R_h (largest over k) | Final loss | eps_u (final) |
|---|---|---|---|
| P = 576, paper pass | 6.6e-4 | 1.3e-3 | 3.8e-3 |
| P = 576, K_max pass | 6.5e-4 | 1.5e-6 | 9.0e-4 |
| P = 1,024, paper pass | 2.2e-4 | 4.5e-4 | 1.7e-2 |
| P = 1,024, K_max pass | 5.8e-3 | 4.7e-3 | 1.1e-2 |

**Findings:**
- At nu <= 0.05, no configuration meets the paper's loss target in 60 iterations.
- **At P = 1,600, nu = 0.01, the iteration oscillates without converging.** The error is
  0.67, and the overshoot max S - 1 reaches 5.9. This is reported with its loss history
  (`P2_2_nu_refinement/viscous_P1600_nu0.01_kmax/iterations.csv`). No damped step was run
  (the reply, Section 4).

**Laptop beside Grace:** the rank-deficient runs differ at the machine-to-machine level,
up to 46% at P = 1,600, nu = 0.02 (0.13 on Grace against 0.25 on the laptop).

### Item 6b (P2-9): boundary-conforming bases

**Labels, as the reply asked:** `lifted_sine_x(1-x)`, `lifted_sine_plain`, and
`paper (<type>)`. delta_P is on every row.

**Findings:**
- **Viscous BL: the plain-sine lifting is the best basis at every P from 256 up.** At P =
  1,024 its error is 6.9e-5 against 2.0e-4 for the paper's basis, within 1.4-2.0 times
  delta_P. The factor x(1 - x) stalls at 0.11-0.15.
- **Gravity BL:** both lifted bases are worse than the paper's cosine-Fourier basis
  (3.3e-2 and 6.9e-2 against 1.2e-2 at P = 1,024), and never meet the target.
- **Bratu:** sin x sin is worse than the paper's basis (it stalls, error / delta_P up to
  58). The weak Chebyshev basis is within a factor of 2 of it.

**Laptop beside Grace:** identical apart from the rank-deficient viscous paper-basis rows
(up to 4%).

### Item 7 (P2-1): scaling in N and P

**CPU, a whole Grace node.**
- Kovasznay, P = 1,875 to 7,500: about P^3.25.
- Kovasznay, N/P = 3 to 20 at P = 1,200: about N^1.07.
- Beltrami at 7,984, 13,764 and 22,288 coefficients: about P^2.90. The largest takes
  5,230 s with a 63.7 GB host peak, and its timing record was written before the off-clock
  SVD (batch 5).

**C9:**
- **Each CPU job held a whole node,** marked `exclusive` in `hardware.json`.
- **The A100 job held a whole node too.** It is not flagged exclusive, but it had all 48 of
  g018's cores and 360 GB, so no other job could start on it.
- **All 23 timed runs** record 48 threads and an untimed warm-up before their single timing,
  which Section 10 asks for and `run.json` discloses.

**C8':** the paper-size points reproduce package1's errors exactly (relative difference 0),
on the CPU at Kovasznay P = 1,875, on the GPU at 1,875, and for Beltrami. The CPU times are
1.11 and 1.02 times package1's (within 15%).

**The A100 series: the one exception to "one commit".**
- **What failed.** The first run (20017337, retried as 20018277) failed at Kovasznay P =
  3,675 in cuSOLVER's QR. The cluster environment had loaded the PyTorch module's CUDA
  12.6.0 libraries ahead of the torch wheel's own.
- **The library record names the pair.** It is the wheel's cuBLAS 12.6.4.1 with the
  module's cuBLASLt 12.6.0.22. Diagnostic job 20028701 found that the QR fails for a band of
  shapes around 11,036 x 3,675 in that environment, and passes for all of them with the
  wheel's libraries first.
- **The fix** is in `env.sh`. Its first submission (5271d8b) failed its preflight, on an
  assembly check that misbehaved without git; the library probe had already passed. The
  check was fixed in e5eb447.
- **e5eb447** carries the fix. The whole A100 series reran in the corrected
  environment, as the GPU addendum, into its own folder.
- **C8' is checked twice, and passes both times.** On the old-environment 1,875 point, in
  Package 1's environment, and on the addendum's 1,875 point, the errors match Package 1's
  exactly (relative difference 0). The corrected libraries change no result.
- **No other result used the failing routine.** Items 5 and 5b and the preflights ran in
  the old environment but do not use it.
- **The addendum ran every size but one.** Kovasznay P = 1,875-7,500 (3,675 included), the
  N series, and Beltrami at 7,984 and 13,764 all ran. Every system has full rank. Kovasznay's
  E_u runs from 2.0e-15 to 7.0e-13 over the P series.
- **A100 against the CPU node.**
  - Kovasznay is 4.6 times faster at P = 1,875, rising to 24 times at 7,500. The N series is
    3.9-5.2 times faster.
  - Beltrami is 24 times faster at 7,984 (11.3 s against 272 s), and 40 times at 13,764
    (48.8 s against 1,941 s).
- **The A100 does not reach the O(N P^2) regime at these sizes.**
  - The solve grows as P^1.71 against the CPU's P^3.32. A QR of a few thousand columns
    leaves the A100 underused, so its time grows slower than the work.
  - The assembly runs on the host (P^2.24, as on the CPU) and is most of the A100's time:
    11.3 of 16.8 s at P = 7,500.
  - So the A100 exponents describe this hardware at these sizes, not the method's
    asymptotics. The CPU exponents are the ones to set against the expected P^3 and N^1.
  - Beltrami's A100 exponent comes from two points only.
- **Beltrami at 22,288 does not fit on the A100 (the size Section 10 asks for).**
  - The QR of the 75,689 x 22,288 system finished. The next step, applying Q^T to b
    (cuSOLVER's ormqr), was refused: its workspace query returned
    `CUSOLVER_STATUS_INVALID_VALUE`.
  - On the laptop's GPU, the same query accepts every smaller system and rejects this one.
    ormqr needs a workspace of about the size of A, here about 2.2e9 doubles, more than its
    32-bit size allows.
  - Two copies of A (the input and the QR's) plus that workspace come to 18.87 GB at
    13,764, against a measured peak of 18.83 GB. At 22,288 the same sum is about 44.6 GB,
    and the A100 has 42.4 GB. So it would not fit even without the 32-bit limit.
  - The largest Beltrami system that runs on the A100 is 13,764 coefficients (48.8 s,
    18.8 GB peak). Recorded as `did_not_fit`, from the job's log.
- **An observation, not acted on.** The memory is dominated by ormqr's workspace, not by A.
  Factorizing [A | b] instead gives Q^T b without ormqr, at about 27 GB, so 22,288 would
  fit. That would be a different GPU path from the paper's, so it was not run.

**Beltrami's CPU path** is unchanged and bit-identical to the release, as the reply (3.6)
asked to have recorded. It is in `DECISIONS.md`, with `tests/test_beltrami_release_identity.py`.

**Laptop beside Grace.** Every run has the same iterations and rank on both machines (the
laptop did not run Beltrami 22,288).
- **Errors above round-off agree closely.** Beltrami's u and p at t = 1 agree to 6e-11 or
  better, and the Kovasznay N series (E_u about 5e-9) to 7e-8.
- **At the round-off floor they differ by up to a factor of 2.** The Kovasznay P series has
  E_u from 1.9e-15 to 7.5e-13 on either machine. The difference is 6e-6 at P = 1,875 on the
  CPU, 9e-4 on the GPU, and up to a factor of 2.2 from P = 2,700 up.
- **Times are not compared.**

### Item 8 (P2-10): manufactured elasticity solution

**The solution compatible with the bases (the reply, Section 3.2).** The error follows
delta_P down to 8.1e-14 (u_x) and 4.2e-13 (u_y) at P = 1,250. error / delta_P is 1.6-3.5
for u_x and 4.8-18 for u_y. Every system has full rank, with kappa from 1.1e2 to 4.4e4.

**The specified solution, in one row:** P = 1,250, error 0.22 (u_x) and 0.14 (u_y),
against delta_P of 7.3e-4 and 2.2e-4. A manufactured solution that breaks the symmetry
built into the bases is not approximated by them: the Section 7.4 guidance, in negative
form.

**Laptop beside Grace:** identical (round-off at P = 1,250).

---

## 3. SU per job against `su_plan.csv`

Grace, at Grace's rates, from `su_per_job.csv`.

| Job | Requested | Expected | Charged |
|---|---|---|---|
| preflight (3 attempts: a cancelled submission, a failed test, OK) | 80 | 15 | 25.2 |
| references | 18 | 4 | 1.4 |
| classical Bratu / Kovasznay / Burgers | 48 / 48 / 48 | 24 / 24 / 4 | 0.4 / 1.4 / 0.4 |
| LM networks Bratu / Burgers / BL | 24 / 48 / 48 | 4 / 4 / 4 | 6.4 / 6.1 / 1.3 |
| certified | 24 | 6 | 3.0 |
| ELM | 72 | 36 | 24.7 |
| nu-refinement | 72 | 32 | 26.8 |
| bases | 24 | 12 | 11.6 |
| scaling CPU a / b | 120 / 288 | 72 / 220 | 79.8 / 148.7 |
| scaling GPU (2 failed attempts) | 288 | 150 | 6.2 |
| elasticity manufactured | 12 | 6 | 0.3 |
| Darcy 5b (4 tasks) | 240 | 176 | 30.4 |
| report jobs | 12 | 1 | about 1.5 |
| **Grace main stage** | **1,514** | **about 794** | **about 376** |
| diagnostics (cuSOLVER test, library A/B) | -- | -- | about 14 |
| GPU addendum (preflight 5271d8b failed; e5eb447 ran) | 380 | about 160 | about 115 |

FASTER (item 5), at the measured 137 SU/h per shared A100:
- preflight about 16;
- array about 164 (12 x 5.5-6.9 min);
- report about 0.4;
- **total about 181**, against 1,382 requested and about 700 expected.

**Stage 2 in all: about 686 SU.** Grace: about 505 (main stage 376, diagnostics 14, GPU addendum
115). FASTER: about 181.

The failed and cancelled attempts are counted above. The unrelated old Grace jobs that
matched FASTER's job numbers through the shared logs folder are excluded.

## 4. Skipped items

- **The A100 reruns of item 2** (`p2s2_lm_networks_gpu`): not needed (`gpu-list` was
  empty).
- **A damped step for item 6a at P = 1,600:** not run, as the reply asked (Section 4).

## 5. Questions

1. **Item 5's finding.** With LM, the boundary treatment changes little and the optimizer
   accounts for almost all of the network/LiL gap on S1-S3. Does this change how Section
   6.8 frames the LiL/NiL comparison?
2. **Item 1.** The square Chebyshev baselines beat LiL-Q on Bratu and Kovasznay in both
   coefficients and time. How should the manuscript position LiL-Q against them? For
   example, by the generality of the trial space rather than its efficiency on smooth
   tensor-product problems.
3. **Item 6b.** The plain-sine lifting is the best viscous basis but the worst for gravity.
   Is the gravity result stated in the manuscript too, or only the viscous one?
4. **Item 7.** On the A100, the paper's GPU path (QR, then ormqr for Q^T b) stops at Beltrami
   22,288, because of ormqr's workspace. Factorizing [A | b] would fit it in about 27 GB. Is
   that worth a sentence in the manuscript's discussion of GPU memory, or is "the largest
   size that fits on a 40 GB A100 is 13,764" enough?

## Files (`package2_results/`)

Stage 1's files (`G1_release/`, `P2_12_reference_errors/`, `P2_6_elasticity_counts.csv`,
`report_stage1.md`) are unchanged. New in Stage 2:
- **`P2_3_classical/` (item 1):** `bratu/`, `kovasznay/`, `burgers/`, `figures/`
  (work-precision). Also `check_c1.*` (Stage 1) and `check_c1_rerun_stage2_code.json`.
- **`P2_8_lm_networks/` (item 2):** one folder per run; `lm_vs_lbfgs.md`,
  `four_method_lm_medians.csv`, `four_method_lm_rows.csv`, `figures/`.
- **`P2_16_certified/` (item 3):** one folder per run; `terminal.csv`, `constants.csv`,
  `check_c4_c6.json`, `figures/`.
- **`P2_14_elm_sweep/` and `P2_4_elm_kovasznay/` (item 4):**
  - `sweep.csv`, `figures/elm_sweep.png`;
  - `qr/`, `normal_shifted/`, `normal_eigh/`, `chebyshev_P300/`, `summary.csv`,
    `check_c5.json`.
- **`P2_15_darcy_hardbc/` (item 5, FASTER) and `P2_15b_darcy_softbc_lm/` (5b, Grace):** one
  folder per field and seed; `darcy_hardbc_rows.csv`, `darcy_softbc_lm_rows.csv`.
- **`P2_2_nu_refinement/` (item 6a):** one folder per run; `nu_refinement.csv`,
  `check_c7.json`, `check_c8.json`, `figures/`.
- **`P2_9_bases/` (item 6b):** one folder per run; `rows.csv`, `references.json`.
- **`P2_1_scaling/` (item 7):** `kovasznay/`, `beltrami/` (CPU and the addendum's A100 runs),
  `scaling.csv`, `exponents.csv`, `check_c8prime.json`, `figures/`.
  - `old_environment/`: the first A100 runs, with
    `check_c8prime_old_environment.json`.
  - `gpu_addendum/`: the library records, `cuda_libraries.json` and
    `cuda_libraries_module_first.json`.
- **`P2_10_elasticity_manufactured/` (item 8):** `compatible/`, `specified/`, `rows.csv`.
- **`reference/`:** Stage 2's references; `stage1/` (Stage 1's copies that differ);
  `stage2_vs_stage1.json`.
- **`code/`:** the code at 4f327f7. Also `COMMIT_stage1`, `COMMIT_stage2` and
  `COMMIT_stage2_gpu`, and the lists of changed files.
- **Provenance:** `provenance.json`, `assembly_log.txt`, `hardware.json`, `environment.txt`.
- **Jobs and SU:** `slurm_logs/` (Grace, FASTER, GPU addendum), `su_per_job.csv`,
  `stage2_*sacct.txt`, `stage2_*report_log.txt`, `su_plan.csv`.
- **`laptop_preview/`:** the laptop previews quoted under each item, with its own README.

# Computational Package 3: report

**To:** R. Younis. **From:** G. Awojinrin. **Date:** 9 October 2026.
**Refers to:** the Package 3 instructions and Addendum 1 (both 8 October).

**Code:** `awojinrin/lilq-pinn`, branch `v3-dev` at 086461c. These are the 9 commits after
7af8855, the commit of the Stage 2 report:

| commit | what | DECISIONS.md entry |
|---|---|---|
| b82a4bb | CC-CGL grids, the sampling constants, K1, K2 | batch 0 |
| 97ed513 | item 1, Beltrami on CC-CGL grids | batch 1 |
| ee286ea | item 2, Buckley–Leverett 2a and 2b | batch 2 |
| c9943bb | item 3, the affine certificates | batch 3 |
| 676304a | item 4 (Addendum 1); B2 judged as Package 1 | Addendum 1 |
| 97e463d | the Grace submission, K0, K8, B3 level 2's gate (the runs' commit) | batch 4 |
| 86cc344 | L1 compares P2-12's terminal row like with like | after wave 1 |
| 3e13572 | the results assembly | the assembly |
| 086461c | the assembly takes the hardware record captured after the runs | the assembly |

Every run ran at 97e463d, under the source lock. The three later commits change only L1's
comparison and the assembly (Section 4).

**Results:** `package3_results/`, in the layout of Section 9. They are on the branch
`package3-results` of `awojinrin/lilq-pinn`, under `results/package3/`. The branch starts at
086461c; its README says what it holds.

## Summary

Package 3 is done, and every check passes: K0–K8, item 3's all-digit re-solves, and L1–L4.
It cost 630 SU on Grace, of the 850 cap. B3 level 2 passed the launch rule and the budget
guard, and completed. No Section 8.2 rule had to be applied, and no Section 8.1 condition
arose.

**For your reply:** L1 compares a value that P2-12 and the control compute in two different
ways (Section 4.1). With like compared with like, every value at k ≤ 4 is identical. Is that
reading of L1 acceptable?

## 1. What ran

All on Grace, partition `medium`, at 97e463d, in two waves:
- **Wave 1** (9 jobs, 9 October): the preflight (the test suite, K1, K2), K0, items 1–4, and
  the report.
- **Wave 2:** B3 level 2 and the report again.
- **After the runs:** a short job, which recorded the hardware and software (Section 4.4).

| item | runs |
|---|---|
| 1 | B1, B2, B3 at levels 1 and 2 (six runs), plus K3's two B1 runs and K6's B1 rerun |
| 2a, 2b | 2 cases × 4 sizes × 3 ratios each (48 runs), plus K5 and K6 |
| 3 | elasticity: 3 solutions × 5 sizes; Darcy: 4 fields; plus K5 (4 runs) |
| 4 | P = 625 (control), 900, 1,024, 1,225, logged and clean-timed. P = 1,600 was not needed: P = 900 and up reached 1.6e-5 |
| checks | K0; K8 (the paper's pinned Beltrami run, and BL P = 576) |

The Beltrami runs ran on whole nodes, at 48 threads. Item 2 ran six runs side by side, at 4
threads each. Item 3's re-solves ran at their original runs' thread counts: 48 for package1,
24 for P2-10. Each run's `run.json` records its count.

## 2. Checks

All in `checks.json`, each with its source file.

| check | result |
|---|---|
| K0 | passed. 600,000 × 7,984 (4.79e9 entries). gelsy: relative residual 1.39e-15, error 1.17e-15. Pivoted QR: 1.38e-15, 1.16e-15 |
| K1 | passed. All 30 printed values of Sections 3.3 and 4.2 match. Direct against Kronecker: 1.8e-15 at most. Against `expected_constants.py` (laptop, batch 0): 8.9e-14 at most |
| K2 | passed. 144 rules, worst error 3.8e-14 |
| K3 | passed. B1 level 1 with the pin weight × 1e-3 and × 1e3: the errors change by 1.7e-14 at most |
| K4 | passed. B2 at k = 1: 2.7e-14 at most (items 1 and 2) |
| K5 | passed. BL: c₁, c₂ and ϱ_r change by 5.6e-13 at most. Elasticity: 6.6e-13. Darcy SPE10 (n_q = 6): 1.7e-7. All against 1e-3 |
| K6 | passed. B1 level 1 rerun and BL viscous P = 256 rerun: bit for bit, timings aside |
| K7 | passed. Every run is full rank |
| K8 | passed. The paper's pinned Beltrami run and viscous BL P = 576 reproduce package1's iteration counts and residual norms to all digits |
| item 3 re-solves | passed. All 19 reproduce the package1 / P2-10 residual to all digits. The assembly equals the paper's bit for bit |
| L1 | passed, as re-evaluated (Section 4.1). Grace's evaluation reported one difference, of 9e-16 |
| L2 | passed. 1.35, 1.13, 1.13 s; median 1.13 s against 1.18 s (−4.5%) |
| L3 | passed. 4.0e-15 at k = 1 |
| L4 | passed. Every clean run ends at the logged pass's ‖R‖_h, bit for bit, in exactly max_iter iterations |

## 3. Results

### Item 1: Beltrami on CC-CGL grids

Errors are at the returned iterate (the 1e-9 iterate), space–time relative L², with p
gauge-corrected. c₁ and c₂ are the overall constants (interior, 21 blocks, overall in
`constants.csv`). Every run is full rank.

| run | N | N/P | 1e-9 at | rule (class) | u | p | combined | error/δ_P | c₁ | c₂ | κ | GB | h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 L1 | 160,386 | 115.1 | 7 | 4 (A) | 1.77e-02 | 1.26e-01 | 1.02e-01 | 21.1 | 0.9518 | 1.0035 | 2.94e+03 | 6 | 0.11 |
| B1 L2 | 348,165 | 249.9 | 7 | 4 (A) | 1.77e-02 | 1.26e-01 | 1.02e-01 | 21.1 | 0.9968 | 1.0000 | 2.94e+03 | 11 | 0.17 |
| B2 L1 | 273,381 | 86.2 | 6 | 4 (A) | 2.96e-03 | 2.39e-02 | 1.93e-02 | 37.0 | 0.9267 | 1.0050 | 1.12e+04 | 17 | 0.36 |
| B2 L2 | 542,382 | 171.0 | 6 | 4 (A) | 2.96e-03 | 2.39e-02 | 1.93e-02 | 37.0 | 0.9904 | 1.0005 | 1.12e+04 | 34 | 0.75 |
| B3 L1 | 348,168 | 43.6 | 4 | 4 (A) | 2.91e-04 | 2.27e-03 | 1.84e-03 | 58.6 | 0.8213 | 1.0082 | 2.91e+04 | 80 | 2.86 |
| B3 L2 | 665,331 | 83.3 | 4 | 4 (A) | 2.91e-04 | 2.27e-03 | 1.84e-03 | 58.6 | 0.9688 | 1.0024 | 2.91e+04 | 153 | 5.75 |

ϱ_r (B1 level 1, every iterate) is 1.000000 to six digits. The level-2 runs reproduce
level 1's errors to 1.6e-12 (B1), 1.2e-9 (B2) and 1.5e-7 (B3).

### Item 2a: Buckley–Leverett, tensor Chebyshev

Errors are against the P2-2 references. The §6 constants are at the rule's iterate, with the
maximum over the run. The rule fired in every run, all class A. Every run is full rank, and
no ϱ_r is flagged as round-off.

| case | P | N/P | rule (class) | error at rule | error/δ_P | c₂/c₁ at rule | max c₂/c₁ | max ϱ_r |
|---|---|---|---|---|---|---|---|---|
| viscous | 64 | 5 | 16 (A) | 1.85e-01 | 11.6 | 1.025 | 1.19 | 1.008 |
| viscous | 64 | 10 | 15 (A) | 1.83e-01 | 11.5 | 1.006 | 1.01 | 1.000 |
| viscous | 64 | 20 | 15 (A) | 1.84e-01 | 11.6 | 1.002 | 1.00 | 1.000 |
| viscous | 256 | 5 | 14 (A) | 6.91e-02 | 45.2 | 1.005 | 1.02 | 1.000 |
| viscous | 256 | 10 | 14 (A) | 6.91e-02 | 45.2 | 1.001 | 1.00 | 1.000 |
| viscous | 256 | 20 | 14 (A) | 6.91e-02 | 45.2 | 1.000 | 1.00 | 1.000 |
| viscous | 576 | 5 | 9 (A) | 1.22e-02 | 65.6 | 1.002 | 1.00 | 1.000 |
| viscous | 576 | 10 | 9 (A) | 1.22e-02 | 65.6 | 1.000 | 1.00 | 1.000 |
| viscous | 576 | 20 | 9 (A) | 1.22e-02 | 65.6 | 1.000 | 1.00 | 1.000 |
| viscous | 1024 | 5 | 7 (A) | 1.11e-03 | 43.1 | 1.001 | 1.00 | 1.000 |
| viscous | 1024 | 10 | 7 (A) | 1.11e-03 | 43.1 | 1.000 | 1.00 | 1.000 |
| viscous | 1024 | 20 | 7 (A) | 1.11e-03 | 43.1 | 1.000 | 1.00 | 1.000 |
| gravity | 64 | 5 | 8 (A) | 2.21e-01 | 5.3 | 1.115 | 2.36 | 1.039 |
| gravity | 64 | 10 | 9 (A) | 2.22e-01 | 5.4 | 1.026 | 1.19 | 1.001 |
| gravity | 64 | 20 | 9 (A) | 2.22e-01 | 5.4 | 1.001 | 1.01 | 1.000 |
| gravity | 256 | 5 | 11 (A) | 1.44e-01 | 12.7 | 1.044 | 2.63 | 1.112 |
| gravity | 256 | 10 | 11 (A) | 1.44e-01 | 12.6 | 1.001 | 1.32 | 1.001 |
| gravity | 256 | 20 | 11 (A) | 1.44e-01 | 12.7 | 1.000 | 1.02 | 1.000 |
| gravity | 576 | 5 | 11 (A) | 8.89e-02 | 16.6 | 1.034 | 3.03 | 1.126 |
| gravity | 576 | 10 | 11 (A) | 8.90e-02 | 16.6 | 1.000 | 1.31 | 1.002 |
| gravity | 576 | 20 | 11 (A) | 8.90e-02 | 16.6 | 1.000 | 1.03 | 1.000 |
| gravity | 1024 | 5 | 8 (A) | 5.53e-02 | 22.0 | 1.023 | 2.59 | 1.017 |
| gravity | 1024 | 10 | 9 (A) | 5.61e-02 | 22.4 | 1.000 | 1.25 | 1.001 |
| gravity | 1024 | 20 | 9 (A) | 5.61e-02 | 22.4 | 1.000 | 1.04 | 1.000 |

A priori interior constants c₂/c₁ (the same for both cases):

| P | N/P = 5 | N/P = 10 | N/P = 20 [m = 3] |
|---|---|---|---|
| 64 | not computable | 1.0212 | 1.0000 [1.0255] |
| 256 | 3.6637 | 1.0111 | 1.0000 [1.0164] |
| 576 | 1.5540 | 1.0088 | 1.0000 [1.0141] |
| 1024 | 1.3977 | 1.0077 | 1.0000 [1.0130] |

### Item 2b: Buckley–Leverett, lifted sine × Chebyshev

| case | P | N/P | rule (class) | error at rule | error/δ_P | c₂/c₁ at rule | max c₂/c₁ | max ϱ_r |
|---|---|---|---|---|---|---|---|---|
| viscous | 64 | 5 | 19 (A) | 1.06e-01 | 16.5 | 1.153 | 1.85 | 1.060 |
| viscous | 64 | 10 | 17 (A) | 1.12e-01 | 17.5 | 1.008 | 1.12 | 1.002 |
| viscous | 64 | 20 | 17 (A) | 1.12e-01 | 17.5 | 1.000 | 1.01 | 1.000 |
| viscous | 256 | 5 | 9 (A) | 8.77e-03 | 21.1 | 1.016 | 1.15 | 1.008 |
| viscous | 256 | 10 | 9 (A) | 8.75e-03 | 21.0 | 1.000 | 1.00 | 1.000 |
| viscous | 256 | 20 | 9 (A) | 8.75e-03 | 21.0 | 1.000 | 1.00 | 1.000 |
| viscous | 576 | 5 | 6 (A) | 3.53e-04 | 3.5 | 1.002 | 1.02 | 1.000 |
| viscous | 576 | 10 | 6 (A) | 3.53e-04 | 3.5 | 1.000 | 1.00 | 1.000 |
| viscous | 576 | 20 | 6 (A) | 3.53e-04 | 3.5 | 1.000 | 1.00 | 1.000 |
| viscous | 1024 | 5 | 6 (A) | 9.41e-05 | 1.9 | 1.000 | 1.00 | 1.000 |
| viscous | 1024 | 10 | 6 (A) | 9.41e-05 | 1.9 | 1.000 | 1.00 | 1.000 |
| viscous | 1024 | 20 | 6 (A) | 9.41e-05 | 1.9 | 1.000 | 1.00 | 1.000 |
| gravity | 64 | 5 | 6 (A) | 1.05e-01 | 6.1 | 7.887 | 7.94 | 1.905 |
| gravity | 64 | 10 | 6 (A) | 1.32e-01 | 7.7 | 1.356 | 2.84 | 1.099 |
| gravity | 64 | 20 | 6 (A) | 1.32e-01 | 7.7 | 1.007 | 1.39 | 1.001 |
| gravity | 256 | 5 | 7 (A) | 6.63e-02 | 14.2 | 5.591 | 8.94 | 2.330 |
| gravity | 256 | 10 | 7 (A) | 8.31e-02 | 17.8 | 1.329 | 3.81 | 1.507 |
| gravity | 256 | 20 | 7 (A) | 8.28e-02 | 17.8 | 1.003 | 1.55 | 1.003 |
| gravity | 576 | 5 | 5 (A) | 3.56e-02 | 20.3 | 4.555 | 10.81 | 1.577 |
| gravity | 576 | 10 | 6 (A) | 3.85e-02 | 21.9 | 1.237 | 5.27 | 1.049 |
| gravity | 576 | 20 | 5 (A) | 3.91e-02 | 22.3 | 1.008 | 2.10 | 1.025 |
| gravity | 1024 | 5 | 7 (A) | 1.89e-02 | 26.2 | 5.143 | 6.78 | 1.681 |
| gravity | 1024 | 10 | 6 (A) | 2.01e-02 | 27.9 | 1.172 | 2.80 | 1.119 |
| gravity | 1024 | 20 | 5 (A) | 1.98e-02 | 27.6 | 1.006 | 1.95 | 1.031 |

### Item 3: a posteriori certificates

| problem | case | P | c₁ | c₂ | c₂/c₁ | c₂/c₁ (span A) | dropped | ϱ_r | flag |
|---|---|---|---|---|---|---|---|---|---|
| elasticity | compatible | 50 | 0.9319 | 1.4809 | 1.589 | 1.589 | 0 | 1.0019 |  |
| elasticity | compatible | 200 | 0.9542 | 1.6721 | 1.752 | 1.752 | 0 | 1.0007 |  |
| elasticity | compatible | 450 | 0.9097 | 2.1074 | 2.317 | 2.317 | 0 | 1.0003 |  |
| elasticity | compatible | 800 | 0.8252 | 2.4720 | 2.996 | 2.996 | 0 | 1.0003 |  |
| elasticity | compatible | 1,250 | 0.6776 | 2.8039 | 4.138 | 4.138 | 1 | 1.0003 | round-off |
| elasticity | paper | 50 | 0.9319 | 1.4809 | 1.589 | 1.589 | 1 | 2.4048 | round-off |
| elasticity | paper | 200 | 0.9542 | 1.6721 | 1.752 | 1.752 | 1 | 1.1404 | round-off |
| elasticity | paper | 450 | 0.9097 | 2.1074 | 2.317 | 2.317 | 1 | 0.5276 | round-off |
| elasticity | paper | 800 | 0.8252 | 2.4720 | 2.996 | 2.996 | 1 | 0.9475 | round-off |
| elasticity | paper | 1,250 | 0.6776 | 2.8039 | 4.138 | 4.138 | 1 | 1.5808 | round-off |
| elasticity | specified | 50 | 0.9319 | 1.4812 | 1.589 | 1.589 | 0 | 1.0047 |  |
| elasticity | specified | 200 | 0.9542 | 1.6721 | 1.752 | 1.752 | 0 | 1.0014 |  |
| elasticity | specified | 450 | 0.9097 | 2.1074 | 2.317 | 2.317 | 0 | 1.0007 |  |
| elasticity | specified | 800 | 0.8252 | 2.4720 | 2.996 | 2.996 | 0 | 1.0004 |  |
| elasticity | specified | 1,250 | 0.6776 | 2.8039 | 4.138 | 4.138 | 0 | 1.0003 |  |
| darcy | S1 | 3,169 | 0.2233 | 1.0430 | 4.670 | 1.698 | 0 | 1.0010 |  |
| darcy | S2 | 3,169 | 0.1899 | 1.0408 | 5.481 | 1.670 | 0 | 1.0041 |  |
| darcy | S3 | 3,169 | 0.3402 | 1.0492 | 3.084 | 1.726 | 0 | 1.0240 |  |
| darcy | SPE10 | 3,169 | 0.5789 | 1.0527 | 1.819 | 1.818 | 0 | 1.0006 |  |

ϱ_r's numerator and denominator are in the CSVs. The Section 6 test flags two cases as
round-off: the paper's elasticity solution, which lies in the span, and compatible at
P = 1,250, whose residual is 8.0e-12.

### Item 4: Burgers LiL-Q at larger sizes (Addendum 1)

Times are the medians of three clean runs on one exclusive node (48 threads).

| P | N | κ | rank | smallest error (k) | k_e (error) | k_r (class, error) | error at k = 60 | to k_e (s) | to k_r (s) |
|---|---|---|---|---|---|---|---|---|---|
| 625 | 5,956 | 6.82e+09 | 625 | 3.67e-05 (3) | none | 6 (A, 4.56e-05) | 4.56e-05 |  | 1.68 |
| 900 | 8,644 | 9.51e+11 | 900 | 1.11e-06 (4) | 3 (9.99e-06) | 6 (C, 1.11e-06) | 1.11e-06 | 1.67 | 3.33 |
| 1,024 | 9,860 | 2.07e+12 | 1024 | 4.36e-07 (37) | 3 (9.72e-06) | 6 (C, 4.36e-07) | 4.36e-07 | 2.44 | 4.78 |
| 1,225 | 11,836 | 6.99e+13 | 1225 | 3.52e-07 (23) | 3 (9.70e-06) | 6 (C, 3.52e-07) | 3.52e-07 | 3.47 | 6.97 |

The P = 625 control's paper pass took 1.13 s (L2).

## 4. Deviations and rules applied

### 4.1 L1's terminal row

Grace's evaluation of L1 reported one difference: ‖R‖_h at k = 4, 5.91211765423539e-05 in
P2-12 against 5.9121176541457504e-05 here. Every other value at k ≤ 4 was identical,
including eps_ref at k = 4.

The two numbers come from different code paths:
- **P2-12** stopped at k = 4, at the paper's target. Its k = 4 row is the tracker's terminal
  row, whose ‖R‖_h is √(the solver's loss).
- **The control** runs to K_max = 60. Its k = 4 row is a solve row, ‖A⁽⁴⁾β⁽⁴⁾ − f⁽⁴⁾‖.

The control's √(loss at iterate 4) is 5.91211765423539e-05, P2-12's bit for bit. L1 now
compares that terminal row with √(the control's loss). It reads only the logged CSVs, so it
was re-evaluated on Grace's files, at 86cc344, without a rerun. Grace's verdict is kept beside
the new one in `checks_item4.json`.

### 4.2 Rules decided in advance

- **Section 8.2.** No rule was needed:
  - the termination rule fired in every BL run, so no run needed the other guess;
  - no run reached its cap;
  - B3 level 2 passed the launch rule, 1.91 × (8 × 1,132.8 s + 1,218.4 s) = 5.45 h < 6.5 h,
    and the budget guard, 346.5 + 336 = 682.5 ≤ 850. It ran in 5 h 48 min.
- **B2 (K4, L3)** is judged at k = 1, as in Package 1, with the run maximum reported.
  The maxima were:
  - item 1: 5.4e-12;
  - item 2: 1.2e-12;
  - item 4: 2.7e-6, its passes going far past convergence.
- **Addendum 1, Section 5.** No rule was needed. Every size from P = 900 up reached 1.6e-5,
  every system was full rank, and L2 passed.

### 4.3 Readings of the instructions (in `DECISIONS.md`)

- **Darcy's blocks** are your reading: Darcy-x, Darcy-y and continuity at the 13,200 cell
  centres, weight 1, no boundary rows. The code agrees.
- **Zero blocks** are left out of both §6 systems:
  - elasticity's Dirichlet blocks on edges where the basis vanishes;
  - its lateral σ_xx blocks where the traction is zero (`specified` keeps them: zero in A,
    not in b);
  - 2b's lateral rows, which stay in the solve as in P2-9.
- **BL's lateral lines** keep P2-16's weights: the (n + 1)-point rule without t = 0, summing
  to 1 − w₀, not renormalized. The difference is 0.79% at P = 64, N/P = 5, and 0.2% or less
  elsewhere.
- **K3's pin weight** scales the squared row weight.
- **κ** is computed at the last iterate (items 1 and 2: SVD below P = 3,200, pivoted QR
  above). In item 4 the tracker's own rule computes it at every iterate (P ≤ 3,200). Neither
  changes the solve.

### 4.4 Records

- **The run jobs** did not write the full hardware and software record (`save_provenance`).
  Their logs give each job's node, threads and cores, and the preflight's versions.
  `hardware.json` and `environment.txt` list these per job.
- **A short job after the runs** wrote the full record: `p3_provenance.slurm`, job 20075205,
  187 s, 2.5 SU. It ran in the runs' class (a whole node, partition `medium`), their
  environment and their source lock (97e463d).
  - **Its record:** `provenance_cpu/`, with its script.
  - **The CPU:** Intel Xeon Gold 6248R, 3.00 GHz, 48 cores.
- **The stack:** torch 2.10.0+cu126, numpy 1.26.4, scipy 1.13.1, FlexiBLAS 3.4.4.

## 5. SU

Charged at Grace's rates (`su.csv`). Two rows are from `sacct` after the job ended: the last
report's (its tarball cannot hold its own charge) and the provenance job's.

| job | name | hours | SU/h | SU |
|---|---|---|---|---|
| 20067146 | lilq-p3-preflight | 0.15 | 24 | 3.6 |
| 20067147 | lilq-p3-k0 | 1.06 | 48 | 51.1 |
| 20067148 | lilq-p3-beltrami-small | 0.95 | 48 | 45.8 |
| 20067150 | lilq-p3-beltrami-b2 | 1.16 | 48 | 55.7 |
| 20067151 | lilq-p3-beltrami-b3-l1 | 2.90 | 48 | 139.4 |
| 20067152 | lilq-p3-bl | 0.98 | 24 | 23.6 |
| 20067155 | lilq-p3-affine | 0.36 | 48 | 17.5 |
| 20067157 | lilq-p3-burgers-large-p | 0.15 | 48 | 7.3 |
| 20067158 | lilq-p3-report | 0.10 | 24 | 2.5 |
| 20070402 | lilq-p3-beltrami-b3-l2 | 5.80 | 48 | 278.6 |
| 20070403 | lilq-p3-report | 0.09 | 24 | 2.1 |
| 20075205 | lilq-p3-provenance | 0.05 | 48 | 2.5 |
|  | **total** |  |  | **629.7** |

The plan was 1,164 SU requested, about 617 expected (`su_plan.csv`).

## Files

In `package3_results/` (`results/package3/` on the branch):
- `report.md`: this memo.
- `checks.json`: K0–K8, item 3's re-solves and L1–L4.
- `su.csv`; `su_plan.csv`; `sacct.txt`; `su_per_job.csv`.
- `DECISIONS_package3.md`: the Package 3 entries of `DECISIONS.md`.
- `environment.txt`, `hardware.json`; `provenance_cpu/` (the full record, Section 4.4).
- `provenance.json`: the runs' commit, this commit, and the commits between.
- `P3_1_beltrami_certified/`: a folder per run; `constants.csv`; `terminal.csv`.
- `P3_2a_bl_chebyshev_certified/`, `P3_2b_bl_lifted_sine_cheb/`: a folder per run;
  `terminal.csv`; 2a's `constants.csv`.
- `P3_3_affine_certificates/`: `elasticity.csv`, `darcy.csv`, a record per run.
- `P3_4_burgers_large_P/`: a folder per size, `terminal.csv`, `checks_item4.json`.
- `P3_checks/`: K0, K1, K2, K8.
- `slurm_logs/`: every job's log.
- `laptop_preview/`: the laptop rehearsals of 8 October, including the batch 0 K1 against
  your script.

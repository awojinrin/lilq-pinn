# Computational Package 5, items 1, 2 and 4: report

**To:** R. Younis. **From:** G. Awojinrin. **Date:** 14 November 2026.
**Refers to:** the Package 5 instructions (13 November 2026) and your reply to the cost check (14 November 2026).

**Code:** `awojinrin/lilq-pinn`, branch `v3-dev` at 57ba02a. These are the 8 commits after 5461f0d, the last commit of the cost check:

| commit | what | DECISIONS.md entry (Package 5, ...) |
|---|---|---|
| 5beed3e | the checks C1 to C3 against the delivered runs | the checks against the delivered runs |
| 55f0609 | your three answers; the Grace jobs | the advisor's reply to the cost check, and the Grace jobs |
| 189f07d | the Beltrami summary line names the run and its weight | (none: a log line) |
| 60c2d8c | a review before the runs: files written by jobs and processes running side by side | the advisor's reply to the cost check, and the Grace jobs |
| 8674498 | the assembly: item 1 beside your values; Beltrami beside the delivered rows | how the runs are set beside the advisor's values and the delivered Beltrami rows |
| 21ec859 | the assembly: the results folder | the assembly |
| 16b128b | an untimed warm-up before every one-pass timing; the re-time job | the one-pass timings get an untimed warm-up; one run re-timed |
| 57ba02a | the re-time in the assembly | the same entry |

Every run ran at 60c2d8c, under the source lock, except one re-timed run of item 2, which ran at 16b128b in its own folder (Section 5). 16b128b changes only what the clock of that run includes.

**Results:** `package5_results/`, on the branch `package5-results` of `awojinrin/lilq-pinn`, under `results/package5/`. The branch holds `v3-dev` at 57ba02a; its README says what it holds.

- **Expected values:** each table gives ours / expected where your files or your text give a value.
- **Times:** Grace's, on one whole node (48 threads).

## Summary

Items 1, 2 and 4 are done. Item 3 is next; its Bratu cost runs wait for your boundary weight.
- **Checks:** C1 to C3 pass (Section 2).
- **Cost:** 122.6 SU, against the approved 135 (up to 170).
- **Item 1:** in the norm of X the ratios are 1.0 to 3.5 at every weight, against L2 ratios up to 72 (Bratu) and 66 (Buckley-Leverett). Raising the weight from 10 to 10^4 lowers the L2 ratio on every scalar problem, and the X ratio everywhere except Burgers at P = 25 (1.03 to 1.06); at 10^6 some Buckley-Leverett runs are worse again. On Beltrami a larger weight raises the pressure ratio at both sizes and the velocity ratio at B1.
- **Item 2:** the three least-squares solvers reach LiL-Q's point in every run, in 6 or 7 Jacobian evaluations on Burgers and 8 to 41 on Poisson-Boltzmann. L-BFGS-B (scipy) reaches it on Burgers in 80 to 875 iterations; the PyTorch L-BFGS reaches it at p = 10 and 18 but not within 2,000 iterations at p = 26 and 30. Neither reaches it within 2,000 on Poisson-Boltzmann.
- **Item 4:** 4a agrees with your indicative values within 3.1%. 4b agrees with your file in 22 of 24 runs; the two others are at P = 800, σ = 2. At P = 1,600 the rule does not fire within 60 iterates in 4 of 9 runs. δ_P by gelsy and by gelsd differs by up to 94% on the random-feature spaces from P = 400.
- **One run was re-timed** (Section 5): its first time held a one-time cost of about 29 s.

**For your reply:**
1. **The Bratu boundary weight for item 3** (your reminder). The Bratu results are in Section 3.1. The ratio in H2 is lowest at weight 10^6 at every size: 1.10, 1.06 and 1.38 at P = 25, 100 and 225. Which weight should item 3 use?
2. **δ_P on the larger random-feature spaces** (Section 6.4). gelsy and gelsd at the same tolerance differ by up to 67% at P = 400, 87% at P = 800 and 94% at P = 1,600. gelsy gives the smaller δ_P, so the ratios to it are the larger. The tables use gelsy, as in Package 4. Should the manuscript keep gelsy, quote both, or leave out the ratio on these spaces?
3. **The 4b runs where the rule does not fire** (Section 6.3): all three draws at P = 1,600, σ = 2, and two of three at σ = 4. Their values are those of the 60th iterate, as returned. How should the manuscript show them?
4. **The H^{2,1} ratio of viscous Buckley-Leverett, tensor Chebyshev, P = 1,024** (Section 3.2). Against the p = 88 reference it passes your factor-of-three rule by a factor of 5.1. At weight 10^6 it was 1.38 against the p = 64 reference (laptop) and is 1.25 against p = 88. Is that settled enough, or should we add a reference at p = 96? That would be one 24-core job, about 15 SU by my estimate.

## 1. What ran

**Eight jobs on Grace, partition medium, and one re-time job:**

| Job | Class | Node | What |
|---|---|---|---|
| preflight | cpu-mem, 24 cores, 96 GB | c302 | versions, inputs, the test suite (861 passed), the points |
| ref-viscous, ref-gravity | cpu-mem | c302, c039 | the fine Buckley-Leverett references, p = 88 and 80, each case |
| item1 | cpu-mem | c080 | item 1's 96 runs, six side by side at 4 threads each |
| beltrami | cpu-mem | c074 | the six Beltrami runs |
| timed | timed-cpu, whole node, 48 threads | c080 | item 2; 4a's timings |
| item4 | cpu-mem | c074 | 4a and 2c logged, 4b, δ_P two ways |
| report | cpu | | summaries, the tarball |
| retime | timed-cpu, whole node | c057 | item 2, Burgers p = 10, the PyTorch L-BFGS, re-timed (Section 5) |

**The software:** numpy 1.26.4 and scipy 1.13.1; OpenBLAS 0.3.27 through FlexiBLAS 3.4.4; PyTorch 2.10.0. The timed nodes are Intel Xeon Gold 6248R (`environment.txt`, `hardware.json`).

**Your expected files** were not on the cluster. The comparisons with them were written at the assembly, on Grace's files; Grace's tables are kept; the rewritten ones agreed with them, δ_P to round-off of 1e-12 (`table_rerun_differences.json`).

## 2. Checks

Section 2 of your instructions defines checks for item 3 only. For items 1, 2 and 4 every changed driver, at its old settings, was compared with the delivered run of the same configuration (both Grace's).

| Check | Result | Numbers |
|---|---|---|
| C1: item 1 at weight 10 against Packages 2 and 3 | pass | 24 runs: ‖N‖, the linearized residual and the L2 error at every iterate up to the delivered stall within 5.1e-12 relative (cutoff 1e-8); the same rule steps |
| C2: item 2's LiL-Q against Package 4 | pass | 7 runs: the same number of solves; errors within 8.6e-6 relative, ‖N‖ within 7.3e-7 (cutoff 5e-3) |
| C3: δ_P of Package 4 by gelsy, recomputed | pass | p = 10 to 30: within 8.3e-4 relative (cutoff 5e-3) |

## 3. Item 1: errors in the norm of X, and the boundary weight

**References:**
- **Burgers:** the derivatives of the Cole-Hopf formula satisfy u_t + u u_x − ν u_xx = 0 on the evaluation grid to 5.4e-13 relative to the equation's terms (asked: 1e-8).
- **Buckley-Leverett:** fine LiL-Q solutions, lifted sine x Chebyshev, weight 10^4, at p = 88 with p = 80 as the neighbour (your answer 1). Both sizes stopped by the rule in both cases; no fallback was needed.

| case | rule step, p = 88 / 80 | L2 difference from the finite-difference reference, p = 88 / 80 | H^{2,1} difference, p = 88 against 80 | smallest δ_X of item 1 | runs left out |
|---|---|---|---|---|---|
| viscous | 5 / 5 | 3.7e-6 / 4.8e-6 | 2.8e-3 | 0.0145 (5.1 times) | none |
| gravity | 4 / 4 | 7.3e-6 / 1.2e-5 | 1.3e-2 | 0.192 (15 times) | none |

### 3.1 The runs

Each cell gives the rule step, then the ratio of the error to δ_P in L2 and in the norm of X (H2 for Bratu; H^{2,1} otherwise). δ_P does not depend on the weight. Every value per run (stall level, both errors, both δ_P, the error at the last iterate, the rank) is in `P5_1_weights/terminal.csv`.

**Bratu (norm of X: H2).** Each cell: rule step; L2 ratio / X ratio.

| P | δ_L2 | δ_X | weight 10 | 10^3 | 10^4 | 10^6 |
|---|---|---|---|---|---|---|
| 25 | 0.0015 | 0.0549 | 4; 71.8 / 1.83 | 4; 9.36 / 1.49 | 4; 8.68 / 1.11 | 4; 8.93 / 1.1 |
| 100 | 4.05e-5 | 0.0142 | 4; 16.1 / 2.63 | 3; 4.08 / 1.63 | 3; 3.9 / 1.53 | 3; 2.55 / 1.06 |
| 225 | 2.39e-6 | 0.00501 | 4; 15 / 2.64 | 4; 10.7 / 2.22 | 4; 6.34 / 1.84 | 3; 3.6 / 1.38 |

**Burgers, tensor Chebyshev (H^{2,1}).** Each cell: rule step; L2 ratio / X ratio.

| P | δ_L2 | δ_X | weight 10 | 10^3 | 10^4 | 10^6 |
|---|---|---|---|---|---|---|
| 25 | 0.268 | 0.9 | 4; 1.57 / 1.03 | 3; 1.35 / 1.06 | 3; 1.35 / 1.06 | 2; 1.34 / 1.06 |
| 100 | 0.0261 | 0.377 | 4; 3.7 / 1.1 | 4; 2.58 / 1.03 | 4; 2.57 / 1.03 | 4; 2.57 / 1.03 |
| 225 | 0.0063 | 0.167 | 4; 5.05 / 1.13 | 4; 3.13 / 1.06 | 4; 3.11 / 1.05 | 4; 3.11 / 1.05 |
| 400 | 0.000782 | 0.0406 | 4; 4.4 / 1.05 | 4; 2.62 / 1.03 | 4; 2.6 / 1.02 | 4; 2.6 / 1.02 |
| 625 | 0.000197 | 0.0145 | 4; 3.04 / 1.02 | 4; 2.01 / 1.01 | 4; 2 / 1.01 | 4; 2 / 1.01 |

**Buckley-Leverett viscous, tensor Chebyshev.** Each cell: rule step; L2 ratio / X ratio.

| P | δ_L2 | δ_X | weight 10 | 10^3 | 10^4 | 10^6 |
|---|---|---|---|---|---|---|
| 64 | 0.0159 | 0.625 | 14; 11.5 / 1.44 | 8; 4.87 / 1.11 | 8; 4.71 / 1.11 | 7; 4.56 / 1.11 |
| 256 | 0.00153 | 0.206 | 13; 45.2 / 2.16 | 8; 11.2 / 1.16 | 8; 10.7 / 1.15 | 8; 10.6 / 1.18 |
| 576 | 0.000186 | 0.054 | 8; 65.6 / 1.88 | 6; 10.6 / 1.07 | 6; 10.1 / 1.05 | 6; 10 / 1.15 |
| 1024 | 2.59e-5 | 0.0145 | 6; 43.1 / 1.52 | 5; 6.05 / 1.18 | 5; 5.7 / 1.08 | 5; 5.62 / 1.25 |

**Buckley-Leverett gravity, tensor Chebyshev.** Each cell: rule step; L2 ratio / X ratio.

| P | δ_L2 | δ_X | weight 10 | 10^3 | 10^4 | 10^6 |
|---|---|---|---|---|---|---|
| 64 | 0.0414 | 0.928 | 8; 5.36 / 1.11 | 5; 2.09 / 1.08 | 4; 1.76 / 1.08 | 2; 1.63 / 1.07 |
| 256 | 0.0114 | 0.735 | 10; 12.6 / 1.44 | 6; 3.39 / 1.06 | 5; 2.31 / 1.1 | 2; 5.35 / 1.57 |
| 576 | 0.00536 | 0.571 | 10; 16.6 / 1.67 | 6; 3.66 / 1.07 | 5; 2.25 / 1.07 | 3; 2.01 / 1.97 |
| 1024 | 0.00251 | 0.416 | 8; 22.4 / 2.07 | 5; 3.76 / 1.09 | 5; 1.92 / 1.04 | 4; 1.68 / 1.67 |

**Buckley-Leverett viscous, lifted sine x Chebyshev.** Each cell: rule step; L2 ratio / X ratio.

| P | δ_L2 | δ_X | weight 10 | 10^3 | 10^4 | 10^6 |
|---|---|---|---|---|---|---|
| 64 | 0.0064 | 0.369 | 16; 17.5 / 1.79 | 7; 2.2 / 1.1 | 6; 2.18 / 1.1 | 3; 8.68 / 1.34 |
| 256 | 0.000416 | 0.0863 | 8; 21 / 1.45 | 5; 1.5 / 1.03 | 5; 1.46 / 1.03 | 4; 1.46 / 1.03 |
| 576 | 0.000102 | 0.0361 | 5; 3.46 / 1.32 | 5; 1.01 / 1 | 5; 1.01 / 1 | 4; 1.01 / 1 |
| 1024 | 4.84e-5 | 0.0224 | 5; 1.95 / 1.32 | 5; 1 / 1 | 5; 1 / 1 | 4; 1 / 1 |

**Buckley-Leverett gravity, lifted sine x Chebyshev.** Each cell: rule step; L2 ratio / X ratio.

| P | δ_L2 | δ_X | weight 10 | 10^3 | 10^4 | 10^6 |
|---|---|---|---|---|---|---|
| 64 | 0.0171 | 0.766 | 5; 7.75 / 1.27 | 3; 2.42 / 1.03 | 3; 2.07 / 1.02 | 2; 2.23 / 1.03 |
| 256 | 0.00466 | 0.509 | 6; 17.8 / 1.8 | 3; 3.09 / 1.08 | 3; 2.12 / 1.02 | 2; 2.74 / 1.02 |
| 576 | 0.00175 | 0.32 | 5; 21.9 / 2.49 | 4; 3.3 / 1.04 | 3; 1.42 / 1 | 2; 1.55 / 1 |
| 1024 | 0.00072 | 0.192 | 5; 27.9 / 3.51 | 4; 3.4 / 1.3 | 3; 1.3 / 1.02 | 3; 1.12 / 1.01 |

**What the runs show:**
- **Bratu:** the X ratio falls with the weight at every size; weight 10^6 gives 1.10, 1.06 and 1.38. The L2 ratio falls from 72, 16 and 15 to 8.9, 2.6 and 3.6.
- **Burgers:** the X ratio is 1.01 to 1.13 at every weight; from 10^3 on, nothing changes beyond the second figure.
- **Buckley-Leverett, lifted sine x Chebyshev:** from weight 10^3 the X ratio is 1.0 to 1.34. In the viscous case from P = 576 the error equals δ_P in both norms. At P = 64, weight 10^6 is worse than 10^4 in L2 (8.68 against 2.18, viscous).
- **Buckley-Leverett, tensor Chebyshev:** weights 10^3 and 10^4 give X ratios 1.04 to 1.18. At 10^6 the gravity case is worse (1.57, 1.97, 1.67 at P = 256 to 1,024), with the rule firing at steps 2 to 4.
- The stall level is the weighted residual, so it grows with the weight and is not comparable across weights.

### 3.2 Beside your values

Your values come from Gauss-Legendre grids, so agreement to one figure is expected.

| problem | p | P | weight | solves | L2 ratio | X ratio | δ_L2 | δ_X |
|---|---|---|---|---|---|---|---|---|
| Bratu | 5 | 25 | 10 | 5 / 5 | 71.8 / 74.1 | 1.83 / 1.83 | 0.0015 / 0.00145 | 0.0549 / 0.0548 |
| Bratu | 5 | 25 | 10^3 | 5 / 5 | 9.36 / 9.64 | 1.49 / 1.49 | 0.0015 / 0.00145 | 0.0549 / 0.0548 |
| Bratu | 5 | 25 | 10^6 | 5 / 5 | 8.93 / 9.22 | 1.1 / 1.1 | 0.0015 / 0.00145 | 0.0549 / 0.0548 |
| Bratu | 10 | 100 | 10 | 5 / 5 | 16.1 / 16.7 | 2.63 / 2.67 | 4.05e-5 / 3.72e-5 | 0.0142 / 0.0139 |
| Bratu | 10 | 100 | 10^3 | 4 / 4 | 4.08 / 4.04 | 1.63 / 1.65 | 4.05e-5 / 3.72e-5 | 0.0142 / 0.0139 |
| Bratu | 10 | 100 | 10^4 | 4 / 4 | 3.9 / 3.9 | 1.53 / 1.54 | 4.05e-5 / 3.72e-5 | 0.0142 / 0.0139 |
| Bratu | 10 | 100 | 10^6 | 4 / 4 | 2.55 / 2.75 | 1.06 / 1.06 | 4.05e-5 / 3.72e-5 | 0.0142 / 0.0139 |
| Bratu | 15 | 225 | 10 | 5 / 5 | 15 / 15.1 | 2.64 / 2.86 | 2.39e-6 / 2.03e-6 | 0.00501 / 0.00452 |
| Bratu | 15 | 225 | 10^3 | 5 / 5 | 10.7 / 10.8 | 2.22 / 2.39 | 2.39e-6 / 2.03e-6 | 0.00501 / 0.00452 |
| Bratu | 15 | 225 | 10^6 | 4 / 4 | 3.6 / 3.44 | 1.38 / 1.43 | 2.39e-6 / 2.03e-6 | 0.00501 / 0.00452 |
| BL viscous, Chebyshev | 8 | 64 | 10 | 15 / 12 | 11.5 / 9.09 | 1.44 / 1.3 | 0.0159 / 0.0158 | 0.625 / 0.625 |
| BL viscous, Chebyshev | 8 | 64 | 10^3 | 9 / 9 | 4.87 / 4.81 | 1.11 / 1.11 | 0.0159 / 0.0158 | 0.625 / 0.625 |
| BL viscous, Chebyshev | 16 | 256 | 10 | 14 / 11 | 45.2 / 27.6 | 2.16 / 1.59 | 0.00153 / 0.00151 | 0.206 / 0.207 |
| BL viscous, Chebyshev | 16 | 256 | 10^3 | 9 / 9 | 11.2 / 11 | 1.16 / 1.15 | 0.00153 / 0.00151 | 0.206 / 0.207 |
| BL viscous, Chebyshev | 24 | 576 | 10 | 9 / 8 | 65.6 / 32.5 | 1.88 / 1.3 | 0.000186 / 0.000184 | 0.054 / 0.0561 |
| BL viscous, Chebyshev | 24 | 576 | 10^3 | 7 / 7 | 10.6 / 10.4 | 1.07 / 1.06 | 0.000186 / 0.000184 | 0.054 / 0.0561 |
| BL viscous, Chebyshev | 32 | 1024 | 10 | 7 / 7 | 43.1 / 19.3 | 1.52 / 1.34 | 2.59e-5 / 2.51e-5 | 0.0145 / 0.0181 |
| BL viscous, Chebyshev | 32 | 1024 | 10^3 | 6 / 6 | 6.05 / 5.61 | 1.18 / 1.19 | 2.59e-5 / 2.51e-5 | 0.0145 / 0.0181 |

- **Bratu** agrees within 8% in both ratios. δ_P differs by up to 18% in L2 and 11% in H2 at P = 225, from the different grids.
- **Buckley-Leverett at weight 10^3** agrees within 8%.
- **Buckley-Leverett at weight 10** differs by up to 2.2 times in L2 and 1.44 in H^{2,1}, with up to 3 more solves. This is the weight split (your answer 3): our runs keep Package 3's rule (each line's squared weight λ x CC weights x |line| / |∂Ω|), and `ref_weight_bl.py` gives the initial line λ and each side line λ/2. **The check** (laptop, the same code with only the split replaced by yours; against the p = 64 reference): at weight 10, L2 / H^{2,1} ratios 8.97 / 1.30, 27.1 / 1.60, 32.0 / 1.33, 19.0 / 1.42 at p = 8, 16, 24, 32, against your 9.1 / 1.3, 27.6 / 1.59, 32.5 / 1.3, 19.3 / 1.34.
- **P = 1,024, viscous, tensor Chebyshev:** δ_X is 0.0145 against your 0.0181, so your p = 40 reference (about 1e-2 in H^{2,1}, as your text says) would leave this row out; ours does not (p = 88: 2.8e-3).

### 3.3 Beltrami

The delivered rows of `P3_1_beltrami_certified/terminal.csv` (weight 10) beside the Package 5 runs. Errors in L2 at the returned iterate as Package 3 defines it (the first with a relative step below 1e-9), divided by the delivered δ_P of each field. The combined error at the rule step differs from that at the returned iterate by less than 3e-6 relative in every run (`beside_delivered.csv`).

| size | P | weight | run | rule step | stall level | returned iterate | error u | u/δ | v/δ | w/δ | error p (pin gauge) | p/δ | combined/δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 1393 | 10 | delivered | 4 | 0.0937 | 7 | 0.0177 | 2.98 | 2.98 | 2.98 | 0.17 | 41.1 | 21.1 |
| B1 | 1393 | 10^3 | Package 5 | 4 | 0.351 | 6 | 0.0253 | 4.27 | 4.27 | 4.27 | 0.201 | 48.6 | 29.8 |
| B1 | 1393 | 10^4 | Package 5 | 3 | 0.993 | 6 | 0.0268 | 4.53 | 4.53 | 4.53 | 0.213 | 51.5 | 31.3 |
| B1 | 1393 | 10^6 | Package 5 | 3 | 9.77 | 6 | 0.027 | 4.56 | 4.56 | 4.56 | 0.214 | 51.9 | 31.5 |
| B2 | 3171 | 10 | delivered | 4 | 0.0173 | 6 | 0.00296 | 5.26 | 5.26 | 5.26 | 0.0726 | 146 | 37 |
| B2 | 3171 | 10^3 | Package 5 | 4 | 0.0405 | 5 | 0.00234 | 4.15 | 4.15 | 4.15 | 0.0845 | 169 | 35.8 |
| B2 | 3171 | 10^4 | Package 5 | 3 | 0.091 | 5 | 0.00238 | 4.23 | 4.23 | 4.23 | 0.0867 | 174 | 36.5 |
| B2 | 3171 | 10^6 | Package 5 | 3 | 0.852 | 5 | 0.00239 | 4.24 | 4.24 | 4.24 | 0.087 | 174 | 36.6 |

- **B1:** a larger weight makes every ratio worse: u from 2.98 to 4.27 to 4.56; p (pin gauge) from 41 to 49 to 52.
- **B2:** u improves from 5.26 to about 4.2; p (pin gauge) worsens from 146 to 169 to 174; the combined ratio barely moves (37 to 36).
- The ratios of u, v and w agree to the printed figures, as in the delivered rows.

## 4. Item 2: LiL-N by nonlinear least squares

Count: LiL-Q solves; Jacobian evaluations for trf, lm and dogbox; iterations for L-BFGS. Status: the least-squares solvers' stopping test. Error: L2 for Burgers, H2 for Poisson-Boltzmann (PB), with the error over δ_P in the same norm. Your file has no rows at Burgers p = 30 and PB p = 22, and none for the PyTorch L-BFGS.

| problem | p | solver | count | status | time (s) | residual ‖N‖ | error | error/δ |
|---|---|---|---|---|---|---|---|---|
| Burgers | 10 | LiL-Q | 5 / 5 | rule fired | 0.0467 | 0.0323 | 0.00399 / 0.00399 | 2.01 |
| Burgers | 10 | trf | 7 / 7 | ftol | 0.137 | 0.0323 | 0.00399 / 0.00399 | 2.01 |
| Burgers | 10 | lm | 6 / 6 | ftol | 0.0497 | 0.0323 | 0.00399 / 0.00399 | 2.01 |
| Burgers | 10 | dogbox | 7 / 7 | ftol | 0.1 | 0.0323 | 0.00399 / 0.00399 | 2.01 |
| Burgers | 10 | L-BFGS (PyTorch) | 138 | stall (f1) | 0.372 (re-timed) | 0.0323 | 0.00399 | 2.01 |
| Burgers | 10 | L-BFGS-B (scipy) | 80 / 77 | convergence: rel reduction of f <= factr*epsmch | 0.0913 | 0.0323 | 0.00399 / 0.00399 | 2.01 |
| Burgers | 18 | LiL-Q | 6 / 6 | rule fired | 0.426 | 0.00111 | 2.55e-5 / 2.55e-5 | 1.04 |
| Burgers | 18 | trf | 6 / 6 | xtol | 0.827 | 0.00111 | 2.55e-5 / 2.55e-5 | 1.04 |
| Burgers | 18 | lm | 6 / 6 | ftol and xtol | 1.47 | 0.00111 | 2.55e-5 / 2.55e-5 | 1.04 |
| Burgers | 18 | dogbox | 6 / 6 | xtol | 0.576 | 0.00111 | 2.55e-5 / 2.55e-5 | 1.04 |
| Burgers | 18 | L-BFGS (PyTorch) | 856 | stall (f1) | 12.1 | 0.00111 | 2.55e-5 | 1.04 |
| Burgers | 18 | L-BFGS-B (scipy) | 289 / 274 | abnormal termination in lnsrch | 4.3 | 0.00111 | 2.55e-5 / 2.55e-5 | 1.04 |
| Burgers | 26 | LiL-Q | 6 / 6 | rule fired | 1.37 | 2.78e-5 | 3.15e-7 / 3.15e-7 | 1 |
| Burgers | 26 | trf | 6 / 6 | gtol | 2.73 | 2.78e-5 | 3.15e-7 / 3.15e-7 | 1 |
| Burgers | 26 | lm | 6 / 6 | xtol | 13 | 2.78e-5 | 3.15e-7 / 3.15e-7 | 1 |
| Burgers | 26 | dogbox | 6 / 6 | gtol | 1.95 | 2.78e-5 | 3.15e-7 / 3.15e-7 | 1 |
| Burgers | 26 | L-BFGS (PyTorch) | 2000 | iteration cap | 160 | 2.78e-5 | 3.33e-7 | 1.06 |
| Burgers | 26 | L-BFGS-B (scipy) | 589 / 594 | convergence: rel reduction of f <= factr*epsmch | 43 | 2.78e-5 | 3.15e-7 / 3.15e-7 | 1 |
| Burgers | 30 | LiL-Q | 6 | rule fired | 2.68 | 4.15e-6 | 3.6e-8 | 1 |
| Burgers | 30 | trf | 6 | gtol | 4.65 | 4.15e-6 | 3.6e-8 | 1 |
| Burgers | 30 | lm | 6 | xtol | 31.6 | 4.15e-6 | 3.6e-8 | 1 |
| Burgers | 30 | dogbox | 6 | gtol | 3.48 | 4.15e-6 | 3.6e-8 | 1 |
| Burgers | 30 | L-BFGS (PyTorch) | 2000 | iteration cap | 459 | 4.25e-6 | 7.51e-7 | 20.8 |
| Burgers | 30 | L-BFGS-B (scipy) | 875 | convergence: rel reduction of f <= factr*epsmch | 116 | 4.15e-6 | 3.6e-8 | 1 |
| PB | 10 | LiL-Q | 3 / 3 | rule fired | 0.0277 | 2.69 | 0.37 / 0.37 | 1.45 |
| PB | 10 | trf | 14 / 13 | ftol | 0.287 | 2.69 | 0.37 / 0.37 | 1.45 |
| PB | 10 | lm | 8 / 12 | gtol | 0.0655 | 2.69 | 0.37 / 0.37 | 1.45 |
| PB | 10 | dogbox | 14 / 14 | ftol | 0.205 | 2.69 | 0.37 / 0.37 | 1.45 |
| PB | 10 | L-BFGS (PyTorch) | 2000 | iteration cap | 6.22 | 2.71 | 0.364 | 1.42 |
| PB | 10 | L-BFGS-B (scipy) | 2000 / 2000 | stop: total no. of iterations reached limit | 3.45 | 2.71 | 0.364 / 0.364 | 1.42 |
| PB | 18 | LiL-Q | 4 / 4 | rule fired | 0.27 | 0.0469 | 0.00818 / 0.00818 | 1.81 |
| PB | 18 | trf | 26 / 26 | xtol | 3.28 | 0.0469 | 0.00818 / 0.00818 | 1.81 |
| PB | 18 | lm | 17 / 18 | xtol | 4.26 | 0.0469 | 0.00818 / 0.00818 | 1.81 |
| PB | 18 | dogbox | 26 / 26 | xtol | 2.27 | 0.0469 | 0.00818 / 0.00818 | 1.81 |
| PB | 18 | L-BFGS (PyTorch) | 2000 | iteration cap | 24.8 | 0.054 | 0.00759 | 1.68 |
| PB | 18 | L-BFGS-B (scipy) | 2000 / 2000 | stop: total no. of iterations reached limit | 20.9 | 0.054 | 0.00765 / 0.00767 | 1.69 |
| PB | 22 | LiL-Q | 5 | rule fired | 0.703 | 0.00318 | 0.000754 | 2.32 |
| PB | 22 | trf | 31 | xtol | 7.68 | 0.00318 | 0.000754 | 2.32 |
| PB | 22 | lm | 22 | xtol | 18.4 | 0.00318 | 0.000754 | 2.32 |
| PB | 22 | dogbox | 41 | xtol | 7.53 | 0.00318 | 0.000754 | 2.32 |
| PB | 22 | L-BFGS (PyTorch) | 2000 | iteration cap | 51.3 | 0.00683 | 0.00198 | 6.08 |
| PB | 22 | L-BFGS-B (scipy) | 2000 | stop: total no. of iterations reached limit | 45 | 0.00681 | 0.00198 | 6.08 |

- **The three least-squares solvers** return LiL-Q's error and residual in every run. Against your file the errors agree within 2.4e-10 relative on Burgers and 2.0e-7 on PB (expected: three figures); at Burgers p = 30 and PB p = 22 they equal LiL-Q's to the printed figures. Jacobian evaluations: 6 or 7 on Burgers, 8 to 41 on PB; your file has 6 to 7 and 12 to 26. At PB p = 10, lm takes 8 against your 12.
- **L-BFGS-B (scipy)** reaches LiL-Q's point on Burgers in 80, 289, 589 and 875 iterations (your 77, 274, 594), at p = 18 with scipy's "abnormal termination in line search", and on PB stops at 2,000 iterations short of it, as your file says.
- **The PyTorch L-BFGS** of the four-method runs reaches the point at p = 10 and 18 (138 and 856 iterations) and stops at the 2,000 cap at p = 26 and 30; at p = 30 its error is 20.8 times δ_P. On PB it stops at 2,000 in every run.
- **Times:** LiL-Q takes 0.03 to 2.7 s; the least-squares solvers 1 to 26 times LiL-Q's time; L-BFGS 2 to 225 times. LiL-Q and the least-squares solvers by the full protocol, L-BFGS by one logged pass (your answer 2).

## 5. The re-timed run

- **What happened:** in the timed job, Burgers p = 10 with the PyTorch L-BFGS took 30.9 s (224 ms per iteration). On the laptop it takes 1.4 s; every other Grace run of item 2 took 1 to 2 times its laptop time. It was the job's first PyTorch L-BFGS run, and the one-pass timings had no warm-up, so a one-time cost fell inside its only timed pass. I did not catch this before the runs; I should have.
- **What its cause is:** not known. It is not the import of PyTorch, which is loaded before any clock starts.
- **The fix** (16b128b): every solver timed by one pass first runs 5 untimed iterations of the same solve. Item 3's one-pass timings use the same warm-up.
- **The re-time** (job 20534654, node c057, 3.9 SU): 0.372 s (2.7 ms per iteration). The same 138 iterations and identical coefficients. The warm-up itself took 59.7 s for its 5 iterations.
- **Where:** `package5_retime/` (as Grace wrote it) and `P5_2_nlls/retime.csv`. Grace's item 2 tables keep the 30.9 s; the table above gives the re-timed value.

## 6. Item 4: the irregular domain

### 6.1 4a: the Chebyshev space at p = 34, 38, 42

Ours / your indicative values (seed 0). Median of three timed passes.

| p | P | seed | rule step | stall level | error L2 | error H2 | L2/δ | H2/δ | rank (gelsy) | median time (s) | spread | solves timed / logged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 34 | 1156 | 0 | 4 / 4 | 4.15e-8 / 4.1e-8 | 1.75e-10 | 1.67e-8 / 1.7e-8 | 7.28 | 3.65 | 1097 / 1098 | 4.71 | 0.7% | 5 / 5 |
| 34 | 1156 | 1 | 4 | 4.44e-8 | 1.74e-10 | 1.46e-8 | 7.24 | 3.19 | 1099 | 4.67 | 0.3% | 5 / 5 |
| 34 | 1156 | 2 | 4 | 4.29e-8 | 1.71e-10 | 1.54e-8 | 7.14 | 3.35 | 1098 | 4.69 | 0.8% | 5 / 5 |
| 38 | 1444 | 0 | 4 / 4 | 6.57e-10 / 6.6e-10 | 3.04e-12 | 1.79e-10 / 1.8e-10 | 9.24 | 2.62 | 1340 / 1340 | 8.43 | 10.3% | 5 / 5 |
| 38 | 1444 | 1 | 4 | 6.68e-10 | 2.87e-12 | 2.59e-10 | 8.71 | 3.78 | 1336 | 8.36 | 2.7% | 5 / 5 |
| 38 | 1444 | 2 | 4 | 6.47e-10 | 2.94e-12 | 2.04e-10 | 8.93 | 2.97 | 1338 | 8.43 | 0.3% | 5 / 5 |
| 42 | 1764 | 0 | 5 / 4 | 1.52e-11 / 1.5e-11 | 6.46e-14 | 5.15e-12 / 5e-12 | 9.98 | 3.24 | 1588 / 1594 | 19.5 | 3.7% | 6 / 6 |
| 42 | 1764 | 1 | 6 | 1.5e-11 | 6e-14 | 6.55e-12 | 9.28 | 4.12 | 1588 | 26.3 | 2.2% | 8 / 7 |
| 42 | 1764 | 2 | 5 | 1.54e-11 | 6.15e-14 | 6.53e-12 | 9.51 | 4.11 | 1589 | 19.6 | 6.3% | 6 / 6 |

- Stall level, H2 error and rank agree with your indicative values within 3.1%; the rule steps are 4, 4 and 5 (your answer: 4, 4, 5).
- **p = 42, seed 1:** the logged run returned iterate 7 (rule step 6); all three timed passes took 8 solves. Its time is therefore for 8 solves. The logged run (24-core job, 4 threads) and the timed passes (whole node, 48 threads) differ in round-off, and on these rank-deficient systems the rule fires one step later. Reported as measured (Section 7 of your instructions); settings unchanged.
- **p = 38, seed 0:** the three timings differ by 10.3%; the median is reported.

### 6.2 2c at p = 34, 38, 42

N = P/2 rows, the minimum-norm Newton step by gelsd, a budget of 30 steps. There are no expected values.

| p | P | seed | N | status | steps | error L2 | error H2 |
|---|---|---|---|---|---|---|---|
| 34 | 1156 | 0 | 578 | budget | 30 | 19.1 | 504 |
| 34 | 1156 | 1 | 578 | budget | 30 | 2.38 | 114 |
| 34 | 1156 | 2 | 578 | budget | 30 | 28.5 | 773 |
| 38 | 1444 | 0 | 722 | budget | 30 | 19.5 | 2.17e3 |
| 38 | 1444 | 1 | 722 | budget | 30 | 10.2 | 860 |
| 38 | 1444 | 2 | 722 | budget | 30 | 46.4 | 2.64e3 |
| 42 | 1764 | 0 | 882 | budget | 30 | 40.8 | 3.11e3 |
| 42 | 1764 | 1 | 882 | diverged | 0 | 1 | 1 |
| 42 | 1764 | 2 | 882 | budget | 30 | 36.2 | 5.26e3 |

Eight runs reach the budget with H2 errors of 10^2 to 5e3; one diverges at its first step (its errors are those of the zero start).

### 6.3 4b: random features

Ours / expected, for P up to 800 and draws 0 and 1.

| P | σ | draw | solves | rank (gelsy) | stall level | error H2 | H2/δ |
|---|---|---|---|---|---|---|---|
| 100 | 2 | 0 | 4 / 4 | 100 / 100 | 0.451 / 0.451 | 0.0648 / 0.0648 | 1.37 / 1.37 |
| 100 | 2 | 1 | 4 / 4 | 100 / 100 | 0.495 / 0.495 | 0.0829 / 0.0829 | 1.41 / 1.41 |
| 100 | 4 | 0 | 4 / 4 | 100 / 100 | 1.11 / 1.11 | 0.129 / 0.129 | 1.15 / 1.15 |
| 100 | 4 | 1 | 4 / 4 | 100 / 100 | 1.35 / 1.35 | 0.159 / 0.159 | 1.15 / 1.15 |
| 100 | 8 | 0 | 3 / 3 | 100 / 100 | 5.33 / 5.33 | 0.507 / 0.507 | 1.13 / 1.13 |
| 100 | 8 | 1 | 3 / 3 | 100 / 100 | 5.42 / 5.42 | 0.513 / 0.513 | 1.17 / 1.17 |
| 200 | 2 | 0 | 4 / 4 | 200 / 200 | 0.0224 / 0.0224 | 0.00544 / 0.00544 | 2.22 / 2.22 |
| 200 | 2 | 1 | 4 / 4 | 200 / 200 | 0.0163 / 0.0163 | 0.00428 / 0.00428 | 2.03 / 2.03 |
| 200 | 4 | 0 | 4 / 4 | 200 / 200 | 0.078 / 0.078 | 0.0177 / 0.0177 | 1.61 / 1.61 |
| 200 | 4 | 1 | 4 / 4 | 200 / 200 | 0.168 / 0.168 | 0.0262 / 0.0262 | 1.42 / 1.42 |
| 200 | 8 | 0 | 4 / 4 | 200 / 200 | 1.37 / 1.37 | 0.151 / 0.151 | 1.12 / 1.12 |
| 200 | 8 | 1 | 3 / 3 | 200 / 200 | 1.86 / 1.86 | 0.216 / 0.216 | 1.19 / 1.19 |
| 400 | 2 | 0 | 5 / 5 | 398 / 398 | 1.52e-5 / 1.52e-5 | 5.7e-6 / 5.7e-6 | 3.35 / 3.35 |
| 400 | 2 | 1 | 5 / 5 | 400 / 400 | 1.8e-5 / 1.8e-5 | 7.46e-6 / 7.49e-6 | 3.4 / 3.42 |
| 400 | 4 | 0 | 4 / 4 | 400 / 400 | 0.00224 / 0.00224 | 0.000464 / 0.000464 | 1.87 / 1.87 |
| 400 | 4 | 1 | 4 / 4 | 400 / 400 | 0.00204 / 0.00204 | 0.000403 / 0.000403 | 1.75 / 1.75 |
| 400 | 8 | 0 | 4 / 4 | 400 / 400 | 0.13 / 0.13 | 0.0277 / 0.0277 | 1.88 / 1.88 |
| 400 | 8 | 1 | 4 / 4 | 400 / 400 | 0.183 / 0.183 | 0.0356 / 0.0356 | 1.75 / 1.75 |
| 800 | 2 | 0 | 5 / 6 | 693 / 693 | 9.72e-9 / 9.72e-9 | 7.27e-9 / 7.25e-9 | 13.6 / 13.6 |
| 800 | 2 | 1 | 6 / 10 | 695 / 695 | 1.64e-8 / 1.63e-8 | 8.84e-9 / 8.69e-9 | 8.38 / 8.27 |
| 800 | 4 | 0 | 5 / 5 | 792 / 792 | 3.52e-6 / 3.51e-6 | 1.68e-6 / 1.68e-6 | 4.51 / 4.51 |
| 800 | 4 | 1 | 5 / 5 | 793 / 793 | 3.62e-6 / 3.62e-6 | 1.59e-6 / 1.59e-6 | 4.17 / 4.16 |
| 800 | 8 | 0 | 4 / 4 | 800 / 800 | 0.00172 / 0.00172 | 0.000605 / 0.000605 | 2.89 / 2.89 |
| 800 | 8 | 1 | 4 / 4 | 800 / 800 | 0.0019 / 0.0019 | 0.000571 / 0.00057 | 2.57 / 2.57 |

- **22 of 24 runs** agree: the same solves and rank, errors within 0.4% (within 1e-7 below P = 400).
- **P = 800, σ = 2** (your answer: side by side): 5 and 6 solves against 6 and 10. The ranks are the same; H2 errors differ by 0.3% and 1.7%.

Draw 2, and P = 1,600 (no expected values):

| P | σ | draw | rule step | returned iterate | rank (gelsy) | stall level | error H2 | H2/δ |
|---|---|---|---|---|---|---|---|---|
| 100 | 2 | 2 | 3 | 4 | 100 | 0.806 | 0.115 | 1.3 |
| 100 | 4 | 2 | 2 | 3 | 100 | 2.64 | 0.334 | 1.19 |
| 100 | 8 | 2 | 2 | 3 | 100 | 6.91 | 0.63 | 1.09 |
| 200 | 2 | 2 | 3 | 4 | 200 | 0.0103 | 0.00322 | 2.25 |
| 200 | 4 | 2 | 3 | 4 | 200 | 0.109 | 0.0152 | 1.35 |
| 200 | 8 | 2 | 3 | 4 | 200 | 1.6 | 0.204 | 1.24 |
| 400 | 2 | 2 | 4 | 5 | 400 | 1.48e-5 | 6e-6 | 3.35 |
| 400 | 4 | 2 | 3 | 4 | 400 | 0.000954 | 0.000281 | 2.41 |
| 400 | 8 | 2 | 3 | 4 | 400 | 0.063 | 0.0101 | 1.47 |
| 800 | 2 | 2 | 8 | 9 | 694 | 1.19e-8 | 5.36e-9 | 8.76 |
| 800 | 4 | 2 | 4 | 5 | 794 | 3.03e-6 | 1.55e-6 | 4.69 |
| 800 | 8 | 2 | 3 | 4 | 800 | 0.00133 | 0.000384 | 2.46 |
| 1600 | 2 | 0 | did not fire | 60 (the last) | 968 | 4.85e-11 | 2.75e-11 | 33.4 |
| 1600 | 2 | 1 | did not fire | 60 (the last) | 984 | 8.09e-11 | 4.3e-11 | 44.7 |
| 1600 | 2 | 2 | did not fire | 60 (the last) | 977 | 4.64e-11 | 4.32e-11 | 59.9 |
| 1600 | 4 | 0 | 4 | 5 | 1465 | 6.79e-10 | 4.7e-10 | 13.1 |
| 1600 | 4 | 1 | did not fire | 60 (the last) | 1462 | 1.4e-9 | 9.78e-10 | 11.3 |
| 1600 | 4 | 2 | did not fire | 60 (the last) | 1475 | 1.06e-9 | 5.24e-10 | 6.42 |
| 1600 | 8 | 0 | 4 | 5 | 1561 | 5.37e-6 | 3.36e-6 | 6.62 |
| 1600 | 8 | 1 | 4 | 5 | 1554 | 1.05e-5 | 4.2e-6 | 4.38 |
| 1600 | 8 | 2 | 4 | 5 | 1558 | 9.13e-6 | 4.32e-6 | 4.84 |

- **At P = 1,600 the rule does not fire within 60 iterates** in all three draws of σ = 2 and in draws 1 and 2 of σ = 4. The values are those of the 60th iterate, as returned: over the last 50 iterates χ stays between 0.11 and 0.24, above τχ = 0.1, while the residual falls slowly. The H2 errors there are 3e-11 to 1e-9.
- The ratios at P = 1,600, σ = 2 (33 to 60) rest on δ_P values that the two least-squares routines do not agree on (Section 6.4).

### 6.4 δ_P two ways

δ_L2 and δ_H2 by gelsy and gelsd at the same tolerance (rcond = ε), for the 54 spaces of 4a, 4b and p = 10 to 30 of Package 4. The difference is relative to the larger value. Spaces where either differs by more than 0.1%:

| space | P | δ_L2 gelsy | δ_L2 gelsd | difference | δ_H2 gelsy | δ_H2 gelsd | difference |
|---|---|---|---|---|---|---|---|
| rf_P400_sigma2_draw0 | 400 | 2.2e-8 | 2.97e-8 | 26% | 1.7e-6 | 1.74e-6 | 2.4% |
| rf_P400_sigma2_draw1 | 400 | 2.29e-8 | 6.91e-8 | 67% | 2.19e-6 | 2.19e-6 | <0.01% |
| rf_P400_sigma2_draw2 | 400 | 2.13e-8 | 2.75e-8 | 23% | 1.79e-6 | 1.8e-6 | 0.6% |
| rf_P400_sigma4_draw0 | 400 | 2.86e-6 | 2.88e-6 | 0.96% | 0.000248 | 0.000248 | <0.01% |
| rf_P400_sigma8_draw0 | 400 | 0.000172 | 0.000175 | 1.3% | 0.0147 | 0.0147 | <0.01% |
| rf_P800_sigma2_draw0 | 800 | 5.66e-11 | 1.94e-10 | 71% | 5.34e-10 | 7.77e-10 | 31% |
| rf_P800_sigma2_draw1 | 800 | 1.24e-10 | 1.76e-10 | 29% | 1.05e-9 | 1.38e-9 | 24% |
| rf_P800_sigma2_draw2 | 800 | 3.99e-11 | 2.02e-10 | 80% | 6.12e-10 | 7.57e-10 | 19% |
| rf_P800_sigma4_draw0 | 800 | 3.04e-9 | 3.33e-9 | 8.9% | 3.73e-7 | 1.8e-6 | 79% |
| rf_P800_sigma4_draw1 | 800 | 3.13e-9 | 3.98e-9 | 22% | 3.81e-7 | 5.44e-7 | 30% |
| rf_P800_sigma4_draw2 | 800 | 2.88e-9 | 1.09e-8 | 74% | 3.32e-7 | 2.41e-6 | 86% |
| rf_P800_sigma8_draw0 | 800 | 1.41e-6 | 2.63e-6 | 46% | 0.000209 | 0.000209 | 0.083% |
| rf_P800_sigma8_draw1 | 800 | 1.79e-6 | 1.81e-6 | 1.4% | 0.000222 | 0.000223 | 0.43% |
| rf_P800_sigma8_draw2 | 800 | 1.1e-6 | 8.41e-6 | 87% | 0.000156 | 0.000156 | <0.01% |
| cheb_p34 | 1156 | 2.4e-11 | 2.39e-11 | 0.2% | 4.58e-9 | 4.58e-9 | 0.048% |
| cheb_p38 | 1444 | 3.29e-13 | 3.27e-13 | 0.81% | 6.85e-11 | 6.85e-11 | 0.011% |
| rf_P1600_sigma2_draw0 | 1600 | 9.55e-13 | 1.14e-11 | 92% | 8.24e-13 | 3.07e-12 | 73% |
| rf_P1600_sigma2_draw1 | 1600 | 9.31e-13 | 1.52e-11 | 94% | 9.62e-13 | 9.74e-12 | 90% |
| rf_P1600_sigma2_draw2 | 1600 | 5.8e-13 | 7.69e-12 | 92% | 7.21e-13 | 3.49e-12 | 79% |
| rf_P1600_sigma4_draw0 | 1600 | 1.26e-12 | 8.12e-12 | 84% | 3.59e-11 | 1.17e-10 | 69% |
| rf_P1600_sigma4_draw1 | 1600 | 2.27e-12 | 7.84e-12 | 71% | 8.67e-11 | 1.38e-10 | 37% |
| rf_P1600_sigma4_draw2 | 1600 | 9.26e-13 | 2.67e-12 | 65% | 8.16e-11 | 4.12e-10 | 80% |
| rf_P1600_sigma8_draw0 | 1600 | 3.61e-9 | 3.98e-9 | 9.4% | 5.08e-7 | 5.42e-7 | 6.4% |
| rf_P1600_sigma8_draw1 | 1600 | 8.42e-9 | 3.49e-8 | 76% | 9.58e-7 | 2.95e-6 | 68% |
| rf_P1600_sigma8_draw2 | 1600 | 5.66e-9 | 8.91e-9 | 36% | 8.92e-7 | 3.83e-6 | 77% |
| cheb_p42 | 1764 | 6.47e-15 | 6.96e-15 | 7.1% | 1.59e-12 | 1.6e-12 | 0.37% |

- **Chebyshev:** within 0.03% for p = 10 to 30; 0.2%, 0.81% and 7.1% in L2 at p = 34, 38, 42 (H2: at most 0.37%).
- **Random features:** within 0.01% up to P = 200; up to 67% at P = 400, 87% at P = 800 and 94% at P = 1,600, in L2 or H2.
- gelsy gives the smaller δ_P in every space where the two differ by more than 0.1%.
- Every value is in `P5_4_pb/delta_P_two_ways.csv`.

## 7. Rules applied

1. **Section 7, report both values:** BL at weight 10 (the split; your answer 3); 4b at P = 800, σ = 2 (your answer); 4a at p = 42 seed 1, timed passes one solve longer than the logged run.
2. **Section 7, report status as returned:** L-BFGS at the 2,000 cap (Burgers p = 26 and 30; every PB run); L-BFGS-B's "abnormal termination in line search" at Burgers p = 18; 2c's diverged run; 4b's runs where the rule did not fire.
3. **Your answer 1, the factor-of-three rule:** no run left out.
4. **Rules not triggered:** the p = 88 references stopped by the rule (no fallback); every item 1 and Beltrami run stopped by the rule within K_max.

## 8. SU

122.6 SU charged at Grace's rates, against the approved 135 (up to 170); 342 were requested (`su.csv`).

| Job | ID | Class | Elapsed | SU |
|---|---|---|---|---|
| preflight | 20533636 | cpu-mem (24/h) | 11.0 min | 4.4 |
| ref-viscous | 20533637 | cpu-mem | 37.1 min | 14.8 |
| ref-gravity | 20533639 | cpu-mem | 32.9 min | 13.2 |
| item1 | 20533640 | cpu-mem | 14.0 min | 5.6 |
| beltrami | 20533641 | cpu-mem | 71.6 min | 28.6 |
| timed | 20533642 | timed-cpu (48/h) | 35.0 min | 28.0 |
| item4 | 20533643 | cpu-mem | 53.7 min | 21.5 |
| report | 20533644 | cpu (24/h) | 6.5 min | 2.6 |
| retime | 20534654 | timed-cpu | 4.8 min | 3.9 |
| **total** | | | | **122.6** |

## Files

- **The items:** `P5_1_weights/` (item 1; `references.json`, `compare_expected.csv`), `P3_1_beltrami_certified/` (item 1's Beltrami runs; `beside_delivered.csv`), `P5_2_nlls/` (item 2; `compare_expected.csv`, `retime.csv`), `P5_4_pb/` (item 4; `terminal_4a.csv`, `compare_indicative_4a.csv`, `underdetermined/`, `terminal_4b.csv`, `compare_expected_4b.csv`, `delta_P_two_ways.csv`, `points/`). Every run folder holds the coefficients of every iterate.
- **The re-time:** `package5_retime/`.
- **The checks:** `checks.json` (C1 to C3).
- **The records:**
  - `su.csv` and `su_per_job.csv`;
  - `hardware.json`, `environment.txt`, and `provenance/timed/` (the timed node's full record);
  - `DECISIONS_package5.md`;
  - `provenance.json`: the commits, and the SHA-256 of your four expected files;
  - `table_rerun_differences.json`;
  - `COMMIT`, `sacct.txt`, `slurm_logs/`, `report_log.txt`, `assembly_log.txt`.
- **The publication notes:** `README.md` and `REDACTED_FILES.txt`.

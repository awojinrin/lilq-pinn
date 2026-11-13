# Computational Package 4: report

The runs of the instructions of 12 November 2026, on Grace, 13 November 2026.

- **Code:** branch `v3-dev`. Every run is at commit `8a1e849`, under the source lock (`COMMIT`).
- **Results:** this folder, branch `package4-results`. It was assembled at `e949cc9`; the assembly changed no code a run uses.
- **Expected values:** each table gives ours / expected. The expected values are those of `expected_burgers.json` and `expected_pb.json`.
- **Times:** the times are Grace's. The expected files' times come from another machine, so they are not compared.

## 1. What ran

**Six jobs**, all CPU, all on Grace, partition medium.

| Job | Class | Node | What |
|---|---|---|---|
| preflight | cpu, 24 cores | c165 | versions, inputs, the test suite (807 passed), the points |
| timed | timed-cpu, whole node, 48 threads | c098 | M0; every timing of 1a, 1b and 2a |
| item1 | cpu, 24 cores | c165 | 1a logged, 1b, 1c |
| item2 | cpu, 24 cores | c204 | 2a logged, 2b, 2c, 2d, 2e, M7, the solution field |
| item2-constants | cpu, 24 cores | c204 | 2f: three seeds side by side at 8 threads each |
| report | cpu, 24 cores | | summaries, checks, the tarball |

**The software:**
- numpy 1.26.4 and scipy 1.13.1;
- OpenBLAS 0.3.27, through FlexiBLAS 3.4.4;
- the timed node: Intel Xeon Gold 6248R (`environment.txt`, `hardware.json`).

**Item 1, Burgers:**
- 1a: p = 6, 10, ..., 30.
- 1b: variants E1 and E2 at tolerances 1e-3 and 1e-5.
- 1c: p = 6 to 30.

**Item 2, Poisson-Boltzmann:**

| Part | Runs |
|---|---|
| 2a | p = 10, 14, ..., 30 × seeds 0, 1, 2 |
| 2b | 30 |
| 2c | 15 |
| 2d | 16 |
| 2e | 15 |
| 2f | the 18 runs of 2a, plus 2d's two extreme sets at p = 22 |

**The Halton points:**
- The runs used points drawn on the laptop with scipy 1.17.0, shipped with the code (`data/package4/`).
- They reproduce the advisor's points.
- Grace's scipy 1.13.1 draws the same points (`P4_2_pb_inclusions/points/halton_compare.json`).

**Checks M1, M8, and M6's values at the five points** need the two expected files. Grace did not have them, so they were evaluated at the assembly, on Grace's files. Grace's own checks are kept as `*_grace.json`; every verdict Grace gave is unchanged.

## 2. Checks

| Check | Result | Numbers |
|---|---|---|
| M0 | pass | Burgers P = 625 and Bratu P = 225: 5 rows each, identical to `package1` |
| M1 | pass | p = 10 and 22: 5 and 6 solves, as expected; stall level, error and δ_P within 6e-12 relative |
| M2 | pass | largest difference 4.1e-13 |
| M3 | pass | 80 runs, every iterate: scaled identity ≤ 1.7e-16, at most 0.006 of P·ε (Section 4, item 6) |
| M4 | pass | 20 embeddings: largest difference 1.7e-16 |
| M5 | pass | 24 levels: stall levels within 6.6e-8 relative; errors within 1.6e-4 (Section 4, item 7) |
| M6 | pass | u* within 1.6e-16 and ρ within 5.6e-16 relative; area 3.009239 against 3.010274 (3.4e-4 relative) |
| M7 | pass | 140 × 840 rule: error L2 0.49%, error H2 0.06%, δ_L2 0.35%, δ_H2 0.21% |
| M8 | pass | p = 14 and 22: rule steps 3 and 4, as expected; stall level and errors within 1.7e-5 relative; points identical to the advisor's |
| M9 | **fails** for one configuration of 29 | every timing pass takes the logged solves; spreads ≤ 2.4%, except 1b E1 at 1e-5, 22.6% (Section 4, item 1) |

## 3. Results beside the expected values

### Item 1a: single solves from zero

Ours / expected. Stall level, error and δ_P agree within 6e-10 relative.

| p | P | N | solves | stall level | error | δ_P | error/δ_P | class | rank (gelsy) | κ (returned) | median time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 36 | 342 | 5 / 5 | 0.1306 / 0.1306 | 0.03957 / 0.03957 | 0.01893 / 0.01893 | 2.09 | A | 36 | 12.5 | 0.0061 |
| 10 | 100 | 930 | 5 / 5 | 0.03235 / 0.03235 | 0.003994 / 0.003994 | 0.00199 / 0.00199 | 2.007 | A | 100 | 28.8 | 0.0509 |
| 14 | 196 | 1806 | 5 / 5 | 0.006371 / 0.006371 | 2.848e-04 / 2.848e-04 | 2.184e-04 / 2.184e-04 | 1.304 | A | 196 | 48.9 | 0.192 |
| 18 | 324 | 2970 | 6 / 6 | 0.001109 / 0.001109 | 2.546e-05 / 2.546e-05 | 2.442e-05 / 2.442e-05 | 1.042 | A | 324 | 75.8 | 0.451 |
| 22 | 484 | 4422 | 6 / 6 | 1.798e-04 / 1.798e-04 | 2.775e-06 / 2.775e-06 | 2.762e-06 / 2.762e-06 | 1.005 | A | 484 | 113 | 0.811 |
| 26 | 676 | 6162 | 6 / 6 | 2.778e-05 / 2.778e-05 | 3.149e-07 / 3.149e-07 | 3.146e-07 / 3.146e-07 | 1.001 | A | 676 | 157 | 1.45 |
| 30 | 900 | 8190 | 6 / 6 | 4.146e-06 / 4.146e-06 | 3.605e-08 / 3.605e-08 | 3.604e-08 / 3.604e-08 | 1.000 | A | 900 | 209 | 2.85 |

### Item 1b: enrichment at stall

Ours / expected. At every level the solves are as expected; stall levels agree within 5e-12 relative and errors within 3e-10.

| variant | tolerance | final P | levels | total solves | status | median time (s) | single solve of 1a at the final P (s) |
|---|---|---|---|---|---|---|---|
| E1 | 1e-3 | 484 / 484 | 5 / 5 | 18 / 18 | tolerance met | 0.834 | 0.811 |
| E1 | 1e-5 | 900 / 900 | 7 / 7 | 24 / 24 | tolerance met | 2.95 | 2.85 |
| E2 | 1e-3 | 484 / 484 | 5 / 5 | 14 / 14 | tolerance met | 0.592 | 0.811 |
| E2 | 1e-5 | 900 / 900 | 7 / 7 | 18 / 18 | tolerance met | 2.11 | 2.85 |

### Item 1c: constants of the point sets

There are no expected values for this item. "Columns dropped" counts columns dropped by the column-pivoted QR at 1e-10.

| p | P | iterates | smallest c1 | largest c2 | largest c2/c1 | largest ϱ_r | columns dropped (max) |
|---|---|---|---|---|---|---|---|
| 6 | 36 | 60 | 0.9992 | 1.005 | 1.006 | 1 | 0 |
| 10 | 100 | 60 | 1 | 1 | 1 | 1 | 1 |
| 14 | 196 | 60 | 1 | 1 | 1 | 1 | 1 |
| 18 | 324 | 60 | 1 | 1 | 1 | 1 | 1 |
| 22 | 484 | 60 | 1 | 1 | 1 | 1 | 1 |
| 26 | 676 | 60 | 1 | 1 | 1 | 1 | 1 |
| 30 | 900 | 60 | 1 | 1 | 1 | 1 | 1 |

### Item 2a: LiL-Q

Ours / expected; the expected file has no p = 30. Stall levels agree within 3e-5 relative, error L2 within 0.10% and error H2 within 0.20%. The ratios to δ are given to three figures (Section 4, item 4).

| p | P | seed | N | rule step | stall level | error L2 | error H2 | L2/δ | H2/δ | class | rank (gelsy) | κ (returned) | median time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 100 | 0 | 919 | 2 / 2 | 2.688 / 2.688 | 0.04854 / 0.04854 | 0.3705 / 0.3705 | 2.25 | 1.45 | A | 100 / 100 | 7.82e+05 | 0.0306 |
| 10 | 100 | 1 | 919 | 2 / 2 | 2.736 / 2.736 | 0.05174 / 0.05174 | 0.3773 / 0.3773 | 2.40 | 1.48 | A | 100 / 100 | 1.04e+06 | 0.0306 |
| 10 | 100 | 2 | 919 | 2 / 2 | 2.758 / 2.758 | 0.05057 / 0.05057 | 0.3794 / 0.3794 | 2.34 | 1.48 | A | 100 / 100 | 9.42e+05 | 0.0304 |
| 14 | 196 | 0 | 1736 | 3 / 3 | 0.3983 / 0.3983 | 0.003956 / 0.003956 | 0.07038 / 0.07038 | 2.14 | 1.73 | A | 196 / 196 | 1.29e+09 | 0.144 |
| 14 | 196 | 1 | 1736 | 3 / 3 | 0.4137 / 0.4137 | 0.004228 / 0.004228 | 0.07559 / 0.07559 | 2.29 | 1.86 | A | 196 / 196 | 1.69e+09 | 0.144 |
| 14 | 196 | 2 | 1736 | 3 / 3 | 0.4157 / 0.4157 | 0.004296 / 0.004296 | 0.07646 / 0.07646 | 2.32 | 1.88 | A | 196 / 196 | 1.71e+09 | 0.146 |
| 18 | 324 | 0 | 2807 | 3 / 3 | 0.04688 / 0.04688 | 3.454e-04 / 3.454e-04 | 0.00818 / 0.00818 | 2.66 | 1.81 | C | 324 / 324 | 1.14e+12 | 0.289 |
| 18 | 324 | 1 | 2807 | 3 / 3 | 0.04606 / 0.04606 | 3.372e-04 / 3.372e-04 | 0.009464 / 0.009464 | 2.60 | 2.09 | C | 324 / 324 | 1.30e+12 | 0.289 |
| 18 | 324 | 2 | 2807 | 3 / 3 | 0.04643 / 0.04643 | 3.441e-04 / 3.441e-04 | 0.007077 / 0.007077 | 2.65 | 1.56 | C | 324 / 324 | 1.50e+12 | 0.290 |
| 22 | 484 | 0 | 4136 | 4 / 4 | 0.003183 / 0.003183 | 1.886e-05 / 1.886e-05 | 7.538e-04 / 7.538e-04 | 3.14 | 2.32 | C | 484 / 484 | 9.45e+11 | 0.738 |
| 22 | 484 | 1 | 4136 | 4 / 4 | 0.003273 / 0.003273 | 1.822e-05 / 1.824e-05 | 8.393e-04 / 8.393e-04 | 3.04 | 2.58 | C | 484 / 484 | 1.05e+12 | 0.766 |
| 22 | 484 | 2 | 4136 | 4 / 4 | 0.003232 / 0.003232 | 1.875e-05 / 1.876e-05 | 5.094e-04 / 5.094e-04 | 3.13 | 1.57 | C | 484 / 484 | 8.32e+11 | 0.737 |
| 26 | 676 | 0 | 5719 | 4 / 4 | 1.261e-04 / 1.261e-04 | 8.250e-07 / 8.255e-07 | 3.725e-05 / 3.733e-05 | 5.28 | 2.75 | C | 670 / 670 | 4.81e+11 | 1.39 |
| 26 | 676 | 1 | 5719 | 4 / 4 | 1.319e-04 / 1.319e-04 | 7.814e-07 / 7.811e-07 | 3.104e-05 / 3.099e-05 | 5.00 | 2.29 | C | 669 / 669 | 5.67e+11 | 1.38 |
| 26 | 676 | 2 | 5719 | 4 / 4 | 1.269e-04 / 1.269e-04 | 7.737e-07 / 7.739e-07 | 3.100e-05 / 3.097e-05 | 4.95 | 2.29 | C | 669 / 669 | 5.05e+11 | 1.37 |
| 30 | 900 | 0 | 7559 | 4 | 3.064e-06 | 1.901e-08 | 1.123e-06 | 7.88 | 3.42 | C | 877 | 5.20e+11 | 2.70 |
| 30 | 900 | 1 | 7559 | 4 | 3.165e-06 | 1.776e-08 | 6.833e-07 | 7.36 | 2.08 | C | 875 | 5.95e+11 | 2.67 |
| 30 | 900 | 2 | 7559 | 4 | 3.047e-06 | 1.729e-08 | 1.144e-06 | 7.16 | 3.48 | C | 875 | 5.72e+11 | 2.67 |

### Item 2b: square collocation

The step at which a run stops; ours / expected, seeds 0, 1, 2. The totals are:

| | Ours | Expected |
|---|---|---|
| diverged | 25 | 24 |
| budget | 3 | 4 |
| converged | 2 | 2 |

| P | points | status | steps |
|---|---|---|---|
| 100 | Fekete | converged, converged, budget / the same | 6, 20, 30 / 8, 21, 30 |
| 100 | scattered | budget, **diverged**, budget / budget, **budget**, budget | 30, 27, 30 / 30, 30, 30 |
| 196 | Fekete | diverged ×3 / the same | 24, 4, 6 / 24, 5, 9 |
| 196 | scattered | diverged ×3 / the same | 5, 5, 5 / 10, 13, 8 |
| 324 | Fekete | diverged ×3 / the same | 0, 0, 5 / 0, 0, 3 |
| 324 | scattered | diverged ×3 / the same | 0, 0, 0 / 0, 0, 0 |
| 484 | Fekete | diverged ×3 / the same | 0, 0, 1 / 1, 0, 0 |
| 484 | scattered | diverged ×3 / the same | 0, 0, 0 / 0, 0, 0 |
| 676 | both | diverged ×3 / the same | 0, 0, 0 / 0, 0, 0 |

The two converged runs (P = 100, Fekete, seeds 0 and 1):

| Seed | Error L2 (ours / expected) | Error H2 (ours / expected) |
|---|---|---|
| 0 | 0.9526 / 0.9526 | 6.365 / 6.365 |
| 1 | 1.121 / 1.121 | 18.30 / 18.30 |

### Item 2c: underdetermined collocation

Every status and step count matches the expected file; both errors agree within 1e-9 relative.

| P | seed | N | status | steps | error L2 | error H2 |
|---|---|---|---|---|---|---|
| 100 | 0 | 50 | budget / budget | 30 / 30 | 1.525 / 1.525 | 6.42 / 6.42 |
| 100 | 1 | 50 | converged / converged | 6 / 6 | 0.8095 / 0.8095 | 7.103 / 7.103 |
| 100 | 2 | 50 | converged / converged | 8 / 8 | 0.7554 / 0.7554 | 3.44 / 3.44 |
| 196 | 0 | 98 | converged / converged | 6 / 6 | 0.1515 / 0.1515 | 2.462 / 2.462 |
| 196 | 1 | 98 | converged / converged | 7 / 7 | 0.2061 / 0.2061 | 1.557 / 1.557 |
| 196 | 2 | 98 | converged / converged | 6 / 6 | 0.1433 / 0.1433 | 1.587 / 1.587 |
| 324 | 0 | 162 | converged / converged | 6 / 6 | 0.04327 / 0.04327 | 0.6696 / 0.6696 |
| 324 | 1 | 162 | converged / converged | 6 / 6 | 0.04183 / 0.04183 | 0.7873 / 0.7873 |
| 324 | 2 | 162 | converged / converged | 6 / 6 | 0.03813 / 0.03813 | 0.4489 / 0.4489 |
| 484 | 0 | 242 | converged / converged | 8 / 8 | 0.0168 / 0.0168 | 0.3284 / 0.3284 |
| 484 | 1 | 242 | converged / converged | 8 / 8 | 0.01275 / 0.01275 | 0.3303 / 0.3303 |
| 484 | 2 | 242 | converged / converged | 7 / 7 | 0.0135 / 0.0135 | 0.2783 / 0.2783 |
| 676 | 0 | 338 | converged / converged | 9 / 9 | 0.006603 / 0.006603 | 0.2663 / 0.2663 |
| 676 | 1 | 338 | converged / converged | 12 / 12 | 0.007142 / 0.007142 | 0.2335 / 0.2335 |
| 676 | 2 | 338 | converged / converged | 7 / 7 | 0.00677 / 0.00677 | 0.2122 / 0.2122 |

### Item 2d: rows per unknown

Ours / expected; seed 0. Every N and rule step matches.
- **P = 196:** stall levels and error H2 agree within 1e-8 relative.
- **P = 484:** see Section 4, item 3.

| P | interior points per unknown | N | rule step | stall level | error H2 | H2/δ | error L2 | L2/δ | class |
|---|---|---|---|---|---|---|---|---|---|
| 196 | 1.5 | 462 | 3 / 3 | 0.2514 / 0.2514 | 0.567 / 0.567 | 13.9 / 13.9 | 0.01111 | 6.01 | A |
| 196 | 2 | 560 | 3 / 3 | 0.2922 / 0.2922 | 0.1694 / 0.1694 | 4.16 / 4.16 | 0.004176 | 2.26 | A |
| 196 | 3 | 756 | 3 / 3 | 0.3486 / 0.3486 | 0.1315 / 0.1315 | 3.23 / 3.23 | 0.003756 | 2.03 | A |
| 196 | 4 | 952 | 3 / 3 | 0.3611 / 0.3611 | 0.09975 / 0.09975 | 2.45 / 2.45 | 0.003598 | 1.95 | A |
| 196 | 6 | 1344 | 3 / 3 | 0.3992 / 0.3992 | 0.07069 / 0.07069 | 1.74 / 1.74 | 0.003997 | 2.16 | A |
| 196 | 8 | 1736 | 3 / 3 | 0.3983 / 0.3983 | 0.07038 / 0.07038 | 1.73 / 1.73 | 0.003956 | 2.14 | A |
| 196 | 12 | 2520 | 3 / 3 | 0.4158 / 0.4158 | 0.07114 / 0.07114 | 1.75 / 1.75 | 0.004187 | 2.27 | A |
| 196 | 16 | 3304 | 3 / 3 | 0.4159 / 0.4159 | 0.07228 / 0.07228 | 1.78 / 1.78 | 0.004203 | 2.27 | A |
| 484 | 1.5 | 990 | 4 / 4 | 0.001369 / 0.001369 | 0.04392 / 0.0443 | 135 / 136 | 3.553e-04 | 59.2 | C |
| 484 | 2 | 1232 | 4 / 4 | 0.001896 / 0.001896 | 0.01764 / 0.01617 | 54.3 / 49.7 | 9.695e-05 | 16.2 | C |
| 484 | 3 | 1716 | 3 / 3 | 0.00258 / 0.00258 | 0.002863 / 0.002864 | 8.81 / 8.81 | 2.405e-05 | 4.01 | C |
| 484 | 4 | 2200 | 4 / 4 | 0.002832 / 0.002832 | 0.001142 / 0.00114 | 3.51 / 3.51 | 1.831e-05 | 3.05 | C |
| 484 | 6 | 3168 | 4 / 4 | 0.003169 / 0.003169 | 7.588e-04 / 7.584e-04 | 2.33 / 2.33 | 1.864e-05 | 3.11 | C |
| 484 | 8 | 4136 | 4 / 4 | 0.003183 / 0.003183 | 7.538e-04 / 7.538e-04 | 2.32 / 2.32 | 1.886e-05 | 3.14 | C |
| 484 | 12 | 6072 | 4 / 4 | 0.003315 / 0.003315 | 5.912e-04 / 5.912e-04 | 1.82 / 1.82 | 1.946e-05 | 3.24 | C |
| 484 | 16 | 8008 | 4 / 4 | 0.003402 / 0.003402 | 5.969e-04 / 5.969e-04 | 1.84 / 1.84 | 1.975e-05 | 3.29 | C |

### Item 2e: boundary weight

Ours / expected; seed 0. Every rule step matches; both errors and H2/δ agree within 0.02% relative.

| P | λ | rule step | stall level | error L2 | error H2 | H2/δ | L2/δ | class |
|---|---|---|---|---|---|---|---|---|
| 196 | 1 | 3 / 3 | 0.2748 | 0.06685 / 0.06685 | 1.625 / 1.625 | 39.9 / 39.9 | 36.2 | A |
| 196 | 10 | 3 / 3 | 0.3128 | 0.02311 / 0.02311 | 0.5839 / 0.5839 | 14.4 / 14.4 | 12.5 | A |
| 196 | 100 | 3 / 3 | 0.3512 | 0.009583 / 0.009583 | 0.1899 / 0.1899 | 4.67 / 4.67 | 5.18 | A |
| 196 | 1000 | 3 / 3 | 0.3983 | 0.003956 / 0.003956 | 0.07038 / 0.07038 | 1.73 / 1.73 | 2.14 | A |
| 196 | 1e4 | 3 / 3 | 0.435 | 0.005135 / 0.005135 | 0.04958 / 0.04958 | 1.22 / 1.22 | 2.78 | A |
| 324 | 1 | 4 / 4 | 0.03256 | 0.007641 / 0.007641 | 0.3588 / 0.3588 | 79.3 / 79.3 | 58.9 | C |
| 324 | 10 | 4 / 4 | 0.03814 | 0.003049 / 0.003049 | 0.1224 / 0.1224 | 27.0 / 27.0 | 23.5 | A |
| 324 | 100 | 3 / 3 | 0.04295 | 0.001271 / 0.001271 | 0.02762 / 0.02762 | 6.10 / 6.10 | 9.80 | A |
| 324 | 1000 | 3 / 3 | 0.04688 | 3.454e-04 / 3.454e-04 | 0.00818 / 0.00818 | 1.81 / 1.81 | 2.66 | C |
| 324 | 1e4 | 3 / 3 | 0.04927 | 4.911e-04 / 4.911e-04 | 0.005118 / 0.005118 | 1.13 / 1.13 | 3.79 | C |
| 484 | 1 | 4 / 4 | 0.002679 | 4.688e-04 / 4.687e-04 | 0.03499 / 0.03499 | 108 / 108 | 78.2 | C |
| 484 | 10 | 4 / 4 | 0.002911 | 1.254e-04 / 1.254e-04 | 0.007681 / 0.007682 | 23.6 / 23.6 | 20.9 | C |
| 484 | 100 | 3 / 3 | 0.003037 | 6.009e-05 / 6.008e-05 | 0.001686 / 0.001685 | 5.18 / 5.18 | 10.0 | C |
| 484 | 1000 | 4 / 4 | 0.003183 | 1.886e-05 / 1.886e-05 | 7.538e-04 / 7.538e-04 | 2.32 / 2.32 | 3.14 | C |
| 484 | 1e4 | 3 / 3 | 0.003368 | 2.570e-05 / 2.570e-05 | 5.028e-04 / 5.028e-04 | 1.55 / 1.55 | 4.29 | C |

### Item 2f: constants of the point sets

There are no expected values for this item. "Columns dropped" counts columns dropped by the column-pivoted QR at 1e-10.

| run | iterates | smallest c1 | largest c2 | largest c2/c1 | largest ϱ_r | columns dropped (max) |
|---|---|---|---|---|---|---|
| p10_seed0 | 60 | 0.4043 | 1.627 | 4.025 | 1.077 | 0 |
| p10_seed1 | 60 | 0.3846 | 1.345 | 3.498 | 1.013 | 0 |
| p10_seed2 | 60 | 0.2381 | 1.369 | 5.752 | 1.014 | 0 |
| p14_seed0 | 60 | 0.1873 | 1.654 | 8.828 | 1.130 | 0 |
| p14_seed1 | 60 | 0.1611 | 1.841 | 11.43 | 1.042 | 0 |
| p14_seed2 | 60 | 0.2369 | 1.592 | 6.719 | 1.034 | 0 |
| p18_seed0 | 60 | 0.1327 | 1.829 | 13.79 | 1.018 | 1 |
| p18_seed1 | 60 | 0.09972 | 1.926 | 19.32 | 1.016 | 1 |
| p18_seed2 | 60 | 0.08391 | 1.809 | 21.56 | 1.028 | 1 |
| p22_seed0 | 60 | 0.1143 | 2.104 | 18.37 | 1.084 | 13 |
| p22_seed1 | 60 | 0.07854 | 1.833 | 23.32 | 1.057 | 13 |
| p22_seed2 | 60 | 0.08249 | 1.871 | 22.64 | 1.055 | 13 |
| p22_ratio1.5 | 30 | 1.726e-04 | 4.087 | 2.344e+04 | 359.5 | 13 |
| p22_ratio16 | 30 | 0.2711 | 1.689 | 6.224 | 1.007 | 13 |
| p26_seed0 | 60 | 0.06201 | 1.980 | 31.84 | 1.123 | 40 |
| p26_seed1 | 60 | 0.07798 | 1.898 | 24.34 | 1.104 | 40 |
| p26_seed2 | 60 | 0.03545 | 1.945 | 54.75 | 1.130 | 40 |
| p30_seed0 | 60 | 0.08537 | 1.906 | 22.32 | 1.096 | 90 |
| p30_seed1 | 60 | 0.0544 | 1.906 | 34.75 | 1.149 | 90 |
| p30_seed2 | 60 | 0.01685 | 1.951 | 114.6 | 1.811 | 90 |

## 4. Deviations and rules applied

1. **M9, 1b E1 at tolerance 1e-5** (Section 6.2, "timings differ by more than 15%"). The three timings are 2.943, 3.607 and 2.946 s; the median is 2.946 s. Every timing pass took the logged 24 solves.

2. **2b differs from the expected run by run** (Section 6.2, "report both values side by side"):
   - one scattered run at P = 100 (seed 1) diverges at step 27, where the expected run reaches the budget;
   - the two converged runs take 6 and 20 steps, against 8 and 21, with the same errors to four figures;
   - the step at which a run diverges differs in 8 of the other runs.

   Square collocation is a Newton iteration that diverges in 25 of 30 runs; where it stops depends on round-off. Running the advisor's script on the laptop (numpy 2.4.2) also differs from the expected file run by run.

3. **2d at P = 484 with 1.5 and 2 interior points per unknown** (Section 6.2, "report both values side by side"):
   - the rule steps, N and stall levels agree (within 2e-4);
   - error H2 differs by 0.9% and 9.1%: H2/δ is 135 against 136, and 54.3 against 49.7;
   - these are the worst-conditioned systems of the sweep (class C).

4. **δ_P of item 2 is reproducible to about three figures at P ≥ 484** (no rule of Section 6.2 covers it; it changes no verdict).
   - **Why:** δ_P is a least-squares projection onto the basis restricted to the domain with holes, which is ill-conditioned.
   - **The spread** of Grace's value, the laptop's and the expected file's "best": 9e-5 relative at P = 484 and 9e-4 at P = 676. Grace and the laptop differ by 1.4e-3 at P = 900. At P ≤ 324 the spread is below 4e-8.
   - **What is reported:** the tables are Grace's, and the ratios to δ are given to three figures. `delta_P_across_machines.json` has the values.

5. **2a at p = 30 is rank deficient** (Section 6.2, "item 2a at p = 30 ... rank deficient: report what was logged").
   - gelsy's rank is 875 to 877 of 900; at p = 26 it is 669 to 670 of 676, as in the expected file.
   - The runs completed; the rule fired at step 4 in every seed; the values logged are reported.

6. **M3 is judged by the scaled identity**, ‖(Aβ − f) − N(β)‖ / ‖ |A||β| + |f| ‖ at every iterate, against P·ε (`DECISIONS_package4.md`).

   Package 3's K4 reading, the plain relative difference at k = 1, is recorded:
   - it is at most 1e-15 in item 1a;
   - it reaches 1.8e-10 at the levels of 1b that start from an embedded iterate, already on the plateau, where ‖N‖ is about the stall level;
   - it reaches 1e-2 in item 2a at p = 30. There the first iterate's coefficients reach 6e9 (κ about 1e11 to 1e12), so forming Aβ in floating point loses about ε·‖ |A||β| ‖.

   The scaled measure is at most 1.7e-16 everywhere.

7. **M5 is judged on the stall levels.**
   - The stall levels agree within 6.6e-8 relative.
   - The errors of a level's returned iterate differ from 1a's by up to 1.6e-4 (E2, p = 10), since a level stops at a different step of the plateau than the single solve.
   - The reference script gives the same differences between its own enrichment and its own single solves, so Section 5's 1e-6 for the errors holds to about three figures there.

8. **The constants of 2f** (Section 6.2, "a constant is poor or the QR drops many columns"):
   - p22_ratio1.5 has smallest c1 1.7e-4 and largest ϱ_r 360;
   - the QR drops 13 columns at p = 22, 40 at p = 26 and 90 at p = 30;
   - the points and weights are unchanged.

   Item 1c drops one column from p = 10, with c1 and c2 at 1.

   The three seeds of 2f ran side by side at 8 BLAS threads each. Which near-dependent columns the QR drops at a plateau iterate depends on round-off, so a single iterate's c1 can move by up to 10% with the thread count. The per-run values of the table agree within 1e-6 either way (measured on the laptop at p = 22).

9. **Rules not triggered:**
   - the rule fired within K_max in every run of 1a and 2a;
   - no enrichment run reached p = 34;
   - M7 changed no value by more than 0.49%; the deliverables use the 70 × 420 rule;
   - Grace's scipy draws the same Halton points.

## 5. SU

36.0 SU charged at Grace's rates, against the cap of 100; 88 were requested (`su.csv`).

| Job | ID | Class | Elapsed | SU |
|---|---|---|---|---|
| preflight | 20515658 | cpu (24/h) | 10 min 33 s | 4.2 |
| timed | 20515659 | timed-cpu (48/h) | 5 min 50 s | 4.7 |
| item1 | 20515660 | cpu | 17 min 22 s | 6.9 |
| item2 | 20515662 | cpu | 15 min 36 s | 6.2 |
| item2-constants | 20515664 | cpu | 31 min 22 s | 12.5 |
| report | 20515667 | cpu | 3 min 41 s | 1.5 |
| **total** | | | | **36.0** |

## Files

- **The items:** `P4_1a_burgers_gauss/`, `P4_1b_burgers_enrichment/`, `P4_1c_burgers_constants/`, `P4_2_pb_inclusions/`. Each is in the Section 7 layout; item 2's points are in `P4_2_pb_inclusions/points/`.
- **The checks:**
  - `checks.json`: M0 to M9;
  - `checks_item1.json` and `P4_2_pb_inclusions/checks_item2.json`: the details;
  - `*_grace.json`: Grace's own checks;
  - `P4_checks/`: M0's runs.
- **The records:**
  - `su.csv` and `su_per_job.csv`;
  - `hardware.json`, `environment.txt`, and `provenance/timed/` (the timed node's full record);
  - `DECISIONS_package4.md`;
  - `provenance.json`: the commits, and the SHA-256 of the expected files;
  - `delta_P_across_machines.json`;
  - `COMMIT`, `sacct.txt`, `slurm_logs/`, `report_log.txt`, `assembly_log.txt`.
- **The publication notes:** `README.md` and `REDACTED_FILES.txt`.

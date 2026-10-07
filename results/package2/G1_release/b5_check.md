# Appendix B.5 check: Table `tab:run_settings` against the runs and the code

**Package 2, Section 2, item 3.** Every entry of the table (`run_settings_B5.csv`) was compared with what the runs of `package1` actually used. The evidence was read from the runs' own records:
- each run's `run.json` (basis, P, N and its composition, collocation construction and seed, row weights, stopping rule, K_max);
- the four-method, B9 and B10 tables;
- the saved networks;
- the Component A configurations.

Where a run records nothing, the evidence comes from the code (the basis study's grid). The comparison is `experiments/b5_check.py`; its output, one row per table row in the table's columns, is `b5_actual.csv`. The code was not changed.

**Result: one discrepancy** (the four-method row's P range) and **six clarifications**. Every other entry matches.

## Discrepancy

**1. Four-method comparison (6.1), column P: "25–1051".**
- **What ran:**
  - LiL-Q and LiL-N use the benchmark's P, 25–1,024.
  - The NiL networks (tanh, two hidden layers of width h, h the smallest with h² + 5h ≥ P) have 37–1,051 parameters, h = 4 to 30.
- **The problem:** the table's range combines LiL's smallest P (25) with NiL's largest parameter count (1,051). No NiL network has 25 parameters: at P = 25, h = 4 and the count is h² + 5h + 1 = 37.
- **Suggested entry:** "LiL: 25–1,024; NiL: 37–1,051 parameters".

## Clarifications

None of these is an error in the table.

**2. Buckley–Leverett (both cases), N/P "9.8".** The runs have 9.81 at P = 64 and 9.75 at P = 256, 576 and 1,024, so a range "9.75–9.81" would be exact. "9.8" is correct as a rounded value.

**3. Seeds of the random-tensor grids.** Bratu, Burgers, both Buckley–Leverett cases and the four-method comparison all use collocation seed 42. The table's seeds column ("–") refers to training seeds, and has none for LiL-Q. The grid seed could be stated in the caption.

**4. K_max (not a column of the table).**
- **What the package's paper passes record:** K_max = 25 (Bratu), 20 (Burgers, gravity Buckley–Leverett, Kovasznay, Beltrami) and 50 (viscous Buckley–Leverett). These are the values of wave 2.
- **The code at this commit** uses K_max = 60 for every LiL-Q paper pass (`LILQ_PAPER_KMAX`, the advisor's follow-up of 1 October).
- **No result depends on the difference.** Every paper pass ends on its stopping criterion before its cap, except gravity Buckley–Leverett at P = 64. That run was rerun at K_max = 60 in wave 4, and it is the run in `package1`.

**5. Beltrami (6.7).** The table's entry (eight pressure pins) is the pinned run, `beltrami_pinned`: N = 21,816, N/P = 2.73. The unpinned Section 3.3 paper pass (`beltrami_P7984_cpu_paper`, one pin, N = 21,809) is also in `package1`. The manuscript's Beltrami numbers should come from the pinned run, as the advisor's follow-up of 1 October states.

**6. Darcy NiL (6.8).** The table's entry matches the float64 runs:
- the manuscript's network: three networks of 2 × 32, 3,555 parameters;
- 150,000 Adam epochs;
- seeds 0, 1 and 2;
- N/P = 39,600 / 3,555 = 11.14.

B9 also ran each seed in float32 (Addendum v2.2's 20% rule). The table does not state the precision.

**7. CGL with CC weights (6.10).** The table's entry matches the 12 Clenshaw–Curtis runs. B10 also contains the 4 equal-weight runs at P = 1,875 (N/P = 5 and 10, paper and kmax passes) used for the comparison of weights.

## Entries that match

| Table row | Checked against the runs |
|---|---|
| Bratu (6.2) | Fourier × Fourier; P 25–225; N/P 10.12–10.60; random tensor; boundary rows; loss target |
| Burgers (6.3) | sine × Fourier; P 25–625; N/P 9.53–10.32; random tensor; boundary and initial rows (the sine basis also satisfies u(±1) = 0); loss target |
| Buckley–Leverett N_g = 0 | Fourier × Fourier; P 64–1,024; random tensor; boundary and initial rows; loss target (initial guess: the initial profile) |
| Buckley–Leverett N_g = −5 | cosine × Fourier; P 64–1,024; random tensor; rows; loss target (initial guess: zero) |
| Four-method (6.1) | NiL tanh, two hidden layers; LiL-N on the LiL-Q basis; the LiL-Q collocation sets; stops by target, budget (`iteration_cap`) or stall (`optimizer_stall`); NiL seeds 0, 1, 2 |
| Basis study (6.3) | the ten bases of Table 3; P = 625; equispaced 73 × 73 interior grid; N/P = 10.03; initial and boundary rows; 50 iterations, stopping rule off |
| Elasticity (6.5) | cosine × sine (u_x), sine × Chebyshev (u_y); P 50–1,250; N/P 10.03–10.60; equispaced; boundary rows; one direct solve |
| Kovasznay (6.6) | Chebyshev × Chebyshev per field; P 75–1,875; N/P 2.94–5.08; equispaced; boundary rows and one pressure pin; relative coefficient change < 1e-9 |
| Beltrami (6.7) | 4D Chebyshev, 6⁴ per velocity and 8⁴ pressure, P = 7,984; N/P 2.73; equispaced 8⁴ interior; rows and eight pressure pins; relative coefficient change < 1e-9 |
| Darcy (6.8) | Fourier families with the lifting (h̃: cos × sin; u, v: augmented families); P = 3,169; N/P 12.50; cell centres (60 × 220); boundary conditions in the basis; one direct solve |
| Oversampling (6.10) | Bratu at P = 100, 225 and Kovasznay at P = 300, 1,200; N/P 1, 1.5, 2, 3, 5, 10, 20; CGL, paper and random distributions, the random ones in five draws (seeds 0–4); 196 runs |
| F1 (6.6) | F1_18: Fourier-feature MLP, 198,531 parameters; 2,000 random points, redrawn; soft boundary penalty; 60 minutes; seeds 0–4 |
| F2 (6.6) | F2_05: Fourier-feature MLP, 4,291 parameters; 8,000 random points, 3 equations, N/P 5.59; exact boundary conditions (Coons lifting); 60 minutes; seeds 0–4 |

To regenerate: `python experiments/b5_check.py --package <package1> --out b5_actual.csv`.

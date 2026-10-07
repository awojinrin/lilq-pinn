# Package 2, Stage 2: the laptop previews

These are the rehearsal runs on the laptop, done before submission. They are
kept so that the final report can set each laptop preview beside the Grace
result (the advisor's reply of 5 October, Section 5).

- **Laptop:** Windows 11, 31 GB RAM, RTX 5080 (17 GB). It has a different
  BLAS from Grace, so its numbers agree with Grace's only up to the
  machine-to-machine level. For the rank-deficient Buckley–Leverett runs that
  level is 2e-4 to 6e-3 in the residual (DECISIONS.md, check C8).
- **Grace results are the reported ones.**
- **Copied from** the session scratchpad on 5 October 2026.

| Item | Folder | What | Code when run |
|---|---|---|---|
| 1 (P2-3) | `P2_3_classical/bratu_kovasznay`, `P2_3_classical/burgers` | the classical baselines; Burgers in its own folder | Stage 2 batches 1 and 3 |
| 2 (P2-8) | `P2_8_lm_networks/all` | LM against L-BFGS, every size: Bratu with 3 seeds, Burgers and BL with 1. Summarized with the current code, including `lm_vs_lbfgs.md` | batch 2d, summary at batch 3 after the reply |
| 3 (P2-16) | `P2_16_certified/all` | the certified-grid runs | batch 2c |
| 4 (P2-14) | `P2_14_elm_sweep/partial` | the Burgers ELM sweep at one sigma, and the two Table 3 rows | batch 2b |
| 4 (P2-4) | `P2_4_elm_kovasznay/all` | ELM on Kovasznay: QR, `normal_shifted` and `normal_eigh` on 5 seeds and on the Chebyshev surrogate, plus C5 | after the reply (batch 3) |
| 6a (P2-2) | `P2_2_nu_refinement/all` | the nu-refinement, both cases. Its `check_c8.json` was recomputed with the current check (laptop against package1: it fails, as it should across machines) | batch 1 |
| 6b (P2-9) | `P2_9_bases/all` | the boundary-conforming bases. Old labels: `lifted_sine` is now `lifted_sine_x(1-x)`, and `lifted_plain_sine` is now `lifted_sine_plain` | batch 2 |
| 7 (P2-1) | `P2_1_scaling/kovasznay`, `P2_1_scaling/beltrami` | the scaling series in P and N, on the CPU and the laptop GPU | batches 3 and 4 |
| 8 (P2-10) | `P2_10_elasticity_manufactured` | both manufactured solutions: compatible and specified | after the reply (batch 2) |
| 5 (P2-15) | `P2_15_darcy_hardbc/S1_seed0_5min_smoke_paper_gain` (the code now), `..._literal_gain` (28eccb6) | smoke runs, not the item: S1 seed 0 with the cap cut to 5 min, on the laptop GPU (about 150 LM iterations). delta_FV 2.22e-4 (paper's gain) and 1.39e-4 (literal gain), against the paper's soft-BC Adam NiL 2.14e-2 and LiL 1.37e-4 | batches 4 and 5 |
| 5b (control) | `P2_15b_darcy_softbc_lm/S1_seed0_5min_smoke` | a smoke run, not the item: the paper's network and Dirichlet rows trained by LM, S1 seed 0, cap cut to 5 min, laptop GPU (146 iterations). delta_FV 2.88e-3 | batch 6 |

**Items 5 and 5b (Darcy)** have only those smoke runs on the laptop. Item 5's 12 runs are on
FASTER; 5b's 4 are on Grace.

# Computational Package 2: Stage 2 before submission

**To:** R. Younis. **From:** G. Awojinrin. **Date:** 5 October 2026.

**Code:** `awojinrin/lilq-pinn`, branch `v3-dev` at b2b99de: the 13 Stage 2 commits after ab0484d, the release check of the Stage-1 reply. `DECISIONS.md` has one entry per step, with the details behind everything below.

## Summary

All Stage 2 code is written except item 5 (Darcy), and has been rehearsed at full size on my laptop. Nothing has been submitted yet: Stage 2 runs from one commit (Section 12.1), so every job goes in together once item 5 is written.

Before submitting, I would like your view on two questions and on the readings in Section 3. Some of these change what runs. The laptop results in Section 4 are previews only: the reported numbers will be Grace's.

**For your reply:**
1. **Item 5's formulation** (Section 2.1): the paper's mixed three-network NiL formulation with the h* output lifted, or one network for h alone?
2. **Check C8** (Section 2.2): it cannot be met at 1e-10 for viscous Buckley–Leverett at P = 576 and 1,024, even on Grace. Will you accept agreement at Grace's own run-to-run level?
3. **Any objection to the readings in Section 3**, and any suggestion beyond the package (Section 5).

## 1. Where Stage 2 stands

| Item | What | Status | Job class |
|---|---|---|---|
| 1 | Classical baselines: Bratu, Kovasznay, and Burgers (4.3, run: item 1 stays below 250 SU) | written, rehearsed | timed-cpu |
| 2 | NiL-N by Levenberg–Marquardt: Bratu, Burgers, viscous BL | written, rehearsed | timed-cpu |
| 3 | Certified grids: Bratu, Burgers | written, rehearsed | cpu |
| 4 | ELM sweep, ELM on Kovasznay, normal-equation control | written, rehearsed | cpu |
| 5 | Darcy hard-BC network (FASTER) | **waits for your answer** | FASTER shared GPU |
| 6 | BL ν-refinement; boundary-conforming bases | written, rehearsed | timed-cpu, cpu |
| 7 | Scaling in P and N, Kovasznay and Beltrami, CPU and A100 | written, rehearsed | timed-cpu, timed |
| 8 | Manufactured elasticity | written, rehearsed | cpu |

The paper's configurations are unchanged by any of this. Wherever shared code was touched (Kovasznay, Beltrami, elasticity, BL, the ELM basis), the paper's runs give results identical bit for bit to the release.

**The submission chain** (one script on Grace):
1. a preflight (versions, GPU, test suite);
2. one job that makes every reference;
3. the 14 compute jobs;
4. a report job that writes `su_per_job.csv` and packs the stage.

## 2. Two questions

### 2.1 Item 5: one network or the mixed formulation?

Section 8 describes "2 hidden layers of 32 tanh neurons … h_θ = y* + ω(y) NN(x, y; θ)", with the lateral no-flow imposed through "−K ∂h/∂n formed with the K of the adjacent cell". The manuscript's Darcy formulation (Section 6.8, eq. `darcy_nondim`) is mixed for both methods. The NiL baseline uses three networks, for h*, u* and v* (3 × 1,185 = 3,555 parameters), with the Darcy-law and continuity residuals at the cell centres.

**My proposal:** keep the paper's mixed NiL formulation and its residuals, and change only:
- the h* output: y* + ω NN_h (hard Dirichlet, ω normalized to unit maximum);
- the activation: tanh, as Section 8 says (the paper's NiL uses SiLU);
- the lateral no-flow: weighted rows (λ = 10) on −K*ℛ ∂h*/∂x*, with the adjacent cell's K*;
- the optimizer: Levenberg–Marquardt with F2's damping.

**Why not one network for h alone.** It would have to collocate ∇·(K∇h) = 0 in strong form, with K constant on each cell. At a cell centre that reduces to K Δh, so the heterogeneity would enter only through the lateral rows. I do not think that is what you intend, but I would rather ask than guess.

### 2.2 Check C8 at 1e-10 for viscous BL

C8 asks that the ν = 0.1 reruns reproduce package1's `norm_R_h` at every k to 1e-10 (P = 576 and 1,024).

**On my laptop the reruns differ by 2e-4 to 6e-3.** They already differ by 1.6e-10 at k = 0, in the least-squares fit of the initial profile, and the difference grows from there. κ is 1e16 at P = 576 and 9e16 at P = 1,024; at P = 1,024 gelsy keeps 1,007 columns here against Grace's 1,004. The code is not the cause: the released code and today's give identical bits on the same machine.

**Grace does not reproduce these runs to 1e-10 either.** package1's paper and K_max passes of this configuration come from different Grace jobs (wave 1 at 8a3f5f7 and wave 2 at 905e58c). The LiL-Q and BL code is identical between those commits, and both ran on 48 threads. Over the iterations they share, they agree to 6.3e-5 (P = 576) and 1.1e-4 (P = 1,024).

**What I will do:**
- run C8 exactly as written, on package1's whole-node 48-thread setting;
- record beside it the stopping iteration (it matched on all four passes on the laptop), the final loss and error, and package1's own run-to-run agreement.

**The question:** would you accept agreement at Grace's own run-to-run level, with the same stopping iteration, as passing?

The analogous check C8′ (item 7) is attainable. package1's Kovasznay P = 1,875 and Beltrami runs agree bit for bit between waves, at κ of about 4e5 and 3e4.

## 3. Readings and departures, for your critique

Each is recorded in `DECISIONS.md`. Where the text left something open, I chose the reading closest to the paper's runs.

### 3.1 Item 1

**Item 1, Burgers (4.3).** The text says "the paper's LiL-Q Burgers runs are not rerun; their reference errors come from Section 6.2", and also "timing as in 4.1". 4.1 reruns the paper's LiL-Q in the same job for the time ratio, and takes its errors from elsewhere. I do the same for Burgers: LiL-Q is rerun for its time only (seconds), and its errors are Stage 1's.

### 3.2 Item 2

**Item 2 (Levenberg–Marquardt).**
- **The same runs.** Each benchmark's own Package 1 `run_nil_n` is called, with only the optimizer swapped. So the network, pretraining, collocation set (seed 42), weights and target are the Package 1 code itself, not a copy.
- **Check C3.** The LM loss is the same quantity as the L-BFGS objective at iteration 0, to at most 4e-16 on all three benchmarks.
- **The "f1" stall rule for LM.** A step is lost when the damping exceeds 1e16 without an accepted step. One restart resets the damping; a second loss in a row is the stall. F2's own 20-step stagnation tolerance is off ("tolerances 0").
- **Viscous BL runs.** It is cheap, so the skip rule does not apply.
- **The comparison with L-BFGS** is in iterations and loss. The L-BFGS times are the tables' GPU runs, and LM runs on the CPU, so time ratios across the two would mix devices.

### 3.3 Item 3

**Item 3 (certified grids).**
- **Check C4's wording is not achievable.** It asks that the CC-weighted norm of a polynomial of degree ≤ M − 1 per direction equal its L² norm. An M-point rule is exact for integrands of degree ≤ M − 1, but a squared norm of degree M − 1 has degree 2M − 2. On the M = 11 grid it is off by 2e-2. I check the exact form instead:
  - integrals of degree ≤ M − 1, relative to the polynomial's L² norm;
  - squared norms of degree ≤ ⌊(M − 1)/2⌋.

  Both hold to 1e-12 on every grid, at most 3e-13.
- **The m = 2 constant is not computable at N/P = 5** for Bratu (all three P) or Burgers P = 25. The text expects it to be. M = ⌈√(0.85·5·P)⌉ comes to exactly 2p_d + 1, so N_interior = dim Q_{2p_d}, and C6 marks the row not computable.
- **Burgers' lateral lines on (0, 1]:** the (n + 1)-point CGL rule with t = 0 dropped, together with its weight (about 0.1% of the line).
- **k_target** reads the run's own CC-weighted loss against the paper's target.

### 3.4 Item 4

**Item 4 (ELM on Kovasznay).** "Per field, a tanh random-feature basis of 600 neurons": I use one random basis of 600 neurons, shared by u, v and p, with each field its own coefficients (3 × 600 = 1,800).

### 3.5 Item 6

**Item 6a (ν-refinement).**
- **The references** are refined by doubling from Package 1's 4,000 intervals. Two successive refinements agree to 1e-6 at 8,000 to 16,000 intervals, at every ν.
- **The runs.** Each configuration has the paper pass and a K_max = 60 pass, as in Package 1. P = 1,600 has no paper target, so it has the K_max pass only.

**Item 6b (bases).** Three additions to Section 9.2, each labelled as such in `rows.csv`:
- **The paper's bases** rerun on the same references, so that each new basis has its comparison row.
- **δ_P on every row:** the reference's distance to the trial space.
- **A second lifted BL basis**, S = (1 − x) + Σ β_ij sin(iπx) φ_j(t), without the factor x(1 − x).

**Why the second lifted basis.** With the factor, the space is a sine series of w/(x(1 − x)), with w = S − (1 − x). That quotient does not vanish at the ends, so the series converges slowly. The δ_P values show it (Section 4). The basis as specified is run in full beside it.

### 3.6 Item 7

**Item 7 (Beltrami sizes).**
- **"The eight pressure pins … by the same rule as the paper's run".** The pins close the N_p-dimensional pressure null space, so I use N_p pins: 8, 9 and 10. Eight pins at N_p = 9 would leave the system rank-deficient.
- **The boundary rule.** I take it from the paper's run, N_ic = the grid and N_bc = N_t_bc = the grid − 2:
  - 10⁴ grid: N_bc = 8, N_ic = 10;
  - 11⁴ grid: N_bc = 9, N_ic = 11.
- **Beltrami's GPU path was broken in the code and is repaired.** It called `torch.linalg.lstsq(driver='gelsd')`, which PyTorch refuses on CUDA. Package 1 ran Beltrami only on the CPU, so it was never exercised. It now uses the GPU QR of Kovasznay's runs; the CPU path is unchanged.
- **Check C8′** has a GPU part only for Kovasznay, since the paper has no Beltrami GPU point.

### 3.7 Item 8

**Item 8 (manufactured elasticity).** Implemented as specified. The result is in Section 4, and Section 5 has two options.

## 4. Early findings (laptop rehearsals: previews, not the results)

### 4.1 Item 1, classical baselines

| Case | Square baseline | LiL-Q |
|---|---|---|
| Kovasznay, full rank, E_u at p_d = 10 / 15 / 20 / 25 | 2.3e-2 / 1.85e-5 / 4.8e-9 / 1.9e-13 | 2.9e-2 / 1.2e-5 / 7.4e-9 / 7.0e-13 (matching P) |
| Burgers, error near P = 400 / 625 | 9.2e-4 / 1.6e-4 (420 / 650 free coefficients) | 1.9e-4 / 4.6e-5 |

- **Burgers is the opposite of Bratu:** LiL-Q is the more accurate at matched size.
- **The Burgers square solver converges spectrally against Cole–Hopf,** reaching 1e-7 at p = 48.

### 4.2 Item 2, LM against L-BFGS

**LM reaches the targets where L-BFGS stalls:**

| Case | LM | L-BFGS |
|---|---|---|
| Burgers P = 625 | target 5e-9 in 443 iterations, error 9e-6 | stalls at 2e-5, error 6e-4 |
| BL P = 1,024 | target in 119 iterations | 7,436 iterations, target met by 2 of 3 seeds |
| Bratu P = 225 | 5e-6 at the 2,000-iteration cap | 7.6e-4 |

**But the loss and the error do not move together:**
- At Burgers P = 225 the errors are the same: 1.0e-3 against 9.8e-4.
- At the smallest sizes LM meets the loose targets with functions far from the solution: Bratu P = 25, error 1.04 against L-BFGS's 0.085; Burgers P = 25, 0.45 against 0.13.

### 4.3 Item 3, certified grids

- **Every run** converges in 4–5 iterations, with ϱ_r at 1.000 at the end.
- **ε/δ_P at k = 60:** 70, 16 and 15 for Bratu P = 25, 100, 225; 1.6 to 5.0 for Burgers.
- **At N/P = 20, c₁ = c₂ = 1:** the CC rule integrates the m = 2 Gram matrix exactly.
- **Burgers never meets the paper's loss targets on this Chebyshev space.** The targets were set for the sine-Fourier basis.

### 4.4 Item 4, ELM

- **Check C5 passes** (2.9e-14).
- **QR on the 1,800-coefficient ELM basis** reaches E_u of about 2–3e-10, with rank about 725 and κ about 5e18.
- **The normal-equation control fails at the first solve on every ELM basis.** Cholesky fails even with the prescribed shift (κ(AᵀA) is 3e20–1e21). As specified, it produces no iterates, so it cannot show the stagnation it was meant to test.

### 4.5 Item 6, ν-refinement and bases

**ν-refinement:**
- Below ν = 0.1, no run meets the paper's target in 60 iterations.
- The error at the stop rises to 0.13–0.22 at ν = 0.02–0.01.
- The overshoot grows as ν falls (the Gibbs onset).
- At P = 1,600 the undamped iteration does not converge: the loss oscillates, the error is 0.25 (ν = 0.02) and 0.79 (ν = 0.01), and the overshoot reaches 3–5.

**Bases:**

| Case | Result |
|---|---|
| Bratu sin⊗sin | poor, 1e-2 to 3e-2: every sine function has zero Laplacian on the boundary, where the solution has Δu = −λ = −6.2 |
| Bratu Chebyshev with weak rows | at parity with the paper's basis |
| BL viscous, lifted sine as specified | 0.11–0.15, never meets the target |
| BL viscous, the same lifting with plain sines | the best basis tested: 3.6e-4 against the paper's 9.0e-4 at P = 576, 1e-4 against 2e-4 at P = 1,024; κ about 1e9 against 1e16 |
| BL gravity | the paper's basis stays best |

### 4.6 Item 7, scaling

| | CPU | A100 (laptop RTX 5080) |
|---|---|---|
| Kovasznay time against P (N/P ≈ 3) | P^3.13 | P^2.34 |
| Kovasznay time against N (P = 1,200) | N^1.15 | N^0.99 |
| Kovasznay P = 7,500 | 474 s | 42 s, 4.5 GB on the GPU |

**Beltrami:**
- The paper's size takes 303 s on the CPU and 31 s on the GPU, with the paper's errors; C8′ passes there even on the laptop.
- 13,764 coefficients used 18.9 GB of GPU memory.
- **The largest size (22,288) is expected not to fit in the A100's 40 GB** (about 44 GB by extrapolation). The job records it as not fitting, as Section 10 asks.

### 4.7 Item 8, manufactured elasticity

**The errors stall at about 0.22 (u_x) and 0.14 (u_y) at every P,** while δ_P falls to 7e-4 and 2e-4. The systems have full rank.

**The cause is the bases' parities.** With the paper's bases, σ_xx is zero on the lateral faces for every coefficient vector, so the prescribed lateral traction (up to 0.06) cannot be met. Even with the displacement prescribed on every face, the errors stall at 0.13 and 0.04. The bases carry the paper solution's parities, which this manufactured solution breaks, so the second derivatives the equations are collocated on are not resolved near the faces.

## 5. Suggestions beyond the package

These are suggestions only; none is planned unless you ask.
1. **Item 8.** A manufactured solution compatible with the bases' parities would give the intended error ≈ δ_P comparison; so would other bases. Either is a change to your specification.
2. **Item 6b.** The plain-sine lifting supports Section 7.4's guidance for the viscous case more strongly than the basis as specified. It may be worth stating in the manuscript.
3. **Item 6a.** At P = 1,600 and ν ≤ 0.02, the undamped quasilinear iteration oscillates. A damped (line-search) LiL-Q step would show whether that is the iteration or the approximation. It would be a method change, so it belongs in the discussion, not the tables, unless you want it run.

## 6. Budget and next steps

**Grace.** The submission is 17 jobs:
- 1,250 SU requested, walltime × rate, about 606 expected (charged by elapsed time);
- item 2's contingent A100 reruns would add up to 480 SU requested, but the rehearsal says no CPU run comes near the 15-minute cap;
- every item stays within its cap of Section 1.

**FASTER (item 5).** The plan of the Stage-1 report stands (1,233 SU requested, about 658 expected) until item 5 is written to your answer.

**After your reply:**
1. I write item 5.
2. Every Stage 2 job goes in from one commit.
3. The results are assembled into the Section 12.1 layout.
4. The final report follows, as planned.

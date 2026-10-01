# Component A tuning log

Every configuration tried, in order, and every manual intervention with its
reason (Package 1 v2.0 Section 7, item 4). Implementation choices where the
package is open are recorded in the repository's DECISIONS.md (2026-09-24,
"Component A, F1 ..." and "Component A, F2 ..."): the gradient-norm balancing
form, the learning-rate schedule's reference point, no pressure pin in F1,
the L-BFGS stopping criterion, test errors off the clock for both families,
the check-A1 configuration, and the hard/soft draw. Changed by Addendum v2.2
Section 2.5 (DECISIONS.md, 2026-09-29): F1's L-BFGS runs with its
tolerances at 0; a call that does not lower the loss is followed by one call
with a fresh optimizer, and the run ends only if that call does not lower it
either (our reading of v2.0 Section 4.3's "the family's own criterion");
F1 configurations are ranked and selected on the unweighted final loss.
An L-BFGS call interrupted by the budget keeps its lowest-loss point,
logged at the time its evaluation finished; an evaluation that finishes
past the budget does not count as reached. A call (500 iterations, 625
evaluations) runs in pieces of 10 iterations, a log row after each, with
the optimizer state carried over and every point evaluated once, so the
trajectory is that of one call; in the first call after Adam the pieces
stop at 624 real evaluations rather than 625 when the evaluation limit
binds, and iteration counts can differ by one in exact-zero edge cases
(accepted by the advisor, reply to Addendum v2.2).

## Runs

- 2026-09-29 23:37 UTC | check A1 | F1 A1_plain seed 0 | cuda, float64, budget 3600 s | end: budget | final loss 9.086e-07 (unweighted 8.262e-07) | eps_u 5.776e-05
- 2026-09-29 23:51 UTC | screen | F1 F1_00 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.142e-03 (unweighted 1.618e-03) | eps_u 6.801e-03
- 2026-09-29 23:51 UTC | screen | F2 F2_00 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 4.901e-15 (unweighted None) | eps_u 5.983e-09
- 2026-09-30 00:01 UTC | screen | F1 F1_01 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.396e-06 (unweighted 1.299e-06) | eps_u 1.017e-04
- 2026-09-30 00:03 UTC | screen | F2 F2_01 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 4.113e-12 (unweighted None) | eps_u 4.113e-06
- 2026-09-30 00:11 UTC | screen | F1 F1_02 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.831e-02 (unweighted 2.517e-03) | eps_u 5.763e-03
- 2026-09-30 00:14 UTC | screen | F2 F2_02 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.098e-14 (unweighted None) | eps_u 3.255e-07
- 2026-09-30 00:21 UTC | screen | F1 F1_03 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 9.875e-05 (unweighted 2.011e-05) | eps_u 1.482e-03
- 2026-09-30 00:24 UTC | screen | F2 F2_03 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 9.077e-19 (unweighted None) | eps_u 1.710e-06
- 2026-09-30 00:32 UTC | screen | F1 F1_04 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.085e-06 (unweighted 2.035e-07) | eps_u 6.149e-02
- 2026-09-30 00:35 UTC | screen | F2 F2_04 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 3.150e-16 (unweighted None) | eps_u 1.836e-08
- 2026-09-30 00:42 UTC | screen | F1 F1_05 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 3.631e-04 (unweighted 3.631e-04) | eps_u 1.430e-03
- 2026-09-30 00:45 UTC | screen | F2 F2_05 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.309e-17 (unweighted None) | eps_u 2.298e-10
- 2026-09-30 00:52 UTC | screen | F1 F1_06 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 3.172e-04 (unweighted 8.443e-05) | eps_u 1.300e-03
- 2026-09-30 00:56 UTC | screen | F2 F2_06 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.242e-16 (unweighted None) | eps_u 9.970e-10
- 2026-09-30 01:02 UTC | screen | F1 F1_07 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.528e-01 (unweighted 1.444e-01) | eps_u 1.999e-02
- 2026-09-30 01:07 UTC | screen | F2 F2_07 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 5.432e-18 (unweighted None) | eps_u 9.572e-08
- 2026-09-30 01:12 UTC | screen | F1 F1_08 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 3.691e-06 (unweighted 1.263e-06) | eps_u 6.264e-05
- 2026-09-30 01:17 UTC | screen | F2 F2_08 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.503e-15 (unweighted None) | eps_u 1.426e-09
- 2026-09-30 01:22 UTC | screen | F1 F1_09 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 8.296e-03 (unweighted 8.296e-03) | eps_u 5.432e-02
- 2026-09-30 01:27 UTC | screen | F2 F2_09 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 6.679e-15 (unweighted None) | eps_u 2.740e-09
- 2026-09-30 01:32 UTC | screen | F1 F1_10 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.025e-05 (unweighted 7.226e-06) | eps_u 8.748e-05
- 2026-09-30 01:38 UTC | screen | F2 F2_10 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 7.913e-17 (unweighted None) | eps_u 1.559e-09
- 2026-09-30 01:43 UTC | screen | F1 F1_11 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.638e-03 (unweighted 1.638e-03) | eps_u 2.378e-03
- 2026-09-30 01:49 UTC | screen | F2 F2_11 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 6.518e-16 (unweighted None) | eps_u 1.539e-07
- 2026-09-30 01:53 UTC | screen | F1 F1_12 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.163e-05 (unweighted 1.163e-05) | eps_u 9.711e-05
- 2026-09-30 02:00 UTC | screen | F2 F2_12 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.315e-18 (unweighted None) | eps_u 1.552e-08
- 2026-09-30 02:03 UTC | screen | F1 F1_13 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 3.366e-02 (unweighted 2.584e-03) | eps_u 7.614e-03
- 2026-09-30 02:10 UTC | screen | F2 F2_13 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.702e-14 (unweighted None) | eps_u 9.320e-09
- 2026-09-30 02:13 UTC | screen | F1 F1_14 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 5.905e-04 (unweighted 1.625e-04) | eps_u 1.371e-03
- 2026-09-30 02:21 UTC | screen | F2 F2_14 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 5.913e-16 (unweighted None) | eps_u 9.862e-10
- 2026-09-30 02:23 UTC | screen | F1 F1_15 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.022e-06 (unweighted 3.083e-07) | eps_u 7.750e-05
- 2026-09-30 02:32 UTC | screen | F2 F2_15 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.425e-18 (unweighted None) | eps_u 7.587e-07
- 2026-09-30 02:33 UTC | screen | F1 F1_16 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.315e-03 (unweighted 7.305e-04) | eps_u 1.501e-03
- 2026-09-30 02:33 UTC | screen | F2 F2_16 seed 0 | cuda, float64, budget 600 s | end: mu_overflow | final loss 8.513e-31 (unweighted None) | eps_u 1.436e-02
- 2026-09-30 02:43 UTC | screen | F1 F1_17 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 8.585e-05 (unweighted 2.882e-05) | eps_u 2.683e-04
- 2026-09-30 02:44 UTC | screen | F2 F2_17 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.981e-18 (unweighted None) | eps_u 5.396e-09
- 2026-09-30 02:53 UTC | screen | F1 F1_18 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.459e-06 (unweighted 6.722e-07) | eps_u 1.014e-04
- 2026-09-30 02:55 UTC | screen | F2 F2_18 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 5.580e-12 (unweighted None) | eps_u 5.385e-06
- 2026-09-30 03:03 UTC | screen | F1 F1_19 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.109e-06 (unweighted 1.047e-06) | eps_u 3.971e-05
- 2026-09-30 03:06 UTC | screen | F2 F2_19 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.881e-10 (unweighted None) | eps_u 5.740e-07
- 2026-09-30 03:14 UTC | screen | F1 F1_20 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.516e-03 (unweighted 5.637e-04) | eps_u 5.944e-04
- 2026-09-30 03:16 UTC | screen | F2 F2_20 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 5.844e-15 (unweighted None) | eps_u 2.595e-09
- 2026-09-30 03:24 UTC | screen | F1 F1_21 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.593e-04 (unweighted 1.593e-04) | eps_u 4.266e-04
- 2026-09-30 03:26 UTC | screen | F2 F2_21 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 2.092e-16 (unweighted None) | eps_u 9.765e-09
- 2026-09-30 03:34 UTC | screen | F1 F1_22 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 3.075e-02 (unweighted 5.483e-03) | eps_u 3.749e-03
- 2026-09-30 03:37 UTC | screen | F2 F2_22 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.122e-11 (unweighted None) | eps_u 1.161e-07
- 2026-09-30 03:44 UTC | screen | F1 F1_23 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 4.637e-05 (unweighted 4.637e-05) | eps_u 2.992e-04
- 2026-09-30 03:47 UTC | screen | F2 F2_23 seed 0 | cuda, float64, budget 600 s | end: budget | final loss 1.535e-14 (unweighted None) | eps_u 1.089e-08
- 2026-09-30 03:51 UTC | full | F2 F2_16 seed 0 | cuda, float64, budget 3600 s | end: mu_overflow | final loss 8.513e-31 (unweighted None) | eps_u 1.436e-02
- 2026-09-30 03:52 UTC | full | F2 F2_16 seed 1 | cuda, float64, budget 3600 s | end: mu_overflow | final loss 4.754e-31 (unweighted None) | eps_u 9.132e-03
- 2026-09-30 03:53 UTC | full | F2 F2_16 seed 2 | cuda, float64, budget 3600 s | end: mu_overflow | final loss 1.187e-30 (unweighted None) | eps_u 7.583e-03
- 2026-09-30 04:42 UTC | full | F2 F2_16 seed 3 | cuda, float64, budget 3600 s | end: mu_overflow | final loss 8.424e-31 (unweighted None) | eps_u 9.345e-05
- 2026-09-30 04:43 UTC | full | F2 F2_16 seed 4 | cuda, float64, budget 3600 s | end: mu_overflow | final loss 2.643e-31 (unweighted None) | eps_u 1.625e-03
- 2026-09-30 04:52 UTC | full | F1 F1_04 seed 0 | cuda, float64, budget 3600 s | end: budget | final loss 3.204e-07 (unweighted 3.013e-08) | eps_u 6.154e-02
- 2026-09-30 05:46 UTC | full | F2 F2_03 seed 0 | cuda, float64, budget 3600 s | end: budget | final loss 1.900e-19 (unweighted None) | eps_u 1.569e-06
- 2026-09-30 05:54 UTC | full | F1 F1_04 seed 1 | cuda, float64, budget 3600 s | end: budget | final loss 1.010e-06 (unweighted 1.065e-07) | eps_u 5.699e-02
- 2026-09-30 06:50 UTC | full | F2 F2_03 seed 1 | cuda, float64, budget 3600 s | end: budget | final loss 3.554e-19 (unweighted None) | eps_u 2.849e-07
- 2026-09-30 06:56 UTC | full | F1 F1_04 seed 2 | cuda, float64, budget 3600 s | end: budget | final loss 1.473e-07 (unweighted 1.323e-08) | eps_u 1.263e-01
- 2026-09-30 07:53 UTC | full | F2 F2_03 seed 2 | cuda, float64, budget 3600 s | end: budget | final loss 4.171e-18 (unweighted None) | eps_u 2.132e-07
- 2026-09-30 07:58 UTC | full | F1 F1_04 seed 3 | cuda, float64, budget 3600 s | end: budget | final loss 8.303e-07 (unweighted 9.301e-08) | eps_u 8.451e-02
- 2026-09-30 08:57 UTC | full | F2 F2_03 seed 3 | cuda, float64, budget 3600 s | end: budget | final loss 5.880e-19 (unweighted None) | eps_u 3.998e-08
- 2026-09-30 09:00 UTC | full | F1 F1_04 seed 4 | cuda, float64, budget 3600 s | end: budget | final loss 4.354e-07 (unweighted 4.772e-08) | eps_u 7.442e-02
- 2026-09-30 10:00 UTC | full | F2 F2_03 seed 4 | cuda, float64, budget 3600 s | end: budget | final loss 3.698e-19 (unweighted None) | eps_u 5.960e-07
- 2026-09-30 10:01 UTC | full | F1 F1_15 seed 0 | cuda, float64, budget 3600 s | end: budget | final loss 8.603e-07 (unweighted 2.601e-07) | eps_u 7.433e-05
- 2026-09-30 11:02 UTC | full | F1 F1_15 seed 1 | cuda, float64, budget 3600 s | end: budget | final loss 4.580e-07 (unweighted 1.193e-07) | eps_u 1.156e-04
- 2026-09-30 11:03 UTC | full | F2 F2_17 seed 0 | cuda, float64, budget 3600 s | end: budget | final loss 4.088e-19 (unweighted None) | eps_u 4.560e-09
- 2026-09-30 12:03 UTC | full | F1 F1_15 seed 2 | cuda, float64, budget 3600 s | end: budget | final loss 5.639e-07 (unweighted 1.576e-07) | eps_u 6.286e-05
- 2026-09-30 12:07 UTC | full | F2 F2_17 seed 1 | cuda, float64, budget 3600 s | end: budget | final loss 1.300e-16 (unweighted None) | eps_u 2.312e-08
- 2026-09-30 13:04 UTC | full | F1 F1_15 seed 3 | cuda, float64, budget 3600 s | end: budget | final loss 8.145e-07 (unweighted 2.356e-07) | eps_u 8.689e-05
- 2026-09-30 13:10 UTC | full | F2 F2_17 seed 2 | cuda, float64, budget 3600 s | end: budget | final loss 1.163e-15 (unweighted None) | eps_u 1.813e-08
- 2026-09-30 14:05 UTC | full | F1 F1_15 seed 4 | cuda, float64, budget 3600 s | end: budget | final loss 1.075e-06 (unweighted 3.123e-07) | eps_u 1.048e-04
- 2026-09-30 14:14 UTC | full | F2 F2_17 seed 3 | cuda, float64, budget 3600 s | end: budget | final loss 4.157e-16 (unweighted None) | eps_u 1.736e-08
- 2026-09-30 15:06 UTC | full | F1 F1_18 seed 0 | cuda, float64, budget 3600 s | end: budget | final loss 9.819e-07 (unweighted 2.704e-07) | eps_u 4.545e-05
- 2026-09-30 15:18 UTC | full | F2 F2_17 seed 4 | cuda, float64, budget 3600 s | end: budget | final loss 4.006e-17 (unweighted None) | eps_u 3.672e-09
- 2026-09-30 16:08 UTC | full | F1 F1_18 seed 1 | cuda, float64, budget 3600 s | end: budget | final loss 5.192e-07 (unweighted 1.599e-07) | eps_u 2.432e-05
- 2026-09-30 17:09 UTC | full | F1 F1_18 seed 2 | cuda, float64, budget 3600 s | end: budget | final loss 7.679e-07 (unweighted 2.132e-07) | eps_u 4.371e-05
- 2026-09-30 18:10 UTC | full | F1 F1_18 seed 3 | cuda, float64, budget 3600 s | end: budget | final loss 7.602e-07 (unweighted 2.008e-07) | eps_u 4.193e-05
- 2026-09-30 19:11 UTC | full | F1 F1_18 seed 4 | cuda, float64, budget 3600 s | end: budget | final loss 4.980e-07 (unweighted 1.295e-07) | eps_u 3.139e-05
- 2026-09-30 20:20 UTC | float32 | F1 F1_04 seed 0 | cuda, adam32, budget 3600 s | end: budget | final loss 1.307e-07 (unweighted 1.292e-08) | eps_u 1.094e-01
- 2026-09-30 20:37 UTC | cpu | F2 F2_16 seed 0 | cpu, float64, budget 3600 s | end: mu_overflow | final loss 8.753e-31 (unweighted None) | eps_u 1.436e-02
- 2026-09-30 20:43 UTC | cpu | F2 F2_16 seed 1 | cpu, float64, budget 3600 s | end: mu_overflow | final loss 4.594e-31 (unweighted None) | eps_u 9.132e-03
- 2026-09-30 20:51 UTC | cpu | F2 F2_16 seed 2 | cpu, float64, budget 3600 s | end: mu_overflow | final loss 1.293e-30 (unweighted None) | eps_u 7.583e-03
- 2026-09-30 21:28 UTC | cpu | F1 F1_04 seed 0 | cpu, float64, budget 3600 s | end: budget | final loss 2.982e-06 (unweighted 2.998e-07) | eps_u 6.169e-02
- 2026-09-30 21:52 UTC | cpu | F2 F2_16 seed 3 | cpu, float64, budget 3600 s | end: budget | final loss 7.245e-14 (unweighted None) | eps_u 9.694e-05
- 2026-09-30 22:00 UTC | cpu | F2 F2_16 seed 4 | cpu, float64, budget 3600 s | end: mu_overflow | final loss 2.854e-31 (unweighted None) | eps_u 1.625e-03
- 2026-09-30 22:29 UTC | cpu | F1 F1_04 seed 1 | cpu, float64, budget 3600 s | end: budget | final loss 1.246e-05 (unweighted 1.428e-06) | eps_u 5.692e-02
- 2026-09-30 23:30 UTC | cpu | F1 F1_04 seed 2 | cpu, float64, budget 3600 s | end: budget | final loss 1.758e-06 (unweighted 1.698e-07) | eps_u 1.257e-01
- 2026-10-01 00:31 UTC | cpu | F1 F1_04 seed 3 | cpu, float64, budget 3600 s | end: budget | final loss 2.250e-06 (unweighted 2.265e-07) | eps_u 8.521e-02
- 2026-10-01 01:32 UTC | cpu | F1 F1_04 seed 4 | cpu, float64, budget 3600 s | end: budget | final loss 4.959e-06 (unweighted 5.636e-07) | eps_u 7.401e-02

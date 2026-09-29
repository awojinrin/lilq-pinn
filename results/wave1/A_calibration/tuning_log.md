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

- 2026-09-29 09:30 UTC | check A1 | F1 A1_plain seed 0 | cuda, float64, budget 3600 s | end: budget | final loss 9.128e-07 (unweighted 8.299e-07) | eps_u 5.799e-05

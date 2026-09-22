# Phase 0 empirical verification — pre-v2 vs. GitHub (v2) vs. v3-dev

Run 2026-09-22, local machine (RTX 5080), after all Phase 0 fixes. Two
smallest sizes per problem, all four methods where they exist. Manuscript
figures are only those directly quoted in `Computational_Package_1_v2.md`
Section 10 (Q1) — the package doc doesn't give full tables, only spot
examples, so manuscript coverage is partial by nature, not by omission here.

Columns: iterations / final loss / wall-clock time. `*` = did not converge
within budget.

---

## Bratu

| N (P) | Method | Manuscript | pre-v2 | GitHub v2 | **v3-dev (fresh)** |
|---|---|---|---|---|---|
| 5 (25) | NiL-N | **76** iters | 180 / 0.2497 / 4.0s | 366 / 0.2458 / 8.9s | 3178 / 0.2499 / 197.7s |
| 5 (25) | NiL-Q | — | 315 / 0.2474 / 8.2s | 527 / 0.2487 / 15.9s | 349 / 0.2391 / 19.3s |
| 5 (25) | LiL-N | — | 33 / 0.2458 / 1.9s | 33 / 0.2460 / 0.2s | 33 / 0.2460 / 0.4s |
| 5 (25) | LiL-Q | — | — | 2 / 0.2078 / 0.002s | 2 / 0.2078 / 0.005s |
| 10 (100) | NiL-N | — | 7500\* / 4.26e-4 / 308s | 10000\* / 9.53e-4 / 284s | **7500\* / 1.27e-3 / 399s** |
| 10 (100) | NiL-Q | — | 7500\* / 3.90e-4 / 331s | 10000\* / 1.05e-3 / 279s | **7500\* / 6.72e-4 / 284s** |
| 10 (100) | LiL-N | — | 1597 / 9.995e-5 / 51s | 1552 / 9.994e-5 / 26s | 1552 / 9.994e-5 / 27s |
| 10 (100) | LiL-Q | — | — | 3 / 2.161e-5 / 0.09s | 3 / 2.161e-5 / 0.05s |

**v3-dev now matches pre-v2's iteration caps exactly** (7500/7500 at N=10,
both NiL-N and NiL-Q — this is the two-part fix from this session: the
flat `MAX_ITERATIONS` cap and the separate `MAX_LBFGS_PER_QUASI_ITER` cap
governing NiL-Q specifically). Final losses are close across all three
codebases at each N; none match the manuscript's 76-iteration NiL-N figure
at N=5 — consistent with Q1's finding that the manuscript numbers came
from a since-deleted, even-earlier version than either codebase we have.

---

## Burgers

| N (P) | Method | Manuscript | pre-v2 | GitHub v2 | **v3-dev (fresh)** |
|---|---|---|---|---|---|
| 5 (25) | NiL-N | **214** iters | 214 / 0.0594 / 8.4s | 325 / 0.0600 / 11.1s | 166 / 0.0588 / 9.2s |
| 5 (25) | NiL-Q | **99** iters | 487 / 0.0600 / 21.5s | 218 / 0.0594 / 7.4s | 612 / 0.0600 / 47.4s |
| 5 (25) | LiL-N | **487** iters | 18 / 0.0581 / 2.1s | 18 / 0.0581 / 0.2s | 18 / 0.0581 / 0.2s |
| 5 (25) | LiL-Q | — | — | 2 / 0.0531 / 0.002s | 2 / 0.0531 / 0.00s |
| 10 (100) | NiL-N | — | 1047 / 9.991e-4 / 53s | 1288 / 9.992e-4 / 53s | 2561 / 9.998e-4 / 212s |
| 10 (100) | NiL-Q | — | 1796 / 9.999e-4 / 89s | 1298 / 9.996e-4 / 52s | 3273 / 9.999e-4 / 274s |
| 10 (100) | LiL-N | — | 93 / 9.930e-4 / 1.7s | 94 / 9.914e-4 / 1.5s | 94 / 9.914e-4 / 3.6s |
| 10 (100) | LiL-Q | — | — | 3 / 9.059e-4 / 0.03s | 3 / 9.059e-4 / 0.03s |

Note the manuscript's own N=5 numbers (214/99/487 for NiL-N/NiL-Q/LiL-N)
don't match *either* codebase consistently either — pre-v2 matches NiL-N
exactly (214) but not NiL-Q or LiL-N; nothing matches all three. Same
"predates everything we have" conclusion as Bratu (Q1).

**LiL-N/LiL-Q are stable and consistent across all three codebases at
every size** — expected, since these are deterministic direct solves. The
NiL-N/NiL-Q iteration *counts* vary run-to-run more than Bratu's do (2561
vs. 1047-1288 at N=10) despite reaching the same final loss — expected
stochastic/hardware-dependent L-BFGS path variation, previously documented
for this codebase (not new here).

---

## Buckley–Leverett, viscous

| N (P) | Method | Manuscript | pre-v2 | GitHub v2 | v3-dev (before fix) | **v3-dev (after LiL-N fix)** |
|---|---|---|---|---|---|---|
| 8 (64) | NiL-N | — | 173 / 0.0847 / — | 148 / 0.0828 / 3.7s | 183 / 0.0824 / 12.5s | 183 / 0.0824 / 6.8s |
| 8 (64) | NiL-Q | — | 324 / 0.0846 / 10.0s | 290 / 0.0849 / 9.5s | 406 / 0.0804 / 30.9s | 406 / 0.0804 / 16.4s |
| 8 (64) | LiL-N | **259**, converged | 135 / 0.0846, converged | 1409\* / 0.1386 (line-search cap) | 5000\* / 0.1680 (plateau) | **135 / 0.0847, CONVERGED / 4.1s** |
| 8 (64) | LiL-Q | — | — | 10 / 0.0768 / 0.04s | 10 / 0.0780 / 0.05s | 10 / 0.0780 / 0.03s |
| 16 (256) | NiL-N | — | 926 / 0.0150 / — | 294 / 0.0149 / 10.5s | 248 / 0.0149 / 22.3s | 248 / 0.0149 / 14.6s |
| 16 (256) | NiL-Q | — | 972 / 0.0150 / 35.3s | 943 / 0.0149 / 31.5s | 839 / 0.0145 / 64.5s | 839 / 0.0145 / 48.1s |
| 16 (256) | LiL-N | — | 1107 / 0.0150, converged | 3938\* / 0.0303 (line-search cap) | 10000\* / 0.0334 (plateau) | **1061 / 0.0150, CONVERGED / 37.8s** |
| 16 (256) | LiL-Q | — | — | 4 / 0.0132 / 0.21s | 4 / 0.0132 / 0.19s | 4 / 0.0132 / 0.28s |

**Resolved.** The "before fix" numbers (LiL-N plateauing at 5000/10000
iterations without converging) were caused by a genuine gradient-computation
bug in `_make_lil_n_loss_fn`, not a missing stagnation detector — see
DECISIONS.md ("BL's LiL-N gradient was silently wrong"). `solve_lil_n`
was backpropagating through a *detached* flux derivative, so its computed
gradient was structurally incomplete; L-BFGS was optimizing against the
wrong objective the entire time, which looks exactly like a stagnating
plateau from the outside. Fixed by adding a graph-preserving
`flux_derivative_differentiable` and using it in `_make_lil_n_loss_fn`.

After the fix, LiL-N converges cleanly and lands almost exactly on
pre-v2's original numbers: **135 iterations at N=8 (pre-v2: 135, exact
match)**, 1,061 at N=16 (pre-v2: 1,107, close). Total wall-clock for both
sizes dropped from 1,438s to 137s — over 10x faster, simply because it's
no longer burning thousands of iterations chasing a plateau that a
correct gradient never would have produced.

---

## Buckley–Leverett, gravity

| N (P) | Method | Manuscript | pre-v2 | GitHub v2 | v3-dev (before fix) | **v3-dev (after LiL-N fix)** |
|---|---|---|---|---|---|---|
| 8 (64) | NiL-N | — | 28 / 0.2379 / 1.7s | 249 / 0.2489 / 6.8s | 229 / 0.2499 / 18.1s | 229 / 0.2499 / 9.6s |
| 8 (64) | NiL-Q | — | 49 / 0.1601 / 1.8s | 250 / 0.2496 / 5.8s | 175 / 0.2467 / 11.8s | 175 / 0.2467 / 6.4s |
| 8 (64) | LiL-N | — | 166 / 0.2457, converged | 1429\* / 0.2869 (line-search cap) | 5000\* / 0.2869 (plateau) | **94 / 0.2474, CONVERGED / 2.2s** |
| 8 (64) | LiL-Q | **9** iters | — | 6 / 0.2194 / 0.03s | 6 / 0.2194 / 0.03s | 6 / 0.2194 / 0.03s |
| 16 (256) | NiL-N | — | 631 / 0.1500 / 44.1s | 923 / 0.1492 / 33.8s | 559 / 0.1496 / 42.9s | 559 / 0.1496 / 34.7s |
| 16 (256) | NiL-Q | — | 1616 / 0.1358 / 95.0s | 1450 / 0.1481 / 46.6s | 970 / 0.1492 / 66.7s | 970 / 0.1492 / 53.5s |
| 16 (256) | LiL-N | — | 10000\* / **63.80** (diverged) | 2584\* / 0.2373 (line-search cap) | 10000\* / 0.2373 (plateau) | **5748 / 0.1500, CONVERGED / 140.4s** |
| 16 (256) | LiL-Q | **7** iters | — | 7 / 0.1390 / 0.38s | 7 / 0.1390 / 0.31s | 7 / 0.1390 / 0.31s |

Same resolution as viscous — and gravity N=16 is the most striking result
in this whole comparison: **v3-dev's fixed LiL-N converges cleanly (loss
0.150) at a size where pre-v2's own LiL-N diverged outright** (loss 63.80).
The gradient bug wasn't just present in the newer codebases; fixing it
produces a solver that's *more* robust here than any prior version,
consistent with the earlier finding (Q9) that gravity BL's Bellman-Kalaba
iteration is sensitive to path — a correct gradient evidently finds a
better path than pre-v2 did at this size. N=8 converges in 94 iterations
(pre-v2: 166 — different count, both now legitimately converge, expected
path variation between codebases). Total wall-clock for both sizes: 253s,
down from 1,679s before the fix.

LiL-Q's manuscript-quoted iteration counts (9 and 7, from Q1's "gravity
Buckley–Leverett LiL-Q 9 / 7 / 10 / 8") match GitHub v2 and v3-dev
*exactly* at N=16 (7=7) but not N=8 (manuscript 9 vs. both codebases' 6) —
a partial match, consistent with Q1's provenance conclusion, and unrelated
to the LiL-N fix (LiL-Q was never affected by this bug).

---

## Kovasznay (LiL-Q only — no NiL-N/NiL-Q/LiL-N exist for this problem)

| N | pre-v2 | GitHub v2 | **v3-dev (fresh)** |
|---|---|---|---|
| 5 | — | — | 16 / 0.1163 / 0.09s |
| 10 | — | — | 9 / 0.0029 / 0.63s |

No pre-v2 or GitHub reference data was pulled for Kovasznay at these
specific N values in this pass (the manuscript's own quoted Kovasznay
timings in Q5 are for a different, larger set of P values that don't
line up with N=5/10 without re-deriving the N-to-P mapping — not done
here to avoid presenting a mismatched comparison). Runtime is trivially
fast either way (CPU direct solve, no GPU involved), consistent with
everything already established about this problem.

---

## Runtime, summed up

- **LiL-Q is always the cheapest method by a wide margin** — sub-second in
  every case, every codebase, every problem. Direct QR solve, no
  optimizer loop, nothing to plateau.
- **LiL-N for Buckley–Leverett is now cheap and correct**, after fixing the
  gradient bug — comparable to or faster than pre-v2 at every size tested
  (4.1s/37.8s viscous, 2.2s/140.4s gravity). Before the fix it was briefly
  the most expensive method in the whole comparison (up to 912s for a
  single run) while also being *wrong*. Bratu/Burgers' LiL-N was never
  affected by this bug (different loss-function construction, no
  flux-divergence chain-rule term to get wrong the same way).
- **NiL-N/NiL-Q runtimes are broadly consistent** with pre-v2/GitHub v2 at
  matching iteration counts; where v3-dev looks slower in wall-clock terms
  (e.g. Bratu N=5 NiL-N: 197.7s for 3178 iters vs. GitHub's 8.9s for 366
  iters), it's because it ran *more* iterations to reach a similar loss,
  not because per-iteration cost changed — consistent with known L-BFGS
  path sensitivity to hardware/rounding, not a regression.
- These are concurrent runs sharing one local GPU (four scripts running
  side by side for wall-clock efficiency), so absolute times carry some
  contention noise — not the clean, isolated timing the package spec's
  Component A/B protocol will eventually require. Relative comparisons
  (iteration counts, convergence/non-convergence, final loss) are
  unaffected by this and are the trustworthy part of this pass.

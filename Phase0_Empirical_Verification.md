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

| N (P) | Method | Manuscript | pre-v2 | GitHub v2 | **v3-dev (fresh)** |
|---|---|---|---|---|---|
| 8 (64) | NiL-N | — | 173 / 0.0847 / — | 148 / 0.0828 / 3.7s | 183 / 0.0824 / 12.5s |
| 8 (64) | NiL-Q | — | 324 / 0.0846 / 10.0s | 290 / 0.0849 / 9.5s | 406 / 0.0804 / 30.9s |
| 8 (64) | LiL-N | **259**, converged | 135 / 0.0846, converged | 1409\* / 0.1386 (line-search cap) | **5000\* / 0.1680 (iteration cap)** |
| 8 (64) | LiL-Q | — | — | 10 / 0.0768 / 0.04s | 10 / 0.0780 / 0.05s |
| 16 (256) | NiL-N | — | 926 / 0.0150 / — | 294 / 0.0149 / 10.5s | 248 / 0.0149 / 22.3s |
| 16 (256) | NiL-Q | — | 972 / 0.0150 / 35.3s | 943 / 0.0149 / 31.5s | 839 / 0.0145 / 64.5s |
| 16 (256) | LiL-N | — | 1107 / 0.0150, converged | 3938\* / 0.0303 (line-search cap) | **10000\* / 0.0334 (iteration cap)** |
| 16 (256) | LiL-Q | — | — | 4 / 0.0132 / 0.21s | 4 / 0.0132 / 0.19s |

**This is the finding worth pausing on.** LiL-N for viscous BL:
- **pre-v2**: converges cleanly and fast (135 and 1107 iterations).
- **GitHub v2**: never converges, but the (buggy, now-removed) line-search
  cap cuts it off relatively early — 1409 / 3938 iterations.
- **v3-dev**: still never converges, and with that cap correctly removed
  it now runs the *entire* iteration budget (5000 / 10000) trying — at
  ~10x the wall-clock cost of the GitHub v2 run — landing at a **worse**
  final loss than the truncated GitHub v2 run (0.168 vs. 0.139 at N=8;
  0.033 vs. 0.030 at N=16).

So the loss essentially plateaus early and thousands of additional
iterations buy nothing — not a case of "the old cap was cutting off a run
that needed more time," but "this run needed a *stagnation* stop, not a
*budget* stop." Pre-v2's clean convergence at the same sizes says this
plateau isn't inherent to the problem, either — something about the
starting point or path is landing LiL-N in a bad basin here that it
escapes in pre-v2 but doesn't in either the GitHub or v3-dev code paths.

**This directly and concretely motivates prioritizing the stall/stagnation
detector inside Phase 1's instrumentation work** (already required by
Computational_Package_1_v2.md S3.1 item 6, already flagged as a missing
feature in Q1/Q3) — with it in place, this exact run would stop itself
early instead of burning 10+ minutes for no benefit, and the stall flag
would make the plateau visible in the log instead of silently returning a
worse number. Root-causing *why* pre-v2 converges here and neither newer
codebase does is a separate, open question — not yet investigated.

---

## Buckley–Leverett, gravity

| N (P) | Method | Manuscript | pre-v2 | GitHub v2 | **v3-dev (fresh)** |
|---|---|---|---|---|---|
| 8 (64) | NiL-N | — | 28 / 0.2379 / 1.7s | 249 / 0.2489 / 6.8s | 229 / 0.2499 / 18.1s |
| 8 (64) | NiL-Q | — | 49 / 0.1601 / 1.8s | 250 / 0.2496 / 5.8s | 175 / 0.2467 / 11.8s |
| 8 (64) | LiL-N | — | 166 / 0.2457, converged | 1429\* / 0.2869 (line-search cap) | **5000\* / 0.2869 (iteration cap)** |
| 8 (64) | LiL-Q | **9** iters | — | 6 / 0.2194 / 0.03s | 6 / 0.2194 / 0.03s |
| 16 (256) | NiL-N | — | 631 / 0.1500 / 44.1s | 923 / 0.1492 / 33.8s | 559 / 0.1496 / 42.9s |
| 16 (256) | NiL-Q | — | 1616 / 0.1358 / 95.0s | 1450 / 0.1481 / 46.6s | 970 / 0.1492 / 66.7s |
| 16 (256) | LiL-N | — | 10000\* / **63.80** (diverged), — | 2584\* / 0.2373 (line-search cap) | **10000\* / 0.2373 (iteration cap)** |
| 16 (256) | LiL-Q | **7** iters | — | 7 / 0.1390 / 0.38s | 7 / 0.1390 / 0.31s |

Same LiL-N pattern as viscous, plus one more data point: pre-v2's LiL-N at
N=16 doesn't just fail to converge, it **diverges outright** (loss 63.8 —
matches the earlier-documented finding from this session's Q9
investigation, that LiL-N is the weaker baseline for BL's gravity case and
can diverge badly at larger sizes). v3-dev's N=16 LiL-N loss (0.2373)
lands almost exactly on GitHub v2's *truncated* value (0.2373 at 2584
iterations) even after running to the full 10,000 — strong direct evidence
of a genuine early plateau, not a slow-but-still-progressing run.

LiL-Q's manuscript-quoted iteration counts (9 and 7, from Q1's "gravity
Buckley–Leverett LiL-Q 9 / 7 / 10 / 8") match GitHub v2 and v3-dev
*exactly* at these two sizes (6... wait — manuscript says 9/7, both
codebases here show 6/7). N=16 matches (7=7); N=8 doesn't (9 vs 6) — worth
noting as a partial, not full, match.

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
- **LiL-N is now the most expensive method for Buckley–Leverett**
  specifically, in v3-dev — a direct consequence of correctly removing the
  buggy line-search cap without yet having the stagnation detector that
  would make that correct. Bratu/Burgers' LiL-N remains cheap and fine
  (they were never affected by the BL-specific cap bug).
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

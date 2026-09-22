# Decisions affecting results

A running log of implementation decisions in this codebase that could change
*computed* results -- hyperparameters, tolerances, grid densities, precision,
stopping criteria -- as distinct from ordinary bug fixes with a single
unambiguous correct behavior. Each entry says what was decided, why, exactly
what it affects, and how to revisit it later if circumstances change.

Cross-references `Post-JCP/Codebase_v3_Proposal.md` for the fuller
investigation behind each entry, and Section 10 (Q1-Q10) of
`Post-JCP/reponse_Package1_v2.md` for the original forensic findings.

Format: newest first.

---

## 2026-09-22 -- Bratu's N=10 iteration cap: reverted 10,000 -> 7,500

`experiments/run_bratu.py`'s `MAX_ITERATIONS` schedule is per-size, not one
flat number; only the N=10 (P=100) entry changed during the consolidation.
Confirmed against `pre-v2-local-codebase/Bratu/run_bratu_experiments.py`:
`{5: 5000, 10: 7500, 15: 10000}` there vs. `{5: 5000, 10: 10000, 15: 10000}`
before this change. N=5 and N=15 were already unchanged; only N=10 reverted.

**Open flag, not yet decided:** while checking this, `MAX_LINE_SEARCHES`
turned out to *also* differ for Bratu, on every entry --
`{5: 24000, 10: 30000, 15: 30000}` pre-GitHub vs. the current
`{5: 15000, 10: 25000, 15: 25000}` -- something Q1's original writeup
claimed was "unchanged" (true for Burgers, checked and confirmed identical
in both versions, but evidently not checked carefully enough for Bratu at
the time). Unlike Buckley-Leverett, Bratu has always had a real,
deliberately-calibrated line-search cap in both codebases -- this isn't a
"restore vs. remove" question the way BL's was, it's "which of two
different deliberate numbers is right." Left as-is (current GitHub values)
pending an explicit decision on whether to revert these too.

---

## 2026-09-22 -- Buckley-Leverett's line-search cap: made inert instead of removed

Pre-GitHub BL had no evaluation-based termination condition at all -- a
pure iteration-count loop, `func_eval_counter` tracked for logging only.
The consolidation added `max_line_searches=100,000` as a second
loop-termination condition. Since each `.step()` is separately capped at
`max_eval=15` (hardcoded in `lilq.solvers`, shared by all problems), the
true worst-case eval count for a never-converging run was always
`max_iterations * 15` regardless -- meaning the added 100,000 cap could
actually cut a run short *before* `max_iterations` did in some
configurations, which pre-GitHub BL would never have done.

Rather than hardcoding a literal "big enough" number, `BLOptConfig` now
derives `max_line_searches = max_iterations * 15` in `__post_init__` when
not explicitly overridden -- it can never bind first by construction,
including if `max_iterations` is changed later, and an explicit override
is still honored if anyone wants a genuinely tighter cap for a specific run.

---

## 2026-09-22 -- NN/LiL pretraining fit-grid density: reverted to pre-GitHub, per-problem formulas

**What changed.** Two separate fixes:

1. **`pretrain_nn`'s callers** (Bratu/Burgers/BL's NiL-N and NiL-Q pretrain
   calls): were passing a flat `n_grid=50` (2,500 fit points) for every
   problem size. Now each problem computes its own historical value via the
   new `nn_pretrain_grid_side(target_dof, floor)` helper in
   `lilq/pretraining.py`:

   | Problem | floor | target_dof |
   |---|---|---|
   | Bratu | 50 | `N_x * N_y` |
   | Burgers | 100 | `N_x * N_t` |
   | Buckley-Leverett | 50 | `N_x * N_t` |

2. **`pretrain_lil`'s internal formula**: was `n_side = max(n_grid, ceil(sqrt(2*n_basis)))`
   (a structural bug -- comparing a floor meant for a *total point count*
   against an already-square-rooted quantity). Now correctly
   `n_side = sqrt(max(100, 2*n_basis))`, floor=100, matching all three
   problems' identical pre-GitHub formula exactly (this one *was* shared
   across Bratu/Burgers/BL historically, unlike the NN-pretrain floors above).

**Why.** The August-June 2026 consolidation flattened three separate
per-problem pretraining implementations into shared `lilq/pretraining.py`
functions, and lost each problem's own grid-density formula in the process
(Q1, item 1). Verified by reading the pre-GitHub source for all three
problems directly (`pre-v2-local-codebase/{Bratu,Burgers,BL}/*_core.py`).

**What it affects.** The exact starting point of NiL-N/NiL-Q's neural
network (via `pretrain_nn`) and LiL-N/LiL-Q's initial coefficients (via
`pretrain_lil`) before main training begins -- a coarser or finer pretraining
fit changes the initial loss landscape position, which can change downstream
iteration counts even though it never changes the converged answer's
correctness.

**Note on magnitude, since it's counterintuitive:** at every P actually used
in this package's Component B (Bratu <=225, Burgers 625, BL <=1,024), the
*reverted* (historical) grids are **smaller** than the flat 2,500-point grid
that was there before this change -- e.g. Bratu P=25 now fits on a 7x7=49
point grid, not 50x50=2,500. The original Q1 write-up characterized this
backwards ("the GitHub version uses a fixed, smaller grid... for training
speed"); it's actually the fixed grid that was larger at these sizes.

**Revisit later, if:**
- We want one *consistent* formula across all three problems instead of
  three different ones (would deviate from paper-faithful reproduction, but
  is simpler to maintain and reason about).
- Profiling shows pretraining grid density measurably affects final
  convergence quality (not just iteration count) at some P we care about,
  in which case a deliberately denser grid might be worth choosing over
  historical fidelity.
- We want a single shared floor/multiplier instead of two different NN
  floors (50 vs 100) that have no obvious reason to differ beyond "that's
  what each problem's author happened to pick."

---

## 2026-09-22 -- Burgers basis comparison: ELM class corrected to Xavier-scaled

`experiments/run_burgers_basis_comparison.py`'s `'elm'` branch instantiated
`ELMBasis2D` (fixed bound, +/-1.2247 regardless of size) instead of
`ELMBasis2D_Xavier` (bound scales with size, ~+/-0.098 at n_hidden=625).
This is a plain bug, not a judgment call -- the paper's Table 3/4 ELM row
(final loss 5.0e-2) was produced with the Xavier-scaled class (Q8); the
un-Xavier'd class produces a different, much lower loss (~1.7e-5), matching
the repository's own stored (wrong-class) reference result almost exactly.
One-line fix, no future revisit needed.

---

## 2026-09-22 -- BL's LiL-Q off-by-one iteration count: found already fixed, no action taken

Q1 described a real bug in the pre-GitHub Buckley-Leverett code
(`solve_quasilinear_lil` reported `len(metrics.data[...])`, which included a
pre-loop initial-state snapshot, as the iteration count -- one higher than
solves actually performed). Checked the current shared
`lilq.solvers.solve_lil_q` (used identically by Bratu/Burgers/BL) directly:
it counts via `n_quasi_iters = quasi_iter + 1` set inside the loop, which
does **not** have this bug. The June 2026 consolidation fixed it as a side
effect of merging three separate implementations into one correct shared
one. No code change made; a regression test
(`tests/test_solve_lil_q_iteration_count.py`) locks in the correct behavior
so it can't silently regress in a future refactor.

---

## 2026-09-22 -- BL's `tolerance_grad`: kept at 1e-8 (no revert)

Pre-GitHub BL's L-BFGS calls never set `tolerance_grad` explicitly, so they
silently ran on PyTorch's own default (1e-7). The current shared solver
explicitly sets 1e-8 for every problem including BL. Considered reverting
BL specifically back to the looser implicit value, but: Bratu's and
Burgers' pre-GitHub *main-solve* optimizers already explicitly set
`tolerance_grad=1e-8` (identical to today's value) -- only their
*pretraining* step used a tighter 1e-9. So 1e-8 was already the deliberate
standard for two of three problems before the consolidation; BL's 1e-7
looks like an omission, not a considered choice. Decision: keep 1e-8
uniformly. No code change was needed (current code already does this);
this entry exists purely to record the decision was made deliberately,
not left as an unexplained side effect of the consolidation.

---

## 2026-09-22 -- Darcy's float32 neural-net path: left as-is, confirmed out of scope

`problems/darcy.py`'s `DarcyPINN` class force-casts its three MLPs to
float32 (`lilq/solvers.py` sets float64 as the process-wide default for
every other problem). Checked which code path Computational_Package_1_v2.md
actually requires for Darcy (Section 3.3): `solve_lilq_darcy`, which uses
basis objects (float64, confirmed) -- not `DarcyPINN` at all. Nothing in the
current package calls `run_nil_n_darcy`/`DarcyPINN`. Decision: leave it
alone; revisit only if a future package brings Darcy's NN comparison into
scope, at which point the float64-everywhere requirement (Section 2 of the
package spec) would need to be satisfied there too.

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

See also `Phase0_Empirical_Verification.md` for the full pre-v2 / GitHub /
v3-dev three-way comparison run.

---

## 2026-09-22 -- Provenance capture (hardware.json/environment.txt) and multi-seed harness added

**Phase 1, batch 2.** Two new, independent, tested building blocks
required by Computational_Package_1_v2.md Section 2:

**`lilq/provenance.py`** -- captures everything Section 2 asks for:
CPU/GPU info (`nvidia-smi` for GPU name/driver/memory, `/proc/cpuinfo` on
Linux or `platform.processor()` elsewhere for CPU model), the
`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` environment
variables actually in effect, numpy/scipy/torch versions, verbatim
`numpy.show_config()`/`scipy.show_config()` text, and the git commit hash
+ branch + any uncommitted diff. Every capture function degrades to
`None`/`"available": False` on a missing tool rather than raising --
provenance capture must never be why a multi-hour run crashes.

This directly retires the "which run/code/machine produced this number"
class of question that most of the earlier Q1/Q2/Q5/Q6 investigation was
spent answering after the fact (Codebase_v3_Proposal.md S2.5). Confirmed
live: the first real capture on this machine shows
`blas_thread_env: {OMP_NUM_THREADS: null, OPENBLAS_NUM_THREADS: null,
MKL_NUM_THREADS: null}` -- directly, immediately visible confirmation of
Q5's finding (no thread-pinning is set anywhere), instead of something
that took code-reading to establish.

Wired into all 8 real experiment scripts (`run_bratu.py`, `run_burgers.py`,
`run_bl.py`, `run_kovasznay.py`, `run_elasticity.py`, `run_beltrami.py`,
`run_darcy.py`, `run_burgers_basis_comparison.py`) via a new
`save_run_provenance()` helper in `exp_utils.py`, called once per script
invocation right alongside the existing `save_master_results()` call --
not `run_all_dry.py`, which is a fast pipeline smoke test, not a real
timed run. Verified end-to-end: a real Kovasznay run now writes
`hardware.json`/`environment.txt` into its results directory automatically.

**Note, not yet acted on:** the spec says these thread-count variables
must be *set*, not just recorded -- BLAS reads them at library load time,
so setting them requires happening before numpy/scipy is first imported
(the process environment, a SLURM job script, or the very first lines of
an entry point). This module can only report what's currently in effect.
Actually setting them is a separate, still-open task.

**`lilq/multiseed.py`** -- implements the spec's "every stochastic method
runs at seeds {0,1,2} (Component B) / {0-4} (Component A), report
median/min/max, never a single run" requirement. `run_multiseed(runner,
config, seeds, *args, **kwargs)` calls a `run_nil_n`/`run_nil_q`-style
function once per seed via `dataclasses.replace` (never mutates the
config passed in), collects each run's summary dict (always the last
element of the returned tuple, true for every `run_*` function in
`problems/*.py`), and aggregates numeric fields into median/min/max plus
a convergence *rate* for the boolean `converged` field (a median of
booleans isn't meaningful; a rate is).

Deliberately scoped to NiL-N/NiL-Q only -- LiL-N/LiL-Q have no random
initialization of their own (coefficients come from a deterministic
least-squares pretrain fit) and the spec explicitly exempts them
("LiL-N from zero is deterministic; one run", Section 3.4); sweeping
seeds for them would only vary the collocation set, which the spec asks
to keep fixed at seed 42 instead. The utility itself is runner-agnostic
and doesn't special-case this -- it's a caller decision which methods to
sweep.

**Not yet wired into the existing experiment scripts.** Wiring this into
`run_bratu.py`/`run_burgers.py`/`run_bl.py`'s NiL-N/NiL-Q calls would
meaningfully change those scripts' output format and roughly triple their
NiL-method runtime -- a bigger, more disruptive change than provenance
capture's "add two files, nothing else changes." Since the real target
for this requirement is Section 3.4's instrumented reruns (which will
restructure these scripts substantially anyway -- the `iterations.csv`
logger, stall detection, etc.), wiring it in now and restructuring again
later would be duplicate work. Verified instead via a real integration
test: three actual seeds against Bratu's `run_nil_n` at a tiny size,
confirming genuinely different results per seed (final loss 0.31 / 0.25 /
0.063) and correct aggregation.

---

## 2026-09-22 -- torch.set_default_dtype() import-time side effect removed

**Phase 1, batch 1 (second half).** `lilq/solvers.py` used to call
`torch.set_default_dtype(torch.float64)` at module import time -- a
process-wide mutation that silently made every later `nn.Module`
construction anywhere in the process default to float64, regardless of
which file did the constructing, based on import order rather than an
explicit choice. Same class of problem as `set_seed`'s old unconditional
CUDA touching (previous entries): a shared setup step mutating global
state instead of taking explicit configuration -- and, per
`Codebase_v3_Proposal.md` S3, the two were always meant to be fixed
together.

**Fix:** `MLP` (`lilq/nn.py`) now takes its own `dtype: torch.dtype =
torch.float64` parameter, passed directly to each `nn.Linear(...)`
construction. The default preserves every current NiL-N/NiL-Q call
site's existing behavior with zero call-site changes needed (Bratu,
Burgers, and BL all construct `MLP(...)` with no explicit dtype and
correctly keep getting float64). The `torch.set_default_dtype` call
itself was deleted from `lilq/solvers.py`.

Audited every bare (no explicit dtype) `torch.tensor`/`linspace`/`zeros`/
`full`/`empty` construction across `problems/`, `lilq/`, and
`experiments/` (grep, cross-checked against multi-line calls to avoid the
false negatives a naive single-line grep would give) and made each one
explicit:
- `problems/bratu.py`: the IC-loss placeholder and the evaluation-grid
  `linspace` calls -> `dtype=torch.float64`.
- `problems/burgers.py`, `problems/buckley_leverett.py`: same
  evaluation-grid `linspace` pattern -> `dtype=torch.float64`.
- `problems/darcy.py`: `DarcyPINN`'s three `MLP(...)` calls now pass
  `dtype=torch.float32` directly instead of constructing at float64 and
  immediately downcasting via `.float()`.

**Bonus find while auditing Darcy:** `DarcyPINN.xleft`/`.xright` (two
bare `torch.zeros`/`torch.full` calls) had **no explicit dtype at all**,
silently inheriting float64 from the global default while every other
tensor in the same class (`xbot`, `xtop`, `xpde`, etc., all built via an
explicit-float32 `_t()` helper) was float32 -- a real, if currently
harmless (no failure observed, likely masked by implicit type promotion
somewhere downstream), latent inconsistency that would have become a
silent float32 gap once the global default was removed. Fixed to
`dtype=torch.float32`, matching the rest of the class's clear intent.
Darcy's PINN path remains out of scope for this package (Phase 0), but
since this codebase-wide dtype audit touched it anyway, worth fixing
while here rather than leaving a known inconsistency for later.

**Verified:**
- `tests/test_dtype_explicit.py`: importing `lilq.solvers` in a fresh
  subprocess leaves `torch.get_default_dtype()` unchanged; `MLP()`
  defaults to float64 even under deliberately hostile global state
  (`torch.set_default_dtype(torch.float32)` set immediately before
  construction); `MLP(dtype=torch.float32)` still works for Darcy.
- End-to-end: ran real NiL-N solves for Bratu, Burgers, and BL and
  confirmed float64 parameters and float64 evaluation-grid output;
  constructed a real `DarcyPINN` and confirmed float32 parameters and
  (now-fixed) float32 `xleft`/`xright`.

---

## 2026-09-22 -- Beltrami's real slowdown cause: unconditional per-iteration SVD, not set_seed

**Phase 1, batch 1.** While verifying the `set_seed` fix below actually
resolved Beltrami's known slowdown (Q6: paper ~297s, GitHub/v3-dev
~513-539s), a same-machine, same-moment, three-way controlled comparison
told a different story than the original Q6 investigation had concluded:

| Run (today, same machine, back to back) | Time |
|---|---|
| pre-GitHub codebase, actual code, run fresh | **302.0s** |
| v3-dev, `set_seed` new default (`deterministic_cuda=False`) | 539.0s |
| v3-dev, `set_seed` old behavior forced (`deterministic_cuda=True`) | 558.8s |

The `set_seed` fix only accounts for ~20s (539 vs. 559) -- nowhere near
the ~1.7x originally attributed to it. That original comparison was made
hours apart in this same session and wasn't a clean, isolated A/B test;
running pre-GitHub's *actual* code right now (302.0s, matching its
original ~297s closely) ruled out "the machine is just slower today" and
confirmed a real, current, code-level gap the `set_seed` fix doesn't explain.

**Real cause, found by reading `problems/beltrami.py`'s solve loop
directly:** `solve_beltrami` computes `np.linalg.cond(A_sys)` -- a full
SVD on the system matrix, P_total=7,984 columns -- **unconditionally,
every outer iteration**, and nothing downstream ever reads the result
(`history['cond_number']` is appended to and never consumed anywhere).
Pre-GitHub's `beltrami_core.py` has the identical computation but gates
it behind `analyze_svd=False` (default off) -- confirmed by reading that
file directly, not by assumption.

This also isn't just an efficiency gap -- it's the *wrong* algorithm per
the computational package spec: Section 3.1 item 8 explicitly requires
full-SVD conditioning only for $P \le 3{,}200$, and prescribes a cheaper
pivoted-QR check **at the final iterate only** for Beltrami specifically
(P=7,984, over that threshold), precisely because per-iteration SVD isn't
practical at this scale. The unconditional call was doing the
spec-prohibited expensive thing by default.

**Fix (stopgap):** added `analyze_conditioning: bool = False` to
`solve_beltrami`, matching pre-GitHub's `analyze_svd` pattern exactly --
gated the `np.linalg.cond` call behind it, logging `nan` when disabled
rather than silently shortening the history list. Verified:
`experiments/run_beltrami.py` never requested it, so no caller needed
updating. Reran at full scale after the fix: **324.0s** -- matching
pre-GitHub's 302.0s within normal run-to-run variance.

**Not the final fix.** This flag is a stopgap that restores correct
default performance now. The real, spec-correct replacement -- SVD for
$P\le3{,}200$, pivoted QR at the final iterate for Beltrami -- belongs to
Phase 1's `iterations.csv` instrumentation work (Codebase_v3_Proposal.md
S2.1), where conditioning logging is being built properly for every
problem anyway. Revisit this flag when that lands; it should likely be
subsumed rather than kept as a separate toggle.

**Process note, worth keeping in mind for the rest of Phase 1:** the
original Q6 diagnosis (below) wasn't wrong that a regression existed, but
it misattributed the *cause* without ever running a controlled, same-
session A/B test -- it reasoned from plausible mechanism (CUDA context
init cost) rather than measuring the actual isolated effect. This entry
exists because re-verifying an old finding before building on it (as
asked) caught that. Worth treating other not-yet-re-verified claims in
this log with the same scrutiny before leaning on them.

---

## 2026-09-22 -- set_seed(): CUDA/cuDNN determinism made opt-in (real but smaller effect than Q6 claimed)

**Phase 1, batch 1.** `set_seed()` used to unconditionally call
`torch.cuda.manual_seed_all()` and force
`cudnn.deterministic=True`/`benchmark=False`, applied identically at
every one of its ~19 call sites regardless of whether that particular
solve path uses CUDA. Q6 (this project's earlier investigation) attributed
Beltrami's full slowdown to this; re-verified this session with a
controlled A/B test and found the real effect is much smaller (~20s of
~540s) -- see the entry above for the actual dominant cause and the
corrected numbers.

The fix stands on its own merits regardless of the corrected magnitude:
confirmed via direct grep that no architecture in this codebase has
Dropout, BatchNorm, or Conv layers (only `nn.Linear` + activation) --
meaning there is no GPU-side random operation and no convolution for
cuDNN to benchmark, so CUDA-level determinism has zero observable effect
on any current result, for a real (if now-modest) cost. Made opt-in via a
new `deterministic_cuda: bool = False` parameter rather than reclassifying
all 19 call sites individually -- since nothing currently needs it, no
call site needed updating, only the shared function's default. `torch.manual_seed`
(CPU-side, covers this codebase's weight init since parameters are
constructed before any `.to(device)` call) is unaffected and still always
set. Pass `deterministic_cuda=True` explicitly if a future addition
introduces GPU-side randomness or convolutions that need it.

---

## 2026-09-22 -- BL's LiL-N gradient was silently wrong (root cause of the plateau above)

**This is a real correctness bug, confirmed and fixed, not a hyperparameter
or config drift like everything else in this log.**

Root cause of the previous entry's open finding: `_make_lil_n_loss_fn`
computed the PDE residual's flux-divergence term as

```python
f_p = physics.flux_derivative(S)   # detached from the autograd graph
f_x = f_p * S_x
```

`BLPhysics.flux_derivative` deliberately detaches its output -- correct
and required for `_make_lil_q_system_fn` (LiL-Q), which is quasilinear and
is *supposed* to freeze this Jacobian coefficient at the current iterate.
It is not correct for `_make_lil_n_loss_fn`: `solve_lil_n` calls
`total.backward()` directly on this loss to get the true nonlinear
gradient w.r.t. `beta`, and the detached path silently drops the
contribution of `f'(S)`'s own dependence on `beta` through `S = A_u @
beta`. The computed "gradient" was missing a term -- not numerically
imprecise, structurally incomplete. L-BFGS, which relies entirely on
accurate gradients to build its quasi-Newton approximation, then
converges to a stationary point of the wrong effective objective, or
simply plateaus.

Confirmed by comparison: the pre-GitHub `BuckleyLeverettPhysics` had two
parallel implementations for exactly this reason -- `flux_derivative`
(detached, for the quasilinear solver) and
`flux_derivative_differentiable` (graph-preserving, for gradient-based
solvers) -- and `solve_nonlinear_lil` (pre-GitHub's LiL-N) correctly used
the `_differentiable` one. The June 2026 consolidation's `BLPhysics` only
kept the detached version, and `_make_lil_n_loss_fn` ended up wired to it.
This also explains why only LiL-N was affected: `_make_lil_q_system_fn`
correctly wants the detached coefficient (unaffected), and NiL-N's
`_compute_pde_residual_nn` computes its derivative a structurally
different way (autograd directly on `f(S(x,t))` w.r.t. `x,t` with
`create_graph=True`, never calling this helper at all -- also unaffected).

**Fix:** added `BLPhysics.flux_derivative_differentiable(S)` -- computed
via `torch.autograd.grad(f.sum(), S, create_graph=True)` on the
*undetached* `S`, rather than a hand-derived closed-form analytic
expression (more robust: it can't drift out of sync with `flux()` the way
two independently-maintained formulas could) -- and switched
`_make_lil_n_loss_fn` to use it. Every other call site of the detached
`flux_derivative` (LiL-Q's system assembly, LiL-Q's convergence-check-only
nonlinear loss, NiL-Q's linearization coefficients) was checked directly
and confirmed to be a genuinely quasilinear/frozen-coefficient context
where the detached version remains correct -- left unchanged.

**Verification, in order of rigor:**
1. `tests/test_bl_lil_n_gradient.py` uses `torch.autograd.gradcheck`
   (numerical finite-difference gradient checking) on the actual loss
   function, both gravity and no-gravity. Passes with the fix.
2. Confirmed the test is meaningful, not a tautology: temporarily reverted
   the one-line fix and reran the same test -- it fails, with the
   analytical and numerical gradients disagreeing by up to ~100 in
   magnitude on individual components (not floating-point noise). Restored
   the fix immediately after confirming.
3. Reran BL viscous and gravity at N=8/16 end-to-end (see
   `Phase0_Empirical_Verification.md` for the before/after numbers) to
   confirm the practical effect on real training, not just the unit gradient.

This was found by taking the "digging into why" request seriously rather
than assuming a config/hyperparameter explanation (everything else in
this log so far) -- worth remembering that not every regression in this
codebase is a caps/tolerances drift; this one was a genuine math bug
hiding behind code that runs without error and produces plausible-looking
(if wrong) numbers.

---

## 2026-09-22 -- Empirically: has the line-search cap ever actually ended training?

Checked directly against every stored `*_summary.json` in both
`reference_results/` (GitHub) and `pre-v2-local-codebase/*/`'s own results,
for every problem, by comparing `total_iterations` against each run's
`max_iterations` and `total_line_searches` against its line-search cap:

- **Bratu, both codebases: no.** Every non-converged run has
  `total_iterations == max_iterations` exactly (the iteration cap bound),
  with `total_line_searches` comfortably below its cap every time (e.g.
  GitHub N=15: 19,818-21,282 evals against a 25,000 cap).
- **Burgers, GitHub (no pre-v2 stored results found to check): no.** Same
  pattern -- every non-converged run hits its iteration cap with evals well
  under the line-search cap (e.g. N=25: ~20,800 evals against a 225,000 cap).
- **Buckley-Leverett LiL-N, GitHub: yes, every time.** Checked
  `reference_results/bl_experiments_fourier/` and
  `bl_gravity_experiments_fourier/` at every N (8/16/24/32): LiL-N's
  `total_iterations` is always well below `max_iterations`, while
  `total_line_searches` sits at exactly `3 * max_iterations` (e.g. N=16:
  iterations=3938, line_searches=30001, against max_iterations=10000) --
  the signature of a cap that bound and cut the run off. NiL-N/NiL-Q at the
  same sizes converge comfortably within the same budget; only LiL-N needs
  enough steps to hit it.

**This traces to something the previous BL line-search entry below didn't
catch: `experiments/run_bl.py` was passing its own explicit
`max_line_searches=MAX_LBFGS_ITERS.get(N, 10000) * 3` at the call site,
which bypasses `BLOptConfig`'s `__post_init__`-derived default entirely**
(the derived value only applies when the field is left `None`; an explicit
value always wins). The `BLOptConfig` fix made in the previous entry was
real and correct, but had **no effect on actual experiment runs** through
`run_bl.py`, since that script never relied on the dataclass default in
the first place. Confirmed pre-GitHub `run_bl_experiments.py` /
`run_bl_gravity_experiments.py` never set anything like this (only
`n_epochs_lbfgs`, a pure iteration count) -- this override is entirely new
to the GitHub consolidation.

**Fix:** removed the explicit override from `run_bl.py`; it now falls
through to `BLOptConfig`'s derived worst case (`max_iterations * 15`),
which is both correct and, per the instruction below, the more generous
choice anyway.

---

## 2026-09-22 -- Bratu's MAX_LINE_SEARCHES: reverted to the higher (pre-GitHub) values

Every entry differs from pre-GitHub, not just N=10's iteration cap:
pre-GitHub `{5:24000, 10:30000, 15:30000}` vs. the GitHub values that had
been in place, `{5:15000, 10:25000, 15:25000}`. Per the empirical check
above, this cap has never actually bound for Bratu in either codebase, so
there's no correctness question here -- reverted to the pre-GitHub
(higher, more generous) numbers on instruction, documented for the record
rather than because evidence favored one value over the other.

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

**Addendum, same day:** this fix alone turned out to be insufficient --
see the "Empirically: has the line-search cap ever actually ended
training?" entry above (newer, listed first) for the follow-up fix this
one needed in `experiments/run_bl.py` itself.

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

---

## 2026-09-22 -- Bratu's NiL-Q inner-iteration cap: second, separate N=10 revert (found via a real run)

Caught while sanity-checking a fresh v3-dev run against expectations: with
the N=10 `MAX_ITERATIONS` fix already in place, NiL-N correctly stopped at
7,500 iterations, but **NiL-Q ran to 10,000** -- the fix hadn't fully taken.

Root cause: NiL-Q's actual iteration budget is `MAX_QUASI_ITERS` (flat, 25
in both codebases) times `MAX_LBFGS_PER_QUASI_ITER[N]` (a wholly separate
per-size dict from `MAX_ITERATIONS`, which only governs NiL-N/LiL-N's flat
loop). This second dict also changed at N=10 during the consolidation:
pre-GitHub `{5:300, 10:300, 15:400}` vs. the GitHub value that had been in
place, `{5:300, 10:400, 15:400}` -- worst-case NiL-Q totals of 7,500 vs.
10,000 at N=10, exactly matching what the fresh run showed. Confirmed
Burgers' and both Buckley-Leverett variants' equivalent per-quasi-iter
schedules are genuinely unchanged between codebases (checked directly),
so this second-knob issue is isolated to Bratu N=10, same as the first.

Reverted `MAX_LBFGS_PER_QUASI_ITER[10]`: 400 -> 300. This is the kind of
thing the empirical run-and-compare pass this entry belongs to exists to
catch -- two independently-named constants that both nominally describe
"Bratu's N=10 iteration cap" but govern different methods, easy to fix one
and miss the other from source-reading alone.

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

## 2026-09-22 -- Residual-band figures (Section 3.5), new experiments/residual_band_figures.py

**Phase 1, sub-batch 7 of the iterations.csv work.** The last piece of
Component B that consumes the instrumentation built in sub-batches 3-6
rather than adding more of it: Section 3.5 asks for four figures
comparing $\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$ and
$\|\mathbf{R}^{(k+1)}\|_h$ against $k$, one panel per $P$, for Bratu,
Burgers, and both Buckley-Leverett variants, with the y-axis correctly
labeled $\|\cdot\|_h$ ("the present figures say MSE, which is wrong" --
moot here since the repository had no such figures at all, correct or
not, to begin with).

New `experiments/residual_band_figures.py`. Design choices:

- **Every P/target-loss/quasi-iter-budget value is imported directly**
  from that problem's own `experiments/run_*.py` (`DEFAULT_N_VALUES`,
  `TARGET_LOSSES`, `MAX_QUASI_ITERS`, etc.) rather than duplicated as
  separate constants -- guarantees this script uses the exact same
  "paper settings" (Section 3.3) already established elsewhere in this
  codebase, with no risk of silently drifting out of sync with them.
  `pretrain_epochs` is deliberately *not* imported from any of them:
  confirmed by re-reading each problem's `run_lil_q` that the shared
  `pretrain_lil` step (used identically by all three) is a least-squares
  fit, not an epoch-based optimizer, so that config value has no effect
  on the LiL-Q path this script exercises.
- $\chi_k$ goes in **a second row of panels**, not a secondary axis --
  the spec explicitly offers both; a secondary axis was rejected because
  $\chi_k$'s own dynamic range (seen directly in the real reruns below:
  span of 15+ orders of magnitude within a single BL-gravity P) needs
  its own log-scale axis, and forcing it to share space with two
  already-log-scale residual curves would be unreadable.
  `test_plot_residual_bands_produces_pdf_and_png` covers this indirectly
  via a real run; the panel layout itself was verified by eye against
  the actual PDFs (see below) before committing, not merely inferred
  from the code.
- Each (problem, $P$) run's full `iterations.csv` is saved alongside the
  figure -- literally "the CSV behind each panel" the spec asks for,
  reusing `IterationLogger.to_csv` rather than inventing a
  plot-data-only format; a reader gets the complete Section 3.1 row set
  for that run, not just the four columns the figure itself plots.
- `--quick` flag (first 2 $P$ per problem) for fast iteration/testing
  without committing to the full paper-scale sweep; `main()`'s default
  (no flag) is the real full sweep matching Section 3.3 exactly.

**Verified against real output, not just "the code runs":** ran the
full (non-quick) sweep for real -- all 4 figures + 15 CSVs
(3+5+4+4 = 16 total (problem, $P$) reruns) generated in well under a
minute total (LiL-Q's direct-solve-per-iteration cost stays cheap even
at $P=1024$; nothing here resembles Beltrami's cost profile). Visually
inspected all four PDFs: axis labels, log scaling, and the two-curve
comparison all match the spec; Bratu/Burgers converge in as few as 2-4
iterations at their tight paper tolerances (consistent with LiL-Q's
already-documented fast convergence elsewhere in this project, not a
bug); Buckley-Leverett (both variants) shows richer 6-10-iteration
trajectories with a visible transient bump in $\|\mathbf{R}_{\mathrm{lin}}\|_h$
before it settles, which is real algorithm behavior on the more
nonlinear problem, not a plotting artifact.

**Not done here, deliberately out of scope**: the generated
`results/residual_band_figures/` output itself is not committed
(`results/` is gitignored, matching this repository's existing
convention of tracking scripts, not their generated artifacts, with
`reference_results/` the sole documented exception). Re-running this
script is how a reader reproduces the figures, not a stored copy in git.

140/140 tests passing.

---

## 2026-09-22 -- Beltrami and Darcy wired to Section 3.1 instrumentation; run.json added

**Phase 1, sub-batch 6 of the iterations.csv work** -- the last two
problems, plus the "once per run" `run.json` metadata file
(Section 3.1's own list, separate from `iterations.csv`'s per-iteration
columns) that Sections 3.3 and 6 require alongside it but which no
problem had until now.

**Beltrami**: same self-contained-loop situation as Kovasznay (sub-batch
5) -- `solve_beltrami` never calls `solve_lil_q`, so this manually drives
a `LilQDiagnosticsTracker` again. New `_make_beltrami_nonlinear_loss_fn`/
`_make_beltrami_residual_vector_fn`, generalizing the one-scalar-per-block
convention to Beltrami's 26 row-blocks (momentum x3, continuity, BC-u/v/w
x6 faces, IC-u/v/w, pressure pin) -- same "independently re-derived, not
shared code with the assembly loop" discipline as every prior residual-
vector function, so check B2 is a real cross-check (verified `rel_err`
down to exact `0.0` on one real run). Interior-row unweighting is
conditional on `lambda_mom == lambda_cont`, same reasoning and same
guard-with-regression-test discipline as Kovasznay. `iteration_logger` is
CPU-only (`config.use_gpu=False`) -- raises `NotImplementedError`
otherwise, since the GPU path uses `torch.linalg.lstsq(driver='gelsd')`,
not `gelsy`, and a GPU instrumentation path is Section 3.2's concern,
explicitly Kovasznay-only in the spec. At the manuscript's own Beltrami
scale (P_total=7,984, confirmed by direct calculation:
`3*6**4 + 8**4 == 7984`), the tracker automatically uses pivoted-QR at
the final iterate only (never a full per-iteration SVD) purely because
P_total exceeds `DEFAULT_SVD_CONDITIONING_THRESHOLD` -- this is Section
3.1 item 8's Beltrami-specific requirement, already satisfied by the
generic tracker design from sub-batch 1 with no Beltrami-specific code
needed; locked in by `test_large_P_uses_pivoted_qr_at_final_iterate_only`.
Also made the CPU `_lstsq` branch's `cond=EPS_MACH` explicit (was
`cond=None`, same LAPACK `gelsy` RCOND=-1 convention as every earlier
instance of this change) to capture `rank_gelsy`, which the original code
discarded -- verified bit-identical against the pre-change implementation
loaded from git HEAD.

**Darcy**: a genuinely different case from every other problem in this
codebase -- `solve_lilq_darcy`'s system (Darcy-x, Darcy-y, continuity) is
**linear** in h_tilde*/u*/v*, so there is no Bellman-Kalaba
quasilinearization loop at all, just one direct `lstsq` solve. This
produces exactly one logged row (`k=1`) rather than driving an iteration
loop. Consequences, each locked in by
`test_single_row_reflects_linear_system_structure`: `order_obs` is NaN
and `stall_flag` is False (no k-1 history, same degenerate case every
other problem's first row already exercises); `chi` evaluates to exactly
`0.0`, not NaN -- because the system is linear, the nonlinear residual
and the linearized residual are literally the same operator
(`A @ beta - b`), so `compute_residual_vector_fn` needed no separate
formula, and a chi of 0 is the mathematically correct statement that
there is no linearization error for a linear problem; `norm_R_interior`
equals `norm_R_h` exactly, because Darcy's boundary conditions are
satisfied exactly by construction (the lifting function and augmented
bases bake them into the basis functions themselves -- see the module
docstring), so there is no separate BC row block and *every* row is a
PDE/interior row (`n_interior_rows = A.shape[0]`, `interior_weight=1.0`).
Darcy also has no `lambda_pde`/`lambda_bc`-style row-weighting scheme at
all (the only row scaling is the `sqrt(K*)` physics normalization,
applied per-row rather than as a single block scalar) -- `run.json`'s
`row_weights` field says this explicitly rather than forcing it into the
other problems' single-scalar-per-block shape. `iteration_logger` is only
supported for `config.solver_method == 'qr'` (the `gelsy` driver) --
raises `NotImplementedError` for `'lstsq'`, which calls
`numpy.linalg.lstsq` (a different driver; the schema's `num_rank_gelsy`
column specifically means the scipy `gelsy` rank). Also made the `qr`
branch's `cond=EPS_MACH` explicit to capture rank, same pattern, verified
bit-identical.

**`lilq/run_metadata.py` (new module)**: `run.json` schema/writer for
Section 3.1's "once per run" metadata list (N/P composition, row
weights, collocation construction, basis description, initial
coefficients, solver driver/rcond, the stopping rule actually used and
K_max, stopping reason, first stall iteration, device/thread count).
Unlike `iterations.csv` (one fixed column schema every problem fills in
identically), every problem's row/field composition genuinely differs
(Darcy's 3 unweighted blocks vs. Beltrami's 26 lambda-weighted ones), so
`N_composition`/`P_composition`/`basis_description`/`row_weights` are
free-form dicts rather than fixed columns -- only the top-level field
*names* are fixed (`build_run_metadata` rejects an unrecognized one, same
"typo should fail loudly" discipline as `IterationLogger.record`).
`first_stall_iteration(rows)` scans an `IterationLogger.rows` list for
the first `stall_flag=True` (or `None` if the run never stalled),
reusable by any problem. Wired into `solve_beltrami`/`solve_lilq_darcy`
via a new optional `run_json_path` parameter, requiring `iteration_logger`
be given alongside it (raises `ValueError` otherwise) since
`first_stall_iteration` needs the logged rows. `thread_count` reads
`OMP_NUM_THREADS` via the existing `lilq.provenance.capture_blas_thread_env`,
falling back to `os.cpu_count()` (not `1`) when unset -- an unset BLAS
thread env var means "use every core available", and reporting `1` would
have been actively wrong, not just imprecise; caught and fixed before
committing, not left as a plausible-looking bug.

**Not done here, out of scope, flagged rather than silently assumed**:
Bratu/Burgers/BL/Kovasznay (sub-batches 3-5) do **not** yet have
`run.json` wired in -- this sub-batch was explicitly scoped to
"Beltrami/Darcy + run.json" by the user, and retrofitting the other four
problems is a separate decision, not assumed as included here. Whether
`experiments/run_*.py` should be changed to always produce `run.json`
alongside `iterations.csv` for real experiment runs is also undecided,
same as the equivalent `iterations.csv` question flagged in sub-batch 3.

123/123 -> 134/134 tests passing.

---

## 2026-09-22 -- Kovasznay wired to Section 3.1 instrumentation (own solver, manually driven)

**Phase 1, sub-batch 5 of the iterations.csv work** -- the Kovasznay
follow-up flagged as out-of-scope for sub-batch 4. Unlike Bratu/Burgers/
BL, `problems/kovasznay.py`'s `solve_kovasznay` never calls
`lilq.solvers.solve_lil_q`; it is a self-contained three-field
(u, v, p) quasilinearization loop with its own hand-rolled `history`
dict and its own `rel_delta`-based convergence check. Wiring it required
manually driving a `LilQDiagnosticsTracker` inside that loop rather than
just adding parameters to a thin `run_lil_q` wrapper.

New pieces:

- `_make_kovasznay_nonlinear_loss_fn` / `_make_kovasznay_residual_vector_fn`
  (both new -- Kovasznay had no pre-existing loss/residual helpers to
  extend): the scalar-MSE and weighted-vector forms of the same total
  residual, generalized to Kovasznay's 12 row-blocks (x-momentum,
  y-momentum, continuity, BC-u/BC-v for 4 edges, pressure pin) with the
  same one-scalar-per-block weighting (`w_mom=sqrt(lambda_mom/n_pde)`,
  `w_cont=sqrt(lambda_cont/n_pde)`, `w_bc=sqrt(lambda_bc/n_edge)` per
  edge, `w_pin=sqrt(lambda_bc)` for the single pin row -- confirmed by
  direct derivation that `norm(vector)**2 == total` exactly). Each
  function independently re-derives the raw nonlinear residuals rather
  than sharing code with the assembly loop or with each other, so check
  B2 is a real cross-check.
- `solve_kovasznay` gained the same optional `iteration_logger` parameter
  (default `None`, `history` and all existing behavior unchanged when
  omitted -- verified bit-identical against the pre-change implementation,
  loaded from git HEAD and run side-by-side, not just asserted).
- Split `t_assemble_s`/`t_solve_s` timing via `time.perf_counter()`
  around the existing `A_sys`/`b_sys` build and `lstsq` call respectively
  (additive -- the pre-existing `time.time()`-based combined `dt` and
  `history['solve_time']` are untouched). Made the `lstsq` call's `cond`
  explicit (`cond=EPS_MACH`, same as the `solve_lil_q` change earlier in
  this log) to capture `rank_gelsy`, which the original code discarded
  entirely (`...lstsq(...)[0]`) -- verified bit-identical before/after,
  same LAPACK `gelsy` RCOND=-1 convention as before.

**Interior-row unweighting is conditional, not universal.** Kovasznay's
leading (PDE) rows are actually two sub-blocks weighted independently --
momentum (`w_mom`, 2*n_pde rows) and continuity (`w_cont`, n_pde rows) --
which only collapse to the tracker's required single leading scalar
weight when `lambda_mom == lambda_cont` (true for `KovasznayConfig`'s
defaults, and for every config touched so far, but not guaranteed in
general). `solve_kovasznay` checks this explicitly and only passes
`n_interior_rows`/`interior_weight` to the tracker when it holds; falls
back to the tracker's existing documented NaN behavior otherwise --
locked in by `test_interior_norms_nan_when_lambda_mom_and_cont_differ`
in the new `tests/test_kovasznay_instrumentation.py`, not left as an
unverified assumption.

Check B2 verified against a real solve (`rel_err ~1e-16`, both via the
automated test and manually against the actual returned coefficients).
Manually inspected a real-scale `iterations.csv` (default config:
Re=40, chebyshev N_x=N_y=15, P_total=675) -- 7 iterations to
convergence, `stall_flag` correctly latching `True` once at the
round-off floor, `num_rank_svd == num_rank_gelsy == P_total` throughout,
`chi` decreasing then noisy at the floor -- consistent with the pattern
already seen on Bratu.

Not done here, out of scope: `eps_u`/`eps_v`/`eps_p`/`eps_p_meanfree`
(per-iteration test error against the known Kovasznay exact solution)
are left unpopulated, same as Bratu/Burgers/BL -- `solve_kovasznay`
already computes final-iterate `rel_l2_u/v/p` once at the end, but
wiring per-iteration error columns (and deciding what `eps_p_meanfree`
should mean given the pressure pin, rather than a true mean-free
projection) is a separate piece of work, not assumed as part of this
batch.

108/108 -> 112/112 tests passing.

---

## 2026-09-22 -- Burgers and Buckley-Leverett (viscous + gravity) wired to Section 3.1 instrumentation

**Phase 1, sub-batch 4 of the iterations.csv work.** Rolled out the exact
pattern sub-batch 3 proved on Bratu to the two other problems that share
`lilq.solvers.solve_lil_q`:

- `problems/burgers.py`: `run_lil_q` gained the same optional
  `iteration_logger` parameter, plus a new `_make_lil_residual_vector_fn`
  (vector form of `_make_lil_nonlinear_loss_fn`, stacked
  PDE-then-IC-then-BC-left-then-BC-right, matching
  `_make_lil_q_system_fn`'s exact weighting). `n_interior_rows=n_pde`,
  `interior_weight=sqrt(lambda_pde/n_pde)` -- confirmed by direct
  derivation that the weighted vector's squared norm equals `total_loss`
  even though the BC block is itself split across two sub-blocks
  (`bc_left`/`bc_right`) with independently-computed weights
  (`w_bl=sqrt(lb/n_bc_l)`, `w_br=sqrt(lb/n_bc_r)`) rather than one shared
  scalar -- summing their squared contributions still reduces to
  `lb*(mean_left + mean_right)`, matching `compute_loss`'s `bc` term
  exactly.
- `problems/buckley_leverett.py`: same pattern, covering **both**
  configurations (`BLConfig()` viscous and `BLConfig.with_gravity()`)
  since they share one `run_lil_q` -- gravity vs. viscous is fully
  internal to `physics.flux`/`flux_derivative`'s dispatch on `config.N_g`,
  invisible to the instrumentation wiring. The residual-vector function
  uses the same detached `physics.flux_derivative` the existing
  `_make_lil_nonlinear_loss_fn` already used (a forward-only evaluation
  for logging, not a gradient path) -- not the graph-preserving
  `flux_derivative_differentiable` needed only for LiL-N's `.backward()`
  (see the LiL-N gradient fix entry further down this log). BL's BC
  targets are non-zero (`physics.bc_left`/`bc_right`, unlike Bratu's/
  Burgers' homogeneous Dirichlet BCs), so the residual vector subtracts
  them explicitly, matching `_make_lil_q_system_fn`'s `b_stacked`.

Both are backward-compatible (bit-identical coefficients/summary with
`iteration_logger=None`, the default) and both pass check B2
(`test_check_b2_*` in the new `tests/test_burgers_instrumentation.py` /
`tests/test_bl_instrumentation.py`, same independent-reconstruction-via-
public-API method as Bratu's check) at `rel_err < 1e-10`. 108/108 tests
passing.

**Kovasznay deliberately excluded from this batch, not silently
dropped.** Unlike Bratu/Burgers/BL, `problems/kovasznay.py`'s
`solve_kovasznay` is a wholly self-contained quasilinearization loop --
it does not call `lilq.solvers.solve_lil_q` at all, has no
`_make_lil_q_system_fn`/`_make_lil_nonlinear_loss_fn`-style helpers to
reuse, and already computes its own condition number and test errors
(`rel_l2_u/v/p`) inline every iteration via a hand-rolled `history` dict
rather than `IterationLogger`. Wiring it to the shared Section 3.1 schema
means manually replicating the tracker-calling pattern inside that loop
(three coupled fields stacked into one system, `eps_p_meanfree` needing a
real decision for a pressure field pinned at one point rather than the
scalar problems' single-field error columns) -- structurally different,
larger work than the three pattern-repeats above. Flagging this now
rather than assuming it belongs in "sub-batch 4" as originally scoped;
proposed as its own follow-up sub-batch.

---

## 2026-09-22 -- Bratu wired end-to-end to Section 3.1 instrumentation; check B2 passes

**Phase 1, sub-batch 3 of the iterations.csv work ("proof of concept":
wire one real problem all the way through and verify before rolling out
further).** Two parts:

1. `LilQDiagnosticsTracker`/`solve_lil_q` extended with `n_interior_rows`/
   `interior_weight` constructor/call parameters to actually compute
   `norm_R_interior`/`norm_Rlin_interior` (sub-batch 2 left these
   permanently `NaN`, documented as deferred). Both were `NaN`-by-default
   and additive, so this needed no new test of the "does this break
   anything" kind beyond the two new interior-row-specific tests added to
   `tests/test_lil_q_diagnostics_tracker.py`.
2. `problems/bratu.py`'s `run_lil_q` gained an optional `iteration_logger`
   parameter (default `None`, existing behavior unchanged when omitted --
   confirmed bit-identical coefficients/summary with and without it in
   `tests/test_bratu_instrumentation.py`). When given, it builds a new
   `_make_lil_residual_vector_fn` (the vector form of
   `_make_lil_nonlinear_loss_fn` -- same weighted interior-then-boundary
   stacking as `_make_lil_q_system_fn`'s `A_stacked`/`b_stacked`, verified
   by construction: `norm(vector)**2 == total_loss`) and passes
   `n_interior_rows=n_pde`, `interior_weight=sqrt(lambda_pde/n_pde)`
   through to `solve_lil_q`, matching the one-scalar-per-block convention
   confirmed directly in Bratu's own `assemble_system_fn` (documented in
   `lilq/iteration_log.py`'s `LilQDiagnosticsTracker` docstring).

**Check B2** (Computational_Package_1_v2.md Section 3.1: the logged
$\|\mathbf{R}^{(k)}\|_h$ must match the nonlinear operator evaluated
directly at the collocation points, to $10^{-10}$ relative) implemented
as an automated test
(`test_check_b2_residual_identity_against_direct_evaluation`), run
against a real (not synthetic) Bratu solve: reconstructs
`A_u`/`A_uxx`/`A_uyy`/`A_bc` from scratch via the public
`create_basis_2d`/`generate_collocation_points_2d`/`basis.evaluate`/
`basis.derivative` API (same config/seed as `run_lil_q`'s own internal
setup -> deterministic, same collocation points -- confirmed
`generate_collocation_points_2d` reseeds `np.random` internally
regardless of prior RNG state consumed during pretraining), independent
of any of `problems.bratu`'s private helper functions. Passed at
`rel_err < 1e-10` (this codebase's own float64 near-machine-precision
scale, not a loosened tolerance).

Manually inspected two real `iterations.csv` outputs (not committed --
scratch verification, not fixtures) to sanity-check the full 30-column
row at scale: a P=100 Bratu solve converging in 1 iteration under the
default `R_tol=1e-4` (matches the P=100 `final_loss` value already
recorded in `tests/test_solve_lil_q_instrumentation.py`'s bit-identical
check), and the same problem forced to run 25 iterations at a much
tighter `R_tol=1e-12` past its actual round-off floor -- `stall_flag`
correctly latches `True` once `chi`/`norm_Rlin_h` stop moving, and
`order_obs` oscillates (sign flips, occasional large magnitudes) once in
that floor regime, which is the expected behavior of a quadratic-rate
estimator applied to noise rather than a bug.

Not yet done, deliberately out of scope for this batch (per the
established "one step at a time" discipline): rolling this same pattern
out to Burgers/BL-viscous/BL-gravity/Kovasznay (sub-batch 4), and
deciding whether `experiments/run_bratu.py` itself should be changed to
always produce `iterations.csv` for real experiment runs -- that changes
a script's default output and is flagged here rather than assumed, the
same way the multi-seed harness's non-wiring was flagged rather than
silently deferred.

---

## 2026-09-22 -- solve_lil_q's lstsq call: cond=None made explicit as cond=EPS_MACH

**Phase 1, sub-batch 2 of the iterations.csv work.** While wiring the
new `rcond` log column, noticed `solve_lil_q`'s `scipy.linalg.lstsq`
call never passed `cond` explicitly (relying on scipy's `cond=None`
default). Per LAPACK's own `gelsy` convention, `cond=None` maps to
`RCOND=-1`, which means "use machine precision" -- i.e. already
numerically identical to explicitly passing `cond=EPS_MACH`. Verified
directly before changing anything: ran the same `lstsq` call both ways
on a random matrix and confirmed bit-identical output
(`np.array_equal(x1, x2) == True`, max diff `0.0`). Made explicit so the
logged `rcond` value matches what the code actually does rather than
relying on an unstated LAPACK convention -- a documentation/clarity
change, confirmed not a numeric one.

---

## 2026-09-22 -- BLAS thread counts now actually set, not just recorded

**Phase 1, batch 2 follow-up.** The provenance-capture entry below
noted this as a "not yet acted on" gap -- correctly challenged as having
no real reason to wait for a later phase, so closed out the same day.

`lilq/blas_threads.py`: sets `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`MKL_NUM_THREADS` (all three, since which one actually governs depends on
which BLAS backend numpy/scipy link against -- setting the irrelevant
ones is harmless) as its own import-time side effect -- the one place in
this codebase where that pattern is the *correct* choice rather than the
antipattern fixed elsewhere in this log: there is no non-environment-
variable way to configure a native BLAS library's thread pool before it
loads, and BLAS reads these at load time, not dynamically. Default value:
`SLURM_CPUS_PER_TASK` when running under sbatch/salloc (correctly matches
the actual allocation rather than the whole node's core count on shared
HPRC nodes), else every logical core on the machine. Never overrides a
value the environment already set (`os.environ.setdefault`, not
assignment) -- an explicit shell/job-script choice always wins.

Because of the load-time requirement, `import lilq.blas_threads` has to
be the first import in every experiment script's entry point, before
`import numpy`/`import scipy` and before any `from lilq...`/
`from problems...` that would pull numpy in transitively. Wired into all
8 real experiment scripts at exactly that position (one script,
`run_burgers_basis_comparison.py`, had numpy imported *before* its own
`sys.path` setup -- reordered so blas_threads and numpy both come after).

Verified end-to-end, tying both provenance-capture pieces together: ran
`run_kovasznay.py` with all three variables explicitly unset in the
calling shell, and confirmed the resulting `hardware.json` shows them all
set to this machine's core count (24) -- the two mechanisms working
together exactly as intended, not just independently unit-tested.

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

**Note:** the spec says these thread-count variables must be *set*, not
just recorded -- this module only ever reported what was already in
effect. See the "BLAS thread counts now actually set" entry above
(newer, listed first) for `lilq/blas_threads.py`, which closes that gap.

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

**Flag raised here, resolved same day:** while checking this,
`MAX_LINE_SEARCHES` turned out to *also* differ for Bratu, on every entry
-- something Q1's original writeup claimed was "unchanged" (true for
Burgers, checked and confirmed identical in both versions, but evidently
not checked carefully enough for Bratu at the time). See the "Bratu's
MAX_LINE_SEARCHES" entry above (newer, listed first) for the resolution
-- reverted to the pre-GitHub values on instruction.

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

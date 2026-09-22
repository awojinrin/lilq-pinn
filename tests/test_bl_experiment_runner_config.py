"""Regression guard: run_bl.py must not reintroduce an explicit
max_line_searches override that bypasses BLOptConfig's derived (inert)
default.

Context (DECISIONS.md, 2026-09-22 "line-search cap actually binds"
entry): experiments/run_bl.py used to pass
``max_line_searches=MAX_LBFGS_ITERS.get(N, 10000) * 3`` explicitly, which
-- unlike BLOptConfig's derived default -- genuinely did truncate LiL-N
before convergence, confirmed against every N in
reference_results/{bl,bl_gravity}_experiments_fourier (total_iterations
well below max_iterations while total_line_searches sat at exactly
3 * max_iterations). Removed so the config's own derived worst case
(max_iterations * 15) takes over.

This is checked at the source level rather than by executing
run_experiment_for_N, which does real file I/O (make_experiment_dir) and
real training as a side effect of being called -- not something a unit
test should trigger just to inspect a constructed config object.
"""

import inspect

import experiments.run_bl as run_bl_module


def test_run_bl_opt_construction_has_no_explicit_max_line_searches():
    source = inspect.getsource(run_bl_module.run_experiment_for_N)
    opt_construction = source[source.index("opt = BLOptConfig("):source.index("R_tol=")]
    code_lines = [
        line for line in opt_construction.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("max_line_searches=" in line for line in code_lines)

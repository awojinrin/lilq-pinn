"""
Four-method Tables (Computational_Package_1_v2.md Section 3.4)
======================================================================

Reruns NiL-N, NiL-Q, and LiL-N -- the three L-BFGS-trained methods, as
opposed to LiL-Q, which already has its own full Section 3.1 log (see
``experiments/residual_band_figures.py`` and each problem's
``run_json_path`` wiring) -- across Bratu, Burgers, and both
Buckley-Leverett variants, at every P from the paper, seeds 0/1/2 for
NiL-N/NiL-Q (LiL-N: one deterministic run, per the spec's own exemption
-- same reasoning as ``lilq.multiseed``), on GPU as in the paper plus an
additional CPU run for the largest P per benchmark.

Every P/target-loss/iteration-and-line-search-budget value is imported
directly from that problem's own ``experiments/run_*.py`` (not
duplicated), same principle as ``residual_band_figures.py`` -- this
script cannot silently drift from the "paper settings"/"repository code
and budgets" (Section 3.3/3.4) already established there.

Output: ``four_method_tables.csv``, one row per (benchmark, P, method,
seed, device) -- see ``lilq.four_method_log`` for the exact schema.

This is a substantially larger compute job than
``residual_band_figures.py``: NiL-N/NiL-Q/LiL-N are real L-BFGS training
loops with iteration caps up to 10,000-20,000, run at up to 3 seeds each,
across ~15 (benchmark, P) combinations, on top of an extra CPU pass for
the largest size per benchmark. Use ``--quick`` for a fast correctness
smoke test (1 size, 1 seed, tiny caps); the full run (no flags) is a
real, possibly long-running job -- run it deliberately, not by default.

Usage::

    python experiments/four_method_tables.py --quick
    python experiments/four_method_tables.py
"""

import sys
import os
import argparse
import dataclasses
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np
import torch

from lilq.utils import DEVICE, clear_gpu_memory
from lilq.four_method_log import FourMethodLogger, classify_stopping_reason, subsample_loss_history

from problems.bratu import (
    BratuConfig, BratuOptConfig,
    run_nil_n as bratu_nil_n, run_nil_q as bratu_nil_q, run_lil_n as bratu_lil_n,
)
from problems.burgers import (
    BurgersConfig, BurgersOptConfig,
    run_nil_n as burgers_nil_n, run_nil_q as burgers_nil_q, run_lil_n as burgers_lil_n,
)
from problems.buckley_leverett import (
    BLConfig, BLOptConfig,
    run_nil_n as bl_nil_n, run_nil_q as bl_nil_q, run_lil_n as bl_lil_n,
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'four_method_tables'
DEFAULT_SEEDS = (0, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-problem run configuration -- sizes/targets/budgets pulled straight
# from the existing experiment scripts (Section 3.3/3.4's "paper
# settings"/"repository code and budgets").
# ─────────────────────────────────────────────────────────────────────────────

def _bratu_runs(quick=False):
    from experiments.run_bratu import (
        DEFAULT_N_VALUES, TARGET_LOSSES, MAX_ITERATIONS, MAX_LINE_SEARCHES,
        MAX_QUASI_ITERS, MAX_LBFGS_PER_QUASI_ITER,
        DEFAULT_LAMBDA, DEFAULT_BASIS, DEFAULT_K_RATIO,
    )
    N_values = DEFAULT_N_VALUES[:1] if quick else DEFAULT_N_VALUES
    runs = []
    for N in N_values:
        config = BratuConfig(lambda_=DEFAULT_LAMBDA, N_x=N, N_y=N,
                              k_ratio=DEFAULT_K_RATIO, basis_type=DEFAULT_BASIS)
        opt = BratuOptConfig(
            max_iterations=MAX_ITERATIONS.get(N, 10000),
            max_line_searches=MAX_LINE_SEARCHES.get(N, 30000),
            R_tol=TARGET_LOSSES.get(N, 1e-4),
            max_quasi_iters_nn=MAX_QUASI_ITERS,
            max_inner_iters_nn=MAX_LBFGS_PER_QUASI_ITER.get(N, 300),
        )
        runs.append((N * N, config, opt))
    return runs


def _burgers_runs(quick=False):
    from experiments.run_burgers import (
        DEFAULT_N_VALUES, TARGET_LOSSES, MAX_LBFGS_ITERS, MAX_LINE_SEARCHES,
        MAX_QUASI_ITERS, MAX_LBFGS_PER_QUASI, DEFAULT_BASIS, VISCOSITY, T_FINAL, K_RATIO,
    )
    N_values = DEFAULT_N_VALUES[:1] if quick else DEFAULT_N_VALUES
    runs = []
    for N in N_values:
        config = BurgersConfig(N_x=N, N_t=N, viscosity=VISCOSITY, T_final=T_FINAL,
                                basis_type=DEFAULT_BASIS, k_ratio=K_RATIO)
        opt = BurgersOptConfig(
            max_iterations=MAX_LBFGS_ITERS.get(N, 10000),
            max_line_searches=MAX_LINE_SEARCHES.get(N, 100000),
            R_tol=TARGET_LOSSES.get(N, 1e-4),
            max_quasi_iters_nn=MAX_QUASI_ITERS,
            max_inner_iters_nn=MAX_LBFGS_PER_QUASI.get(N, 300),
        )
        runs.append((N * N, config, opt))
    return runs


def _bl_runs(gravity, quick=False):
    from experiments.run_bl import (
        DEFAULT_N_VALUES, TARGET_LOSSES, MAX_LBFGS_ITERS, MAX_LBFGS_PER_QUASI,
        GRAVITY_TARGET_LOSSES, GRAVITY_MAX_QUASI_ITERS, GRAVITY_MAX_LBFGS_PER_QUASI,
        MAX_QUASI_ITERS, DEFAULT_BASIS, K_RATIO,
    )
    N_values = DEFAULT_N_VALUES[:1] if quick else DEFAULT_N_VALUES
    targets = GRAVITY_TARGET_LOSSES if gravity else TARGET_LOSSES
    quasi_iters = GRAVITY_MAX_QUASI_ITERS if gravity else MAX_QUASI_ITERS
    inner_per_quasi = GRAVITY_MAX_LBFGS_PER_QUASI if gravity else MAX_LBFGS_PER_QUASI
    runs = []
    for N in N_values:
        base = BLConfig.with_gravity() if gravity else BLConfig()
        config = dataclasses.replace(base, N_x=N, N_t=N,
                                      basis_type=DEFAULT_BASIS, k_ratio=K_RATIO)
        # max_line_searches intentionally left unset -- BLOptConfig derives
        # it from max_iterations (see tests/test_bl_experiment_runner_config.py
        # and DECISIONS.md: an explicit override here previously truncated
        # LiL-N before convergence at every N).
        opt = BLOptConfig(
            max_iterations=MAX_LBFGS_ITERS.get(N, 10000),
            R_tol=targets.get(N, 1e-3),
            max_quasi_iters_nn=quasi_iters,
            max_inner_iters_nn=inner_per_quasi.get(N, 200),
        )
        runs.append((N * N, config, opt))
    return runs


# label, run-list builder, NiL-N fn, NiL-Q fn, LiL-N fn
PROBLEMS = [
    ('bratu', _bratu_runs, bratu_nil_n, bratu_nil_q, bratu_lil_n),
    ('burgers', _burgers_runs, burgers_nil_n, burgers_nil_q, burgers_lil_n),
    ('bl', lambda quick: _bl_runs(False, quick), bl_nil_n, bl_nil_q, bl_lil_n),
    ('bl_gravity', lambda quick: _bl_runs(True, quick), bl_nil_n, bl_nil_q, bl_lil_n),
]


def _apply_quick_budgets(opt, max_iterations=15, max_line_searches=200,
                         max_quasi_iters_nn=2, max_inner_iters_nn=10):
    """Shrink an opt config's budgets for --quick smoke testing, without
    touching R_tol (keeping the real target means the run genuinely
    exercises "did not converge, hit a cap" -- the interesting path --
    rather than trivially converging on a target loosened along with
    everything else)."""
    return dataclasses.replace(
        opt,
        max_iterations=max_iterations, max_line_searches=max_line_searches,
        max_quasi_iters_nn=max_quasi_iters_nn, max_inner_iters_nn=max_inner_iters_nn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run + log
# ─────────────────────────────────────────────────────────────────────────────

def _run_and_log(logger, benchmark, P, config, opt, method_name, runner,
                 seeds, devices, verbose=True):
    """One row per (seed, device). ``seeds=None`` means "run once at
    config.seed, log seed as None" -- LiL-N's deterministic case.
    """
    actual_seeds = list(seeds) if seeds else [None]

    for device in devices:
        for seed in actual_seeds:
            run_config = dataclasses.replace(config, seed=seed) if seed is not None else config

            t0 = time.perf_counter()
            result = runner(run_config, opt, device=device, verbose=False)
            elapsed = time.perf_counter() - t0
            metrics, summary = result[-2], result[-1]

            if method_name == 'NiL-Q':
                iterations_used = summary['n_quasi_iters']
                iterations_cap = opt.max_quasi_iters_nn
            else:
                iterations_used = summary['total_iterations']
                iterations_cap = opt.max_iterations

            stopping_reason = classify_stopping_reason(
                converged=summary['converged'],
                iterations_used=iterations_used, iterations_cap=iterations_cap,
                line_searches_used=summary['total_line_searches'],
                line_searches_cap=opt.max_line_searches,
            )

            logger.record(
                benchmark=benchmark, P=P, method=method_name, seed=seed, device=str(device),
                total_iterations=summary['total_iterations'],
                total_line_searches=summary['total_line_searches'],
                training_time_s=summary['training_time'],
                final_loss=summary['final_loss'],
                converged=summary['converged'], stopping_reason=stopping_reason,
                iterations_cap=iterations_cap, line_searches_cap=opt.max_line_searches,
                loss_history_every_10=subsample_loss_history(metrics.to_dict()),
            )

            if verbose:
                status = "CONVERGED" if summary['converged'] else stopping_reason
                seed_label = f"seed={seed}" if seed is not None else "seed=n/a"
                print(f"    {method_name:6s} {seed_label:10s} device={device!s:6s} "
                      f"iters={summary['total_iterations']:6d} "
                      f"loss={summary['final_loss']:.3e} [{status}] ({elapsed:.1f}s)")

            clear_gpu_memory()


def run_problem(benchmark, runs_fn, nil_n_fn, nil_q_fn, lil_n_fn, logger,
                quick=False, seeds=DEFAULT_SEEDS, verbose=True):
    runs = runs_fn(quick)
    for i, (P, config, opt) in enumerate(runs):
        if quick:
            opt = _apply_quick_budgets(opt)
        is_largest = (i == len(runs) - 1)
        cpu = torch.device('cpu')
        devices = [DEVICE]
        if is_largest and str(DEVICE) != 'cpu':
            devices = [DEVICE, cpu]

        if verbose:
            print(f"\n  P={P} (devices={[str(d) for d in devices]})")

        run_seeds = seeds[:1] if quick else seeds
        _run_and_log(logger, benchmark, P, config, opt, 'NiL-N', nil_n_fn,
                    seeds=run_seeds, devices=devices, verbose=verbose)
        _run_and_log(logger, benchmark, P, config, opt, 'NiL-Q', nil_q_fn,
                    seeds=run_seeds, devices=devices, verbose=verbose)
        _run_and_log(logger, benchmark, P, config, opt, 'LiL-N', lil_n_fn,
                    seeds=None, devices=devices, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="Section 3.4 four-method tables")
    parser.add_argument('--quick', action='store_true',
                        help='Smoke-test budgets (1 size, 1 seed, tiny iteration/line-search '
                             'caps) instead of the full paper reruns -- this is a real, '
                             'possibly long-running job without this flag.')
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--problems', type=str, nargs='+', default=None,
                        choices=['bratu', 'burgers', 'bl', 'bl_gravity'],
                        help='Subset of problems to run (default: all four).')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = PROBLEMS
    if args.problems:
        problems = [p for p in PROBLEMS if p[0] in args.problems]

    logger = FourMethodLogger()
    t_start = time.time()

    for benchmark, runs_fn, nil_n_fn, nil_q_fn, lil_n_fn in problems:
        print(f"\n{'=' * 60}\n{benchmark}\n{'=' * 60}")
        run_problem(benchmark, runs_fn, nil_n_fn, nil_q_fn, lil_n_fn, logger,
                   quick=args.quick)

    out_path = out_dir / 'four_method_tables.csv'
    logger.to_csv(out_path)
    print(f"\nWrote {len(logger)} rows to {out_path}")
    print(f"Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()

"""
Buckley-Leverett — Experiment Runner (with optional gravity)
=============================================================

Runs all 4 methods with saving, plotting, checkpoints.

Usage::

    python experiments/run_bl.py
    python experiments/run_bl.py --gravity --N 8 16 24 32
    python experiments/run_bl.py --lil-q-only
"""

import sys, os, time, argparse
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from lilq.utils import set_seed, clear_gpu_memory, DEVICE
from problems.buckley_leverett import (
    BLConfig, BLOptConfig,
    run_nil_n, run_nil_q, run_lil_n, run_lil_q,
    evaluate_lil_solution, evaluate_nn_solution,
)

from experiments.exp_utils import (
    make_experiment_dir, make_figures_dir, experiment_base_dir,
    save_lil_checkpoint, save_nn_checkpoint, save_metrics_csv,
    save_master_results, save_summary_json, save_run_provenance,
    plot_convergence_by_method, plot_convergence_by_size,
    plot_solution_field, print_summary_table,
    run_stochastic_with_seeds,
)

# ── Config matching Current Codebase/BL/run_bl_experiments.py ──

DEFAULT_N_VALUES = [8, 16, 24, 32]
DEFAULT_BASIS = 'fourier'
K_RATIO = 10
PRETRAIN_EPOCHS = 1000
MAX_QUASI_ITERS = 50

TARGET_LOSSES = {8: 8.5e-2, 16: 1.5e-2, 24: 7.5e-4, 32: 7.5e-5}
MAX_LBFGS_ITERS = {8: 5000, 16: 10000, 24: 15000, 32: 20000}
MAX_LBFGS_PER_QUASI = {8: 100, 16: 200, 24: 300, 32: 400}

GRAVITY_TARGET_LOSSES = {8: 2.5e-1, 16: 1.5e-1, 24: 7.5e-2, 32: 3.5e-2}
GRAVITY_MAX_QUASI_ITERS = 20
GRAVITY_MAX_LBFGS_PER_QUASI = {8: 30, 16: 200, 24: 400, 32: 500}
# The gravity IC (a steepness-100 smooth step, essentially monotonic in x)
# is poorly represented by DEFAULT_BASIS's mode_x='both' split (half the
# x-modes wasted on sine components that don't help a monotonic profile).
# 'cos_fourier' (full cosine resolution in x) fits the same IC to ~2-2.4x
# lower relative error at every paper N, and empirically took LiL-N from
# not converging at all at N=24/32 (hits the full iteration cap) to
# converging in under 9s and 23s respectively. Gravity-specific --
# DEFAULT_BASIS is unchanged for the non-gravity case, which doesn't show
# this problem. See DECISIONS.md.
GRAVITY_BASIS = 'cos_fourier'

ALL_METHODS = ['NiL-N', 'NiL-Q', 'LiL-N', 'LiL-Q']


def run_experiment_for_N(N, basis_type, gravity, methods, verbose=True, seeds=None):
    """``seeds``, when given, runs NiL-N/NiL-Q across every seed in it
    (Section 2/3.4: seeds 0, 1, 2) via ``run_stochastic_with_seeds``
    instead of once at ``config.seed``. LiL-N/LiL-Q always run once
    regardless. Omitted (``None``, the default), behavior is unchanged.
    """
    if gravity:
        config = BLConfig.with_gravity()
        config.N_x = N; config.N_t = N
        config.basis_type = basis_type; config.k_ratio = K_RATIO
        label = 'bl_gravity'
        targets = GRAVITY_TARGET_LOSSES
        quasi_iters = GRAVITY_MAX_QUASI_ITERS
        inner_per_quasi = GRAVITY_MAX_LBFGS_PER_QUASI
    else:
        config = BLConfig(N_x=N, N_t=N, basis_type=basis_type, k_ratio=K_RATIO)
        label = 'bl'
        targets = TARGET_LOSSES
        quasi_iters = MAX_QUASI_ITERS
        inner_per_quasi = MAX_LBFGS_PER_QUASI

    opt = BLOptConfig(
        max_iterations=MAX_LBFGS_ITERS.get(N, 10000),
        # max_line_searches intentionally left unset here -- BLOptConfig's
        # __post_init__ derives it from max_iterations (the true worst
        # case). This script used to pass an explicit `*3` multiplier,
        # which the pre-GitHub codebase never had and which was found to
        # actively truncate LiL-N before convergence at every N, both
        # viscous and gravity (see DECISIONS.md).
        R_tol=targets.get(N, 1e-3),
        max_quasi_iters_nn=quasi_iters,
        max_inner_iters_nn=inner_per_quasi.get(N, 200),
        max_quasi_iters_lil=quasi_iters,
        pretrain_epochs=PRETRAIN_EPOCHS,
    )

    n_dir = make_experiment_dir(label, basis_type, N)
    fig_dir = make_figures_dir(label, basis_type)
    results = {'N': N, 'basis_type': basis_type, 'gravity': gravity}

    runners = {'NiL-N': run_nil_n, 'NiL-Q': run_nil_q,
               'LiL-N': run_lil_n, 'LiL-Q': run_lil_q}

    for method in methods:
        if verbose:
            print(f"\n{'='*60}\nN={N} | {method} | "
                  f"{'gravity' if gravity else 'no gravity'}\n{'='*60}")
        try:
            tag = method.lower().replace('-', '_')
            is_lil = method.startswith('LiL')
            t0 = time.time()

            if is_lil:
                if method == 'LiL-Q':
                    basis, coeffs, metrics, summary = runners[method](
                        config, opt, verbose=verbose)
                else:
                    basis, coeffs, metrics, summary = runners[method](
                        config, opt, device=DEVICE, verbose=verbose)
                save_lil_checkpoint(n_dir / f'{tag}_checkpoint.npz',
                                    basis, coeffs, {'N': N, 'gravity': gravity})
                X, T, U = evaluate_lil_solution(basis, coeffs, config, n_eval=200)
                plot_solution_field(
                    T, X, U,
                    f'BL {method} (N={N}{", gravity" if gravity else ""})',
                    fig_dir, f'{label}_{tag}_N{N}_solution',
                    xlabel='t', ylabel='x', clabel='S(x,t)')
            else:
                model, metrics, summary = run_stochastic_with_seeds(
                    runners[method], config, opt, seeds, DEVICE, verbose)
                save_nn_checkpoint(n_dir / f'{tag}_checkpoint.pt',
                                   model, {'N': N})
                X, T, U = evaluate_nn_solution(model, config, n_eval=200)
                plot_solution_field(
                    T, X, U, f'BL {method} (N={N})',
                    fig_dir, f'{label}_{tag}_N{N}_solution',
                    xlabel='t', ylabel='x', clabel='S(x,t)')

            save_metrics_csv(n_dir / f'{tag}_metrics.csv', metrics)
            summary['metrics_history'] = (metrics.to_dict()
                                          if hasattr(metrics, 'to_dict') else {})
            save_summary_json(n_dir / f'{tag}_summary.json', summary)
            results[method] = summary

            if verbose:
                status = "CONVERGED" if summary.get('converged') else "completed"
                print(f"  {method}: loss={summary['final_loss']:.4e}, "
                      f"time={time.time()-t0:.2f}s, {status}")
        except Exception as e:
            results[method] = {'error': str(e), 'final_loss': float('nan')}
            if verbose:
                print(f"  {method} FAILED: {e}")
                import traceback; traceback.print_exc()
        clear_gpu_memory()

    return results


def main():
    parser = argparse.ArgumentParser(description="Buckley-Leverett Experiments")
    parser.add_argument('--N', type=int, nargs='+', default=DEFAULT_N_VALUES)
    parser.add_argument('--basis', type=str, default=None,
                        help=f"Basis type. Default: '{DEFAULT_BASIS}' for viscous, "
                             f"'{GRAVITY_BASIS}' for --gravity (see DECISIONS.md) -- "
                             "pass explicitly to override either.")
    parser.add_argument('--gravity', action='store_true')
    parser.add_argument('--lil-q-only', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds for NiL-N/NiL-Q (Section 3.4: 0 1 2 for '
                             'the paper reruns). LiL-N/LiL-Q always run once '
                             'regardless. Omitted: single run, unchanged '
                             'default behavior.')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    methods = ['LiL-Q'] if args.lil_q_only else ALL_METHODS
    verbose = not args.quiet
    plabel = 'BL-gravity' if args.gravity else 'BL'
    basis = args.basis if args.basis is not None else (GRAVITY_BASIS if args.gravity else DEFAULT_BASIS)

    print(f"{plabel} Experiments | Device: {DEVICE} | basis: {basis}")
    print(f"N values: {args.N}, Methods: {methods}")
    if args.seeds:
        print(f"NiL-N/NiL-Q seeds: {args.seeds}")

    all_results = {}
    t0 = time.time()
    for N in args.N:
        set_seed(42)
        all_results[N] = run_experiment_for_N(
            N, basis, args.gravity, methods, verbose, seeds=args.seeds)

    print_summary_table(all_results, args.N, methods, plabel)
    print(f"\nTotal: {time.time()-t0:.1f}s")

    tag = 'bl_gravity' if args.gravity else 'bl'
    base_dir = experiment_base_dir(tag, basis)
    tgt = GRAVITY_TARGET_LOSSES if args.gravity else TARGET_LOSSES
    save_master_results(base_dir / f'{tag}_master_results.json', all_results,
                        {'n_values': args.N, 'gravity': args.gravity, 'basis': basis,
                         'target_losses': tgt,
                         'seeds': list(args.seeds) if args.seeds else None})
    save_run_provenance(tag, basis)

    fig_dir = make_figures_dir(tag, basis)
    try:
        plot_convergence_by_method(all_results, args.N, plabel, fig_dir)
        plot_convergence_by_size(all_results, args.N, plabel, fig_dir)
    except Exception as e:
        print(f"  Plot warning: {e}")


if __name__ == '__main__':
    main()

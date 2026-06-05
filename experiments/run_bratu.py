"""
Bratu Equation — Experiment Runner
=====================================

Runs all 4 methods (NiL-N, NiL-Q, LiL-N, LiL-Q) for specified N values,
saves checkpoints, metrics CSVs, master results JSON, convergence plots,
and solution field visualisations.

Usage::

    python experiments/run_bratu.py
    python experiments/run_bratu.py --basis fourier --N 5 10 15
    python experiments/run_bratu.py --lil-q-only
    python experiments/run_bratu.py --plot-only
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np
import torch

from lilq.utils import set_seed, clear_gpu_memory, DEVICE
from lilq.analysis import make_svd_callback
from problems.bratu import (
    BratuConfig, BratuOptConfig,
    run_nil_n, run_nil_q, run_lil_n, run_lil_q,
    evaluate_lil_solution, evaluate_lil_residual,
    evaluate_nn_solution,
)

from experiments.exp_utils import (
    make_experiment_dir, make_figures_dir, experiment_base_dir,
    save_lil_checkpoint, save_nn_checkpoint, save_metrics_csv,
    save_master_results, save_summary_json,
    plot_convergence_by_method, plot_convergence_by_size,
    plot_solution_field, print_summary_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_VALUES = [5, 10, 15]
DEFAULT_LAMBDA = 6.2
DEFAULT_BASIS = 'fourier'
DEFAULT_K_RATIO = 10
DEFAULT_PRETRAIN_EPOCHS = 1000
DEFAULT_PRETRAIN_TOL = 1e-5
N_HIDDEN_LAYERS = 2

TARGET_LOSSES = {5: 2.5e-1, 10: 1e-4, 15: 2.5e-7}
MAX_ITERATIONS = {5: 5000, 10: 10000, 15: 10000}
MAX_LINE_SEARCHES = {5: 15000, 10: 25000, 15: 25000}
MAX_QUASI_ITERS = 25
MAX_LBFGS_PER_QUASI_ITER = {5: 300, 10: 400, 15: 400}

ALL_METHODS = ['NiL-N', 'NiL-Q', 'LiL-N', 'LiL-Q']


# ─────────────────────────────────────────────────────────────────────────────
# Single-N Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_for_N(N, basis_type, lambda_, methods, verbose=True):
    """Run specified methods for a single N value. Returns results dict."""
    config = BratuConfig(
        lambda_=lambda_,
        N_x=N, N_y=N,
        k_ratio=DEFAULT_K_RATIO,
        basis_type=basis_type,
    )

    R_tol = TARGET_LOSSES.get(N, 1e-4)
    opt = BratuOptConfig(
        max_iterations=MAX_ITERATIONS.get(N, 10000),
        max_line_searches=MAX_LINE_SEARCHES.get(N, 30000),
        R_tol=R_tol,
        max_quasi_iters_nn=MAX_QUASI_ITERS,
        max_inner_iters_nn=MAX_LBFGS_PER_QUASI_ITER.get(N, 300),
        max_quasi_iters_lil=MAX_QUASI_ITERS,
        pretrain_epochs=DEFAULT_PRETRAIN_EPOCHS,
    )

    # Output directory for this N
    n_dir = make_experiment_dir('bratu', basis_type, N)
    fig_dir = make_figures_dir('bratu', basis_type)

    results = {'N': N, 'basis_type': basis_type, 'lambda': lambda_}

    method_runners = {
        'NiL-N': run_nil_n,
        'NiL-Q': run_nil_q,
        'LiL-N': run_lil_n,
        'LiL-Q': run_lil_q,
    }

    for method_name in methods:
        runner = method_runners[method_name]
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"N={N} | {method_name} | basis={basis_type}")
            print('=' * 60)

        try:
            t0 = time.time()
            is_lil = method_name.startswith('LiL')
            tag = method_name.lower().replace('-', '_')

            if is_lil:
                if method_name == 'LiL-Q':
                    svd_store = []
                    callback = make_svd_callback(verbose=False, store=svd_store)
                    basis, coeffs, metrics, summary = runner(
                        config, opt, verbose=verbose,
                        diagnostics_callback=callback,
                    )
                else:
                    # LiL-N uses device, not diagnostics_callback
                    basis, coeffs, metrics, summary = runner(
                        config, opt, device=DEVICE, verbose=verbose,
                    )

                # Save checkpoint
                save_lil_checkpoint(
                    n_dir / f'{tag}_checkpoint.npz', basis, coeffs,
                    {'N': N, 'basis_type': basis_type, 'lambda': lambda_})

                # Plot solution field
                X, Y, U = evaluate_lil_solution(basis, coeffs, config, n_eval=200)
                plot_solution_field(
                    X, Y, U,
                    title=f'Bratu {method_name} (N={N})',
                    fig_dir=fig_dir,
                    filename=f'bratu_{tag}_N{N}_solution',
                )

                # Plot residual
                X_r, Y_r, R = evaluate_lil_residual(basis, coeffs, config, n_eval=200)
                plot_solution_field(
                    X_r, Y_r, R,
                    title=f'Bratu {method_name} Residual (N={N})',
                    fig_dir=fig_dir,
                    filename=f'bratu_{tag}_N{N}_residual',
                    cmap='RdBu_r', clabel='residual',
                )
            else:
                model, metrics, summary = runner(
                    config, opt, device=DEVICE, verbose=verbose,
                )

                # Save NN checkpoint
                save_nn_checkpoint(
                    n_dir / f'{tag}_checkpoint.pt', model,
                    {'N': N, 'basis_type': basis_type, 'lambda': lambda_})

                # Plot NN solution field
                X, Y, U = evaluate_nn_solution(model, config, n_eval=200)
                plot_solution_field(
                    X, Y, U,
                    title=f'Bratu {method_name} (N={N})',
                    fig_dir=fig_dir,
                    filename=f'bratu_{tag}_N{N}_solution',
                )

            # Save metrics CSV
            save_metrics_csv(n_dir / f'{tag}_metrics.csv', metrics)

            # Save summary JSON
            summary['metrics_history'] = (metrics.to_dict()
                                          if hasattr(metrics, 'to_dict') else {})
            save_summary_json(n_dir / f'{tag}_summary.json', summary)

            results[method_name] = summary
            elapsed = time.time() - t0

            if verbose:
                status = "CONVERGED" if summary.get('converged', False) else "completed"
                print(f"  {method_name}: loss={summary['final_loss']:.4e}, "
                      f"time={elapsed:.2f}s, {status}")

        except Exception as e:
            results[method_name] = {'error': str(e), 'final_loss': float('nan')}
            if verbose:
                print(f"  {method_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

        clear_gpu_memory()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full Experiment Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(N_values, basis_type, lambda_, methods, verbose=True):
    """Run the full experiment matrix and generate plots."""
    all_results = {}
    total_t0 = time.time()

    for N in N_values:
        set_seed(42)
        all_results[N] = run_experiment_for_N(
            N, basis_type, lambda_, methods, verbose)

    total_elapsed = time.time() - total_t0

    # Summary table
    print_summary_table(all_results, N_values, methods, 'Bratu')
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # Save master results JSON
    base_dir = experiment_base_dir('bratu', basis_type)
    config_info = {
        'n_values': N_values, 'basis_type': basis_type, 'lambda': lambda_,
        'target_losses': TARGET_LOSSES, 'k_ratio': DEFAULT_K_RATIO,
        'max_iterations': MAX_ITERATIONS,
        'max_quasi_iters': MAX_QUASI_ITERS,
        'pretrain_epochs': DEFAULT_PRETRAIN_EPOCHS,
    }
    save_master_results(base_dir / 'bratu_master_results.json',
                        all_results, config_info)

    # Generate convergence plots
    fig_dir = make_figures_dir('bratu', basis_type)
    print("\nGenerating convergence plots...")
    try:
        plot_convergence_by_method(all_results, N_values, 'Bratu', fig_dir)
        plot_convergence_by_size(all_results, N_values, 'Bratu', fig_dir)
    except Exception as e:
        print(f"  Warning: Convergence plot failed: {e}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bratu Equation Experiments")
    parser.add_argument('--N', type=int, nargs='+', default=DEFAULT_N_VALUES,
                        help='Basis sizes to test')
    parser.add_argument('--basis', type=str, default=DEFAULT_BASIS,
                        help='LiL basis type')
    parser.add_argument('--lambda', type=float, default=DEFAULT_LAMBDA,
                        dest='lambda_', help='Bratu parameter')
    parser.add_argument('--lil-q-only', action='store_true',
                        help='Run only LiL-Q method')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    methods = ['LiL-Q'] if args.lil_q_only else ALL_METHODS
    verbose = not args.quiet

    print(f"Bratu Experiments | Device: {DEVICE}")
    print(f"N values: {args.N}, basis: {args.basis}, lambda: {args.lambda_}")
    print(f"Methods: {methods}")
    print(f"Results -> results/bratu_experiments_{args.basis}/")

    run_all_experiments(args.N, args.basis, args.lambda_, methods, verbose)


if __name__ == '__main__':
    main()

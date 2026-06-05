"""
Burgers Equation — Experiment Runner
=======================================

Runs all 4 methods for specified N values with saving, plotting, checkpoints.

Usage::

    python experiments/run_burgers.py
    python experiments/run_burgers.py --N 5 10 15 20 25
    python experiments/run_burgers.py --lil-q-only
"""

import sys, os, time, argparse
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np

from lilq.utils import set_seed, clear_gpu_memory, DEVICE
from problems.burgers import (
    BurgersConfig, BurgersOptConfig,
    run_nil_n, run_nil_q, run_lil_n, run_lil_q,
    evaluate_lil_solution, evaluate_nn_solution,
)

from experiments.exp_utils import (
    make_experiment_dir, make_figures_dir, experiment_base_dir,
    save_lil_checkpoint, save_nn_checkpoint, save_metrics_csv,
    save_master_results, save_summary_json,
    plot_convergence_by_method, plot_convergence_by_size,
    plot_solution_field, print_summary_table,
)

# ── Config matching Current Codebase/Burgers/run_burgers_experiments.py ──

DEFAULT_N_VALUES = [5, 10, 15, 20, 25]
DEFAULT_BASIS = 'sin_fourier'  # Fourier sin(x) x Fourier both(t), matching original
VISCOSITY = 0.1
T_FINAL = 1.0
K_RATIO = 10
PRETRAIN_EPOCHS = 500
MAX_QUASI_ITERS = 20

TARGET_LOSSES = {5: 6e-2, 10: 1e-3, 15: 2e-5, 20: 1e-7, 25: 5e-9}
MAX_LBFGS_ITERS = {5: 1000, 10: 5000, 15: 7500, 20: 10000, 25: 15000}
MAX_LINE_SEARCHES = {5: 30000, 10: 75000, 15: 112500, 20: 150000, 25: 225000}
MAX_LBFGS_PER_QUASI = {5: 100, 10: 250, 15: 375, 20: 500, 25: 750}

ALL_METHODS = ['NiL-N', 'NiL-Q', 'LiL-N', 'LiL-Q']


def run_experiment_for_N(N, basis_type, methods, verbose=True):
    config = BurgersConfig(
        N_x=N, N_t=N, viscosity=VISCOSITY, T_final=T_FINAL,
        basis_type=basis_type, k_ratio=K_RATIO,
    )
    opt = BurgersOptConfig(
        max_iterations=MAX_LBFGS_ITERS.get(N, 10000),
        max_line_searches=MAX_LINE_SEARCHES.get(N, 100000),
        R_tol=TARGET_LOSSES.get(N, 1e-4),
        max_quasi_iters_nn=MAX_QUASI_ITERS,
        max_inner_iters_nn=MAX_LBFGS_PER_QUASI.get(N, 300),
        max_quasi_iters_lil=MAX_QUASI_ITERS,
        pretrain_epochs=PRETRAIN_EPOCHS,
    )

    n_dir = make_experiment_dir('burgers', basis_type, N)
    fig_dir = make_figures_dir('burgers', basis_type)
    results = {'N': N, 'basis_type': basis_type}

    runners = {'NiL-N': run_nil_n, 'NiL-Q': run_nil_q,
               'LiL-N': run_lil_n, 'LiL-Q': run_lil_q}

    for method in methods:
        if verbose:
            print(f"\n{'='*60}\nN={N} | {method}\n{'='*60}")
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
                                    basis, coeffs, {'N': N})
                X, T, U = evaluate_lil_solution(basis, coeffs, config, n_eval=200)
                plot_solution_field(T, X, U, f'Burgers {method} (N={N})',
                                   fig_dir, f'burgers_{tag}_N{N}_solution',
                                   xlabel='t', ylabel='x', clabel='u(x,t)')
            else:
                model, metrics, summary = runners[method](
                    config, opt, device=DEVICE, verbose=verbose)
                save_nn_checkpoint(n_dir / f'{tag}_checkpoint.pt',
                                   model, {'N': N})
                X, T, U = evaluate_nn_solution(model, config, n_eval=200)
                plot_solution_field(T, X, U, f'Burgers {method} (N={N})',
                                   fig_dir, f'burgers_{tag}_N{N}_solution',
                                   xlabel='t', ylabel='x', clabel='u(x,t)')

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
    parser = argparse.ArgumentParser(description="Burgers Experiments")
    parser.add_argument('--N', type=int, nargs='+', default=DEFAULT_N_VALUES)
    parser.add_argument('--basis', type=str, default=DEFAULT_BASIS)
    parser.add_argument('--lil-q-only', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    methods = ['LiL-Q'] if args.lil_q_only else ALL_METHODS
    verbose = not args.quiet

    print(f"Burgers Experiments | Device: {DEVICE}")
    print(f"N values: {args.N}, basis: {args.basis}, Methods: {methods}")

    all_results = {}
    t0 = time.time()
    for N in args.N:
        set_seed(42)
        all_results[N] = run_experiment_for_N(N, args.basis, methods, verbose)

    print_summary_table(all_results, args.N, methods, 'Burgers')
    print(f"\nTotal: {time.time()-t0:.1f}s")

    base_dir = experiment_base_dir('burgers', args.basis)
    save_master_results(base_dir / 'burgers_master_results.json', all_results,
                        {'n_values': args.N, 'basis': args.basis,
                         'target_losses': TARGET_LOSSES})

    fig_dir = make_figures_dir('burgers', args.basis)
    try:
        plot_convergence_by_method(all_results, args.N, 'Burgers', fig_dir)
        plot_convergence_by_size(all_results, args.N, 'Burgers', fig_dir)
    except Exception as e:
        print(f"  Plot warning: {e}")


if __name__ == '__main__':
    main()

"""
Kovasznay Flow — Experiment Runner
=====================================

LiL-Q only. Saves checkpoints, metrics, solution field plots.

Usage::

    python experiments/run_kovasznay.py
    python experiments/run_kovasznay.py --N 5 10 15 20 25 --Re 40
"""

import sys, os, time, argparse, json
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np

from lilq.utils import set_seed
from problems.kovasznay import (
    KovasznayConfig, KovasznayPhysics,
    solve_kovasznay, evaluate_all_fields,
)

from experiments.exp_utils import (
    make_experiment_dir, make_figures_dir, experiment_base_dir,
    save_master_results, save_summary_json, save_metrics_csv,
    plot_solution_panel, print_summary_table,
)

# ── Config matching Current Codebase/Navier-Stokes/Kovasznay ──

DEFAULT_N_VALUES = [5, 10, 15, 20, 25]
DEFAULT_RE = 40.0
DEFAULT_BASIS = 'chebyshev'
K_RATIO = 4
MAX_ITER = 20
TOL = 1e-9


def run_experiment_for_N(N, Re, basis_type, verbose=True):
    config = KovasznayConfig(
        N_x=N, N_y=N, Re=Re, basis_type=basis_type,
        k_ratio=K_RATIO, max_iter=MAX_ITER, tol=TOL,
    )

    n_dir = make_experiment_dir('kovasznay', basis_type, N)
    fig_dir = make_figures_dir('kovasznay', basis_type)

    if verbose:
        print(f"\n{'='*60}\nKOVASZNAY  N={N}  Re={Re}\n{'='*60}")

    t0 = time.time()
    result = solve_kovasznay(config, verbose=verbose)
    elapsed = time.time() - t0

    summary = {
        'N': N, 'method': 'LiL-Q',
        'n_outer_iters': result['n_outer_iters'],
        'solve_time_total': result['solve_time_total'],
        'training_time': result['solve_time_total'],
        'pde_mse': result['pde_mse'],
        'cont_mse': result['cont_mse'],
        'rel_l2_u': result['rel_l2_u'],
        'rel_l2_v': result['rel_l2_v'],
        'rel_l2_p': result['rel_l2_p'],
        'final_loss': result['pde_mse'],
        'converged': True,
    }

    save_summary_json(n_dir / 'lil_q_summary.json', summary)

    # Save coefficients
    np.savez(n_dir / 'lil_q_checkpoint.npz',
             theta_u=result['theta_u'], theta_v=result['theta_v'],
             theta_p=result['theta_p'])

    # Save convergence history as CSV
    if 'history' in result:
        save_metrics_csv(n_dir / 'lil_q_metrics.csv', result['history'])

    # Convergence plot (PDE + continuity residual vs iteration)
    if 'history' in result and result['history'].get('iteration'):
        h = result['history']
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, ylabel in [(axes[0], 'pde_residual', 'Momentum MSE'),
                                 (axes[1], 'continuity_residual', 'Continuity MSE')]:
            if key in h:
                ax.semilogy(h['iteration'], h[key], 'o-', ms=4, color='#E63946')
                ax.set_xlabel('Outer iteration'); ax.set_ylabel(ylabel)
                ax.set_title(f'Kovasznay N={N}'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(fig_dir / f'kovasznay_N{N}_convergence.{ext}',
                        bbox_inches='tight', dpi=150)
        plt.close()
        print(f"    Saved convergence plot")

    # Plot solution fields
    physics = KovasznayPhysics(config)
    pred, exact = evaluate_all_fields(result, physics, n_eval=200)

    plot_solution_panel(
        {'u predicted': (pred['X'], pred['Y'], pred['u']),
         'v predicted': (pred['X'], pred['Y'], pred['v']),
         'p predicted': (pred['X'], pred['Y'], pred['p'])},
        f'Kovasznay LiL-Q (N={N}, Re={Re})',
        fig_dir, f'kovasznay_N{N}_solution',
    )

    # Error fields
    plot_solution_panel(
        {'u error': (pred['X'], pred['Y'], pred['u'] - exact['u']),
         'v error': (pred['X'], pred['Y'], pred['v'] - exact['v']),
         'p error': (pred['X'], pred['Y'], pred['p'] - exact['p'])},
        f'Kovasznay Error (N={N}, Re={Re})',
        fig_dir, f'kovasznay_N{N}_error',
    )

    if verbose:
        print(f"\n  Iters: {summary['n_outer_iters']}, Time: {elapsed:.3f}s")
        print(f"  rel_l2:  u={summary['rel_l2_u']:.3e}  "
              f"v={summary['rel_l2_v']:.3e}  p={summary['rel_l2_p']:.3e}")

    return {'LiL-Q': summary, 'N': N}


def main():
    parser = argparse.ArgumentParser(description="Kovasznay Flow Experiments")
    parser.add_argument('--N', type=int, nargs='+', default=DEFAULT_N_VALUES)
    parser.add_argument('--Re', type=float, default=DEFAULT_RE)
    parser.add_argument('--basis', type=str, default=DEFAULT_BASIS)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    verbose = not args.quiet
    print(f"Kovasznay Experiments | Re={args.Re}, basis={args.basis}")

    all_results = {}
    t0 = time.time()
    for N in args.N:
        set_seed(42)
        all_results[N] = run_experiment_for_N(N, args.Re, args.basis, verbose)

    print_summary_table(all_results, args.N, ['LiL-Q'], 'Kovasznay')
    print(f"\nTotal: {time.time()-t0:.1f}s")

    base_dir = experiment_base_dir('kovasznay', args.basis)
    save_master_results(base_dir / 'kovasznay_master_results.json', all_results,
                        {'n_values': args.N, 'Re': args.Re, 'basis': args.basis})


if __name__ == '__main__':
    main()

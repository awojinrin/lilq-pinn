"""
Linear Elasticity — Experiment Runner
========================================

LiL-Q only (linear PDE, single QR solve per N). Saves checkpoints and plots.

Usage::

    python experiments/run_elasticity.py
    python experiments/run_elasticity.py --N 5 10 15 20 25
"""

import sys, os, time, argparse
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np

from lilq.utils import set_seed
from problems.elasticity import (
    ElasticityConfig, ElasticityPhysics,
    solve_elasticity, evaluate_all_fields,
)

from experiments.exp_utils import (
    make_experiment_dir, make_figures_dir, experiment_base_dir,
    save_master_results, save_summary_json,
    plot_solution_panel, print_summary_table,
)

# ── Config matching Current Codebase/Navier-Stokes/LinearElasticity ──

DEFAULT_N_VALUES = [5, 10, 15, 20, 25]
K_RATIO = 10


def run_experiment_for_N(N, verbose=True):
    config = ElasticityConfig(N_x=N, N_y=N, k_ratio=K_RATIO)

    n_dir = make_experiment_dir('elasticity', 'mixed', N)
    fig_dir = make_figures_dir('elasticity', 'mixed')

    if verbose:
        print(f"\n{'='*60}\nELASTICITY  N={N}\n{'='*60}")

    t0 = time.time()
    result = solve_elasticity(config, verbose=verbose)
    elapsed = time.time() - t0

    summary = {
        'N': N, 'method': 'LiL-Q',
        'n_params': result['n_params'],
        'solve_time_qr': result['solve_time_qr'],
        'solve_time_total': result['solve_time_total'],
        'training_time': result['solve_time_total'],
        'pde_mse': result['pde_mse'],
        'rel_l2_ux': result['rel_l2_ux'],
        'rel_l2_uy': result['rel_l2_uy'],
        'rel_l2_sxx': result.get('rel_l2_sxx', float('nan')),
        'rel_l2_syy': result.get('rel_l2_syy', float('nan')),
        'rel_l2_sxy': result.get('rel_l2_sxy', float('nan')),
        'final_loss': result['pde_mse'],
        'converged': True,
    }

    save_summary_json(n_dir / 'lil_q_summary.json', summary)

    np.savez(n_dir / 'lil_q_checkpoint.npz',
             theta_u=result['theta_u'], theta_v=result['theta_v'])

    # Plot solution fields
    physics = ElasticityPhysics(config)
    pred, exact = evaluate_all_fields(result, physics, n_eval=200)

    plot_solution_panel(
        {'$u_x$ predicted': (pred['X'], pred['Y'], pred['ux']),
         '$u_y$ predicted': (pred['X'], pred['Y'], pred['uy'])},
        f'Elasticity LiL-Q (N={N})',
        fig_dir, f'elasticity_N{N}_displacement',
    )

    plot_solution_panel(
        {'$u_x$ error': (pred['X'], pred['Y'], pred['ux'] - exact['ux']),
         '$u_y$ error': (pred['X'], pred['Y'], pred['uy'] - exact['uy'])},
        f'Elasticity Error (N={N})',
        fig_dir, f'elasticity_N{N}_error',
    )

    # Stress fields
    if 'sxx' in pred:
        plot_solution_panel(
            {'$\\sigma_{xx}$': (pred['X'], pred['Y'], pred['sxx']),
             '$\\sigma_{yy}$': (pred['X'], pred['Y'], pred['syy']),
             '$\\sigma_{xy}$': (pred['X'], pred['Y'], pred['sxy'])},
            f'Elasticity Stress (N={N})',
            fig_dir, f'elasticity_N{N}_stress',
        )

    if verbose:
        print(f"\n  DOFs: {summary['n_params']}, Time: {elapsed:.3f}s")
        print(f"  rel_l2:  ux={summary['rel_l2_ux']:.3e}  "
              f"uy={summary['rel_l2_uy']:.3e}")

    return {'LiL-Q': summary, 'N': N}


def main():
    parser = argparse.ArgumentParser(description="Elasticity Experiments")
    parser.add_argument('--N', type=int, nargs='+', default=DEFAULT_N_VALUES)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    verbose = not args.quiet
    print(f"Elasticity Experiments")

    all_results = {}
    t0 = time.time()
    for N in args.N:
        set_seed(42)
        all_results[N] = run_experiment_for_N(N, verbose)

    print_summary_table(all_results, args.N, ['LiL-Q'], 'Elasticity')
    print(f"\nTotal: {time.time()-t0:.1f}s")

    base_dir = experiment_base_dir('elasticity', 'mixed')
    save_master_results(base_dir / 'elasticity_master_results.json', all_results,
                        {'n_values': args.N, 'k_ratio': K_RATIO})


if __name__ == '__main__':
    main()

"""
Beltrami Flow — Experiment Runner
====================================

LiL-Q only, 3D+time Navier-Stokes (Ethier-Steinman).
Matches original: N_vel=6 per dim for u,v,w; N_p=8 per dim for pressure.

Usage::

    python experiments/run_beltrami.py
    python experiments/run_beltrami.py --N_vel 6 --N_p 8
"""

import sys, os, time, argparse
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np

from lilq.utils import set_seed
from problems.beltrami import (
    BeltramiConfig, BeltramiPhysics, solve_beltrami,
    evaluate_fields_at_slice, evaluate_fields_3d,
)

from experiments.exp_utils import (
    make_experiment_dir, make_figures_dir, experiment_base_dir,
    save_master_results, save_summary_json, save_metrics_csv,
    print_summary_table,
    plot_fields_2d_beltrami, plot_3d_volumes_beltrami,
)

# ── Config matching Current Codebase/Navier-Stokes/Beltrami_3D ──
# Original experiment: (N_vel=6, N_p=8, "N6_Np8")

DEFAULT_BASIS = 'chebyshev'

# Collocation per N_vel (from original COLLOC dict)
COLLOC = {
    4: dict(N_x=6, N_y=6, N_z=6, N_t=6, N_bc=5, N_t_bc=5, N_ic=6),
    5: dict(N_x=7, N_y=7, N_z=7, N_t=7, N_bc=6, N_t_bc=6, N_ic=7),
    6: dict(N_x=8, N_y=8, N_z=8, N_t=8, N_bc=6, N_t_bc=6, N_ic=8),
}


def run_experiment(N_vel, N_p, basis_type, verbose=True):
    cs = COLLOC.get(N_vel, COLLOC[6])

    config = BeltramiConfig(
        N_vel=N_vel, N_p=N_p,
        basis_type=basis_type,
        **cs,
    )

    label = f"N{N_vel}_Np{N_p}"
    n_dir = make_experiment_dir('beltrami', basis_type, label)

    if verbose:
        Pv = N_vel**4; Pp = N_p**4; Pt = 3*Pv + Pp
        print(f"\n{'='*70}")
        print(f"BELTRAMI 3D  {label}: N_vel={N_vel} (P={Pv}), "
              f"N_p={N_p} (P={Pp}), P_total={Pt}")
        print(f"{'='*70}")

    t0 = time.time()
    result = solve_beltrami(config, verbose=verbose)
    elapsed = time.time() - t0

    summary = {
        'N_vel': N_vel, 'N_p': N_p, 'label': label,
        'method': 'LiL-Q',
        'n_outer_iters': result['n_outer_iters'],
        'n_params': result['n_params'],
        'solve_time_total': result['solve_time_total'],
        'training_time': result['solve_time_total'],
        'pde_mse': result['pde_mse'],
        'cont_mse': result['cont_mse'],
        'rel_l2_u': result.get('rel_l2_u', float('nan')),
        'rel_l2_v': result.get('rel_l2_v', float('nan')),
        'rel_l2_w': result.get('rel_l2_w', float('nan')),
        'rel_l2_p': result.get('rel_l2_p', float('nan')),
        'final_loss': result['pde_mse'],
        'converged': True,
    }

    save_summary_json(n_dir / 'lil_q_summary.json', summary)

    np.savez(n_dir / 'lil_q_checkpoint.npz',
             theta_u=result['theta_u'], theta_v=result['theta_v'],
             theta_w=result['theta_w'], theta_p=result['theta_p'])

    # Save convergence history
    if 'history' in result:
        save_metrics_csv(n_dir / 'lil_q_metrics.csv', result['history'])

    if verbose:
        print(f"\n  Iters: {summary['n_outer_iters']}, Time: {elapsed:.3f}s")
        print(f"  rel_l2:  u={summary['rel_l2_u']:.3e}  "
              f"v={summary['rel_l2_v']:.3e}  w={summary['rel_l2_w']:.3e}  "
              f"p={summary['rel_l2_p']:.3e}")

    # ── Plots ──
    fig_dir = make_figures_dir('beltrami', basis_type)

    # Convergence plot
    if 'history' in result and result['history'].get('iteration'):
        h = result['history']
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, ylabel in [(axes[0], 'pde_residual', 'Momentum MSE'),
                                 (axes[1], 'continuity_residual', 'Continuity MSE')]:
            if key in h:
                ax.semilogy(h['iteration'], h[key], 'o-', ms=4, color='#E63946')
                ax.set_xlabel('Outer iteration'); ax.set_ylabel(ylabel)
                ax.set_title(f'Beltrami {label}'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(fig_dir / f'beltrami_{label}_convergence.{ext}',
                        bbox_inches='tight', dpi=150)
        plt.close()
        print(f"    Saved convergence plot")
    physics = BeltramiPhysics(config)

    print("\n  Generating 2D field slice plots...")
    try:
        exact_2d, pred_2d = evaluate_fields_at_slice(
            result, physics, z_val=0.0, t_val=1.0, n_eval=51)
        plot_fields_2d_beltrami(exact_2d, pred_2d, fig_dir, label,
                                t_val=1.0, z_val=0.0)
    except Exception as e:
        print(f"    [WARN] 2D plot failed: {e}")

    print("  Generating 3D cube volume plots...")
    try:
        exact_3d, pred_3d = evaluate_fields_3d(
            result, physics, t_val=1.0, n_eval=21)
        plot_3d_volumes_beltrami(exact_3d, pred_3d, fig_dir, label, t_val=1.0)
    except Exception as e:
        print(f"    [WARN] 3D volume plot failed: {e}")
        import traceback; traceback.print_exc()

    return {'LiL-Q': summary, 'label': label}


def main():
    parser = argparse.ArgumentParser(description="Beltrami 3D Flow Experiments")
    parser.add_argument('--N_vel', type=int, default=6,
                        help="Basis order per dim for velocity (u,v,w)")
    parser.add_argument('--N_p', type=int, default=8,
                        help="Basis order per dim for pressure")
    parser.add_argument('--basis', type=str, default=DEFAULT_BASIS)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    verbose = not args.quiet
    set_seed(42)

    print(f"Beltrami Experiments | N_vel={args.N_vel}, N_p={args.N_p}, "
          f"basis={args.basis}")

    result = run_experiment(args.N_vel, args.N_p, args.basis, verbose)

    base_dir = experiment_base_dir('beltrami', args.basis)
    save_master_results(base_dir / 'beltrami_master_results.json',
                        {result['label']: result},
                        {'N_vel': args.N_vel, 'N_p': args.N_p,
                         'basis': args.basis})


if __name__ == '__main__':
    main()

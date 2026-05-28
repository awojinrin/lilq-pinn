"""
Kovasznay Flow — Experiment Runner and Plotting
=================================================

Runs the LiL-Q solver (quasilinearized Navier-Stokes) on the
Kovasznay flow benchmark for several basis sizes.

This is a NONLINEAR problem: the quadratic convective terms are
linearized via Bellman-Kalaba quasilinearization at each outer
iteration, producing a sequence of convex least-squares problems.

Reusable files (place in same directory):
    - lil_basis.py
    - pub_style.py

Usage:
    python run_kovasznay_experiments.py

Author: Gbenga 
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from kovasznay_core import (
    KovasznayPhysics, KovasznayDiscretization,
    create_kovasznay_basis, solve_lilq_kovasznay,
    evaluate_all_fields, evaluate_exact_fields,
)

import lilq_shared.style as pub_style

FULL_WIDTH = pub_style.FULL_WIDTH


# =============================================================================
# CONFIGURATION
# =============================================================================

RE = 40.0                  # Reynolds number
N_VALUES = [5, 10, 15, 20, 25]

# Basis type for all three fields (u, v, p)
BASIS_TYPE_U = "chebyshev"
BASIS_TYPE_V = "chebyshev"
BASIS_TYPE_P = "chebyshev"

# Weights
LAMBDA_MOM  = 1.0
LAMBDA_CONT = 1.0
LAMBDA_BC   = 10.0

# Quasilinearization
MAX_ITER = 20
TOL = 1e-9

# Collocation
K_RATIO = 4
COLLOCATION_RATIOS = (0.8, 0.1, 0.1)

OUTPUT_DIR = Path(f"kovasznay_Re{int(RE)}_{BASIS_TYPE_U}")


# =============================================================================
# RUNNER
# =============================================================================

def run_all_experiments():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    physics = KovasznayPhysics(Re=RE, x_domain=(-0.5, 1), y_domain=(-0.5, 1.5))
    all_results = {}

    for N in N_VALUES:
        print(f"\n{'='*70}")
        print(f"  N = {N},  P_per_field = {N**2}")
        print(f"{'='*70}")
        # K_RATIO = 1204 / (3 * N**2)
        # COLLOCATION_RATIOS = (1000/1204, 102/1204, 120/1204)

        disc = KovasznayDiscretization(
            N_x=N, N_y=N, k_ratio=K_RATIO,
            collocation_ratios=COLLOCATION_RATIOS,
            seed=42,
        )

        basis_u = create_kovasznay_basis(BASIS_TYPE_U, N, N,
                                          physics.x_domain, physics.y_domain)
        basis_v = create_kovasznay_basis(BASIS_TYPE_V, N, N,
                                          physics.x_domain, physics.y_domain)
        basis_p = create_kovasznay_basis(BASIS_TYPE_P, N, N,
                                          physics.x_domain, physics.y_domain)

        result = solve_lilq_kovasznay(
            physics, disc, basis_u, basis_v, basis_p,
            lambda_mom=LAMBDA_MOM, lambda_cont=LAMBDA_CONT, lambda_bc=LAMBDA_BC,
            max_iter=MAX_ITER, tol=TOL, verbose=True,
        )
        result['N'] = N
        all_results[N] = result

        for k in range(result['history']['iteration'][-1] + 1):
            print(f"  Iter {k:3d}: PDE MSE={result['history']['pde_residual'][k]:.3e}  "
                  f"Cont MSE={result['history']['continuity_residual'][k]:.3e}  "
                  f"Time={result['history']['solve_time'][k]:.3f}s  "
                  f"Cond(A)={result['history']['cond_number'][k]:.2e}")

        np.savez(OUTPUT_DIR / f"checkpoint_N{N}.npz",
                 theta_u=result['theta_u'],
                 theta_v=result['theta_v'],
                 theta_p=result['theta_p'], N=N)

    save_summary_table(all_results, OUTPUT_DIR)
    return all_results, physics


def save_summary_table(all_results, output_dir):
    rows = []
    for N in sorted(all_results.keys()):
        r = all_results[N]
        rows.append({
            'N': N,
            'P_total': r['n_params'],
            'Iters': r['n_outer_iters'],
            'Time_s': f"{r['solve_time_total']:.3f}",
            'PDE_MSE': f"{r['pde_mse']:.3e}",
            'Cont_MSE': f"{r['cont_mse']:.3e}",
            'Rel_L2_u': f"{r['rel_l2_u']:.3e}",
            'Rel_L2_v': f"{r['rel_l2_v']:.3e}",
            'Rel_L2_p': f"{r['rel_l2_p']:.3e}",
        })

    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    hdr = (f"{'N':>3} {'P_tot':>7} {'Iters':>6} {'Time':>8} "
           f"{'PDE MSE':>10} {'Cont MSE':>10} "
           f"{'relL2 u':>10} {'relL2 v':>10} {'relL2 p':>10}")
    print(hdr)
    print("-" * 100)
    for row in rows:
        print(f"{row['N']:>3} {row['P_total']:>7} {row['Iters']:>6} {row['Time_s']:>8} "
              f"{row['PDE_MSE']:>10} {row['Cont_MSE']:>10} "
              f"{row['Rel_L2_u']:>10} {row['Rel_L2_v']:>10} {row['Rel_L2_p']:>10}")

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {output_dir / 'summary.json'}")


# =============================================================================
# PUBLICATION FIGURES
# =============================================================================

def _save_fig(fig, filepath):
    filepath = Path(filepath)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    fig.savefig(filepath.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def _pcolor(ax, X, Y, C, title, cmap='jet', vmin=None, vmax=None):
    im = ax.pcolormesh(X, Y, C, cmap=cmap, shading='gouraud',
                        vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_title(title)
    pub_style.style_axis(ax)
    return im


def plot_fields_and_errors(exact, pred, output_dir, N_plot):
    """3 rows (u, v, p) x 3 cols (exact, LiL-Q, |error|)."""
    fig, axes = plt.subplots(3, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.95),
                             layout='compressed', sharex=True, sharey=True)
    X, Y = exact['X'], exact['Y']

    fields = ['u', 'v', 'p']
    labels_exact = [r'$u$ (Exact)',           r'$v$ (Exact)',           r'$p$ (Exact)']
    labels_pred  = [r'$u$ (LiL-Q)',     r'$v$ (LiL-Q)',     r'$p$ (LiL-Q)']
    labels_err   = [r'$u$ (Error)',     r'$v$ (Error)',     r'$p$ (Error)']

    for i, fld in enumerate(fields):
        # Color limits from actual data range
        fv_min = min(exact[fld].min(), pred[fld].min())
        fv_max = max(exact[fld].max(), pred[fld].max())

        # Col 0: exact
        _pcolor(axes[i, 0], X, Y, exact[fld], labels_exact[i],
                cmap='jet', vmin=fv_min, vmax=fv_max)
        # Col 1: predicted
        im_field = _pcolor(axes[i, 1], X, Y, pred[fld], labels_pred[i],
                           cmap='jet', vmin=fv_min, vmax=fv_max)
        # Shared colorbar for cols 0-1 of this row
        fig.colorbar(im_field, ax=axes[i, 0:2], shrink=0.85, pad=0.02, aspect=15)

        # Col 2: pointwise absolute error
        err = np.abs(pred[fld] - exact[fld])
        err_max = err.max() if err.max() > 0 else 1.0
        im_err = _pcolor(axes[i, 2], X, Y, err, labels_err[i],
                         cmap='gist_heat_r', vmin=0, vmax=err_max)
        fig.colorbar(im_err, ax=axes[i, 2], shrink=0.85, pad=0.02, aspect=15)

    # Axis labels and tick cleanup
    for ax in axes[0:2, :].flat:
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axes[:, 1:].flat:
        ax.tick_params(left=False, labelleft=False)
    for ax in axes[2, :]:
        ax.set_xlabel(r'$x$')
    for i in range(3):
        axes[i, 0].set_ylabel(r'$y$')
        axes[i, 0].set_yticks([-0.5, 0, 0.5, 1, 1.5])
        axes[i, 0].set_yticklabels(['-0.5', '0', '0.5', '1', '1.5'])
        axes[2, i].set_xticks([-0.5, 0, 0.5, 1])
        axes[2, i].set_xticklabels(['-0.5', '0', '0.5', '1'])

    _save_fig(fig, output_dir / f'kovasznay_fields_and_errors_N{N_plot}.pdf')


def plot_convergence_history(all_results, output_dir, n_colors=None):
    """Single panel: PDE residual (MSE) vs outer iteration, one curve per P."""
    if n_colors is None:
        n_colors = pub_style.N_COLORS_DEFAULT

    fig, ax = plt.subplots(1, 1, figsize=(FULL_WIDTH * 0.5, FULL_WIDTH * 0.38),
                           constrained_layout=True)

    for N in sorted(all_results.keys()):
        h = all_results[N]['history']
        iters = np.array(h['iteration'])
        pde_res = np.array(h['pde_residual'])
        color = pub_style.get_n_color(N, n_colors)
        P = all_results[N]['n_params']
        ax.semilogy(iters, pde_res, 'o-', color=color, markersize=3,
                    label=f'$P={P}$')
    ax.set_xlabel('Outer iteration')
    ax.set_ylabel('PDE residual (MSE)')
    ax.legend(loc='best')
    pub_style.style_axis(ax, grid=True)

    _save_fig(fig, output_dir / 'kovasznay_convergence_history.pdf')


# =============================================================================
# MAIN
# =============================================================================

def generate_all_plots(all_results, physics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exact = evaluate_exact_fields(physics, n_eval=200)

    # Best-performing N: fields + errors figure
    N_plot = max(all_results.keys())
    r = all_results[N_plot]
    pred = evaluate_all_fields(r['basis_u'], r['basis_v'], r['basis_p'],
                                r['theta_u'], r['theta_v'], r['theta_p'],
                                physics, n_eval=200)

    print("\nGenerating publication figures...")
    plot_fields_and_errors(exact, pred, OUTPUT_DIR, N_plot)
    plot_convergence_history(all_results, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    all_results, physics = run_all_experiments()
    generate_all_plots(all_results, physics)



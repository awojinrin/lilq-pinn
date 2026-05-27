"""
Linear Elasticity — Experiment Runner and Plotting (v2)
========================================================

Runs the LiL direct solver on the 2D linear elasticity benchmark from
Haghighat et al. (CMAME 2021) for several basis sizes, then generates
publication-quality plots.

Key changes from v1:
  - Independent basis types for u_x and u_y fields
  - Proper mixed BCs matching Fig. 1 of the paper (default)
  - Optional all-Dirichlet mode for comparison

Reusable files (place in same directory, no modifications needed):
  - lil_basis.py
  - pub_style.py

Usage:
    python run_elasticity_experiments.py

Author: Gbenga
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from elasticity_core import (
    ElasticityPhysics, ElasticityDiscretization,
    create_elasticity_basis, solve_lil_elasticity,
    evaluate_all_fields, evaluate_exact_fields,
)

import lilq_shared.style as pub_style

FULL_WIDTH = pub_style.FULL_WIDTH
FONT_SIZE = pub_style.FONT_SIZE


# =============================================================================
# CONFIGURATION
# =============================================================================

# Basis sizes to sweep: N_x = N_y = N, so P_per_field = N^2
N_VALUES = [5, 10, 15, 20, 25]

# Independent basis types for the two displacement fields.
# u_x(x,y) = cos(2*pi*x)*sin(pi*y)  -->  consider cos-type in x, sin-type in y
# u_y(x,y) = sin(pi*x)*Q*y^4/4      -->  consider sin-type in x, polynomial in y
BASIS_TYPE_U = "cos_sin"    # Basis for u_x field
BASIS_TYPE_V = "sin_cheb"    # Basis for u_y field

# Boundary condition mode: "paper" (mixed, matching Fig.1) or "exact" (all-Dirichlet)
BC_MODE = "paper"

# Loss weights
LAMBDA_PDE = 1.0
LAMBDA_BC  = 10.0

# Collocation
K_RATIO = 10
COLLOCATION_RATIOS = (0.85, 0.15)

# Output
OUTPUT_DIR = Path(f"elasticity_experiments_{BASIS_TYPE_U}_{BASIS_TYPE_V}_{BC_MODE}")


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_all_experiments():
    """Run the LiL solver for all basis sizes and collect results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    physics = ElasticityPhysics()
    all_results = {}

    for N in N_VALUES:
        Pu_label = f"{BASIS_TYPE_U}({N}x{N})"
        Pv_label = f"{BASIS_TYPE_V}({N}x{N})"
        print(f"\n{'='*70}")
        print(f"  N = {N}")
        print(f"  Basis u_x: {Pu_label}")
        print(f"  Basis u_y: {Pv_label}")
        print(f"{'='*70}")

        disc = ElasticityDiscretization(
            N_x=N, N_y=N, k_ratio=K_RATIO,
            collocation_ratios=COLLOCATION_RATIOS,
            domain_sampling="uniform",
            boundary_sampling="uniform",
            seed=42,
        )

        basis_u = create_elasticity_basis(BASIS_TYPE_U, N, N,
                                           physics.x_domain, physics.y_domain)
        basis_v = create_elasticity_basis(BASIS_TYPE_V, N, N,
                                           physics.x_domain, physics.y_domain)

        result = solve_lil_elasticity(
            physics, disc, basis_u, basis_v,
            lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC,
            bc_mode=BC_MODE,
            verbose=True,
        )
        result['N'] = N
        all_results[N] = result

        # Save checkpoint
        np.savez(
            OUTPUT_DIR / f"checkpoint_N{N}.npz",
            theta_u=result['theta_u'],
            theta_v=result['theta_v'],
            N=N,
        )

    save_summary_table(all_results, OUTPUT_DIR)
    return all_results, physics


def save_summary_table(all_results, output_dir):
    """Save a summary CSV and print a formatted table."""
    rows = []
    for N in sorted(all_results.keys()):
        r = all_results[N]
        rows.append({
            'N': N,
            'P_u': r['n_params_u'],
            'P_v': r['n_params_v'],
            'P_total': r['n_params'],
            'QR_time_s': f"{r['solve_time_qr']:.4f}",
            'Total_time_s': f"{r['solve_time_total']:.4f}",
            'PDE_MSE': f"{r['pde_mse']:.3e}",
            'Rel_L2_ux': f"{r['rel_l2_ux']:.3e}",
            'Rel_L2_uy': f"{r['rel_l2_uy']:.3e}",
            'Linf_ux': f"{r['linf_err_ux']:.3e}",
            'Linf_uy': f"{r['linf_err_uy']:.3e}",
            'Rel_L2_sxx': f"{r['rel_l2_sxx']:.3e}",
            'Rel_L2_syy': f"{r['rel_l2_syy']:.3e}",
            'Rel_L2_sxy': f"{r['rel_l2_sxy']:.3e}"
        })

    print("\n" + "="*110)
    print("SUMMARY TABLE")
    print("="*110)
    hdr = (f"{'N':>3} {'P_u':>6} {'P_v':>6} {'P_tot':>6} {'QR(s)':>8} "
           f"{'PDE MSE':>10} {'relL2 ux':>10} {'relL2 uy':>10} "
           f"{'relL2 sxx':>10} {'relL2 syy':>10} {'relL2 sxy':>10}")
    print(hdr)
    print("-"*110)
    for row in rows:
        print(f"{row['N']:>3} {row['P_u']:>6} {row['P_v']:>6} {row['P_total']:>6} "
              f"{row['QR_time_s']:>8} {row['PDE_MSE']:>10} "
              f"{row['Rel_L2_ux']:>10} {row['Rel_L2_uy']:>10} "
              f"{row['Rel_L2_sxx']:>10} {row['Rel_L2_syy']:>10} {row['Rel_L2_sxy']:>10} ")

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {output_dir / 'summary.json'}")


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def _save_fig(fig, filepath):
    filepath = Path(filepath)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    fig.savefig(filepath.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def _pcolor_panel(ax, X, Y, C, title, cmap='jet', vmin=None, vmax=None):
    im = ax.pcolormesh(X, Y, C, cmap=cmap, shading='gouraud',
                        vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_title(title)
    pub_style.style_axis(ax)
    return im


# =============================================================================
# FIELD COMPARISON PLOTS
# =============================================================================

def plot_displacement_comparison(exact, pred, output_dir, N_plot):
    # Swapped constrained_layout for layout='compressed'
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH * 0.75, FULL_WIDTH * 0.7), 
                             layout='compressed', sharex=True, sharey=True)
    X, Y = exact['X'], exact['Y']

    fields = ['ux', 'uy']
    labels_exact = [r'$u_x^*$', r'$u_y^*$']
    labels_pred  = [r'$u_x$ (LiL)', r'$u_y$ (LiL)']

    max_val = max(np.abs(exact[f]).max() for f in fields)
    vmin, vmax = -max_val, max_val

    for j, (fld, le, lp) in enumerate(zip(fields, labels_exact, labels_pred)):
        _pcolor_panel(axes[0, j], X, Y, exact[fld], le, vmin=vmin, vmax=vmax)
        im = _pcolor_panel(axes[1, j], X, Y, pred[fld], lp, vmin=vmin, vmax=vmax)

    # Cleanly remove ticks and labels for inner axes to prevent overlap/clutter
    for ax in axes[0, :]: 
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axes[:, 1]: 
        ax.tick_params(left=False, labelleft=False)
        
    axes[1, 0].set_xlabel(r'$x$'); axes[1, 1].set_xlabel(r'$x$')
    axes[0, 0].set_ylabel(r'$y$'); axes[1, 0].set_ylabel(r'$y$')

    fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    
    _save_fig(fig, output_dir / f'displacement_P{N_plot}.pdf')


def plot_stress_comparison(exact, pred, output_dir, N_plot):
    # Swapped constrained_layout for layout='compressed'
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.45), 
                             layout='compressed', sharex=True, sharey=True)
    X, Y = exact['X'], exact['Y']

    fields = ['sxx', 'syy', 'sxy']
    labels_exact = [r'$\sigma_{xx}^*$', r'$\sigma_{yy}^*$', r'$\sigma_{xy}^*$']
    labels_pred  = [r'$\sigma_{xx}$',   r'$\sigma_{yy}$',   r'$\sigma_{xy}$']

    max_val = max(np.abs(exact[f]).max() for f in fields)
    vmin, vmax = -max_val, max_val

    for j, (fld, le, lp) in enumerate(zip(fields, labels_exact, labels_pred)):
        _pcolor_panel(axes[0, j], X, Y, exact[fld], le, vmin=vmin, vmax=vmax)
        im = _pcolor_panel(axes[1, j], X, Y, pred[fld], lp, vmin=vmin, vmax=vmax)

    for ax in axes[0, :]: 
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axes[:, 1:].flat: 
        ax.tick_params(left=False, labelleft=False)
        
    for ax in axes[1, :]: ax.set_xlabel(r'$x$')
    axes[0, 0].set_ylabel(r'$y$'); axes[1, 0].set_ylabel(r'$y$')

    fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)

    _save_fig(fig, output_dir / f'stress_P{N_plot}.pdf')


def plot_strain_comparison(exact, pred, output_dir, N_plot):
    # Swapped constrained_layout for layout='compressed'
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.5), 
                             layout='compressed', sharex=True, sharey=True)
    X, Y = exact['X'], exact['Y']
    
    fields = ['exx', 'eyy', 'exy']
    labels_e = [r'$\varepsilon_{xx}^*$', r'$\varepsilon_{yy}^*$', r'$\varepsilon_{xy}^*$']
    labels_p = [r'$\varepsilon_{xx}$',   r'$\varepsilon_{yy}$',   r'$\varepsilon_{xy}$']

    max_val = max(np.abs(exact[f]).max() for f in fields)
    vmin, vmax = -max_val, max_val

    for j, (fld, le, lp) in enumerate(zip(fields, labels_e, labels_p)):
        _pcolor_panel(axes[0, j], X, Y, exact[fld], le, vmin=vmin, vmax=vmax)
        im = _pcolor_panel(axes[1, j], X, Y, pred[fld], lp, vmin=vmin, vmax=vmax)

    for ax in axes[0, :]: 
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axes[:, 1:].flat: 
        ax.tick_params(left=False, labelleft=False)
        
    for ax in axes[1, :]: ax.set_xlabel(r'$x$')
    axes[0, 0].set_ylabel(r'$y$'); axes[1, 0].set_ylabel(r'$y$')
    
    fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)

    _save_fig(fig, output_dir / f'strain_P{N_plot}.pdf')


def plot_error_fields(exact, pred, output_dir, N_plot):
    # Swapped constrained_layout for layout='compressed'
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.75, FULL_WIDTH * 0.32), 
                             layout='compressed', sharey=True)
    X, Y = exact['X'], exact['Y']

    err_ux = np.abs(pred['ux'] - exact['ux'])
    err_uy = np.abs(pred['uy'] - exact['uy'])

    vmax = max(err_ux.max(), err_uy.max())
    vmin = 0.0

    im1 = axes[0].pcolormesh(X, Y, err_ux, cmap='gist_heat_r', shading='gouraud', vmin=vmin, vmax=vmax)
    axes[0].set_aspect('equal'); axes[0].set_title(r'$|u_x - u_x^*|$')
    axes[0].set_xlabel(r'$x$'); axes[0].set_ylabel(r'$y$')
    pub_style.style_axis(axes[0])

    im2 = axes[1].pcolormesh(X, Y, err_uy, cmap='gist_heat_r', shading='gouraud', vmin=vmin, vmax=vmax)
    axes[1].set_aspect('equal'); axes[1].set_title(r'$|u_y - u_y^*|$')
    axes[1].set_xlabel(r'$x$')
    
    axes[1].tick_params(left=False, labelleft=False)
    pub_style.style_axis(axes[1])

    fig.colorbar(im2, ax=axes, shrink=0.85, pad=0.03)

    _save_fig(fig, output_dir / f'error_fields_P{N_plot}.pdf')


def plot_fields_and_errors(exact, pred, output_dir, N_plot):
    """2 rows x 3 cols: each row is one field (u_x, u_y),
    columns are exact / LiL / pointwise error."""
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.4),
                             layout='compressed', sharex=True, sharey=True)
    X, Y = exact['X'], exact['Y']
 
    fields = ['ux', 'uy']
    labels_exact = [r'$u_x$ (Actual)',           r'$u_y$ (Actual)']
    labels_pred  = [r'$u_x$ (Predicted)',       r'$u_y$ (Predicted)']
    labels_err   = [r'$u_x$ (Error)',   r'$u_y$ (Error)']
 
    for i, fld in enumerate(fields):
        # Symmetric color limits for this field
        fv = max(np.abs(exact[fld]).max(), np.abs(pred[fld]).max())
        fv_min, fv_max = -fv, fv
 
        # Col 0: exact
        _pcolor_panel(axes[i, 0], X, Y, exact[fld], labels_exact[i],
                      cmap='jet', vmin=fv_min, vmax=fv_max)
        # Col 1: predicted
        im_field = _pcolor_panel(axes[i, 1], X, Y, pred[fld], labels_pred[i],
                                 cmap='jet', vmin=fv_min, vmax=fv_max)
        # Shared colorbar for cols 0-1 of this row
        fig.colorbar(im_field, ax=axes[i, 0:2], shrink=0.85, pad=0.02, aspect=15)
 
        # Col 2: pointwise absolute error
        err = np.abs(pred[fld] - exact[fld])
        err_max = err.max() if err.max() > 0 else 1.0
        im_err = _pcolor_panel(axes[i, 2], X, Y, err, labels_err[i],
                               cmap='gist_heat_r', vmin=0, vmax=err_max)
        fig.colorbar(im_err, ax=axes[i, 2], shrink=0.85, pad=0.02, aspect=15)
 
    # Axis labels and tick cleanup
    for ax in axes[0, :]:
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axes[:, 1:].flat:
        ax.tick_params(left=False, labelleft=False)
    for ax in axes[1, :]:
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_xlabel(r'$x$')
    axes[0, 0].set_ylabel(r'$y$')
    axes[0, 0].set_yticks([0, 0.5, 1])
    axes[0, 0].set_yticklabels(['0', '0.5', '1'])
    axes[1, 0].set_ylabel(r'$y$')
    axes[1, 0].set_yticks([0, 0.5, 1])
    axes[1, 0].set_yticklabels(['0', '0.5', '1'])
 
    _save_fig(fig, output_dir / f'elasticity_fields_and_errors_N{N_plot}.pdf')


# =============================================================================
# MAIN
# =============================================================================

def generate_all_plots(all_results, physics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exact = evaluate_exact_fields(physics, n_eval=200)

    N_plot = max(all_results.keys())
    r = all_results[N_plot]
    pred = evaluate_all_fields(r['basis_u'], r['basis_v'],
                                r['theta_u'], r['theta_v'],
                                physics, n_eval=200)

    print("\nGenerating plots...")
    plot_displacement_comparison(exact, pred, OUTPUT_DIR, N_plot)
    plot_stress_comparison(exact, pred, OUTPUT_DIR, N_plot)
    plot_strain_comparison(exact, pred, OUTPUT_DIR, N_plot)
    plot_error_fields(exact, pred, OUTPUT_DIR, N_plot)

    print("\nGenerating publication figures...")
    plot_fields_and_errors(exact, pred, OUTPUT_DIR, N_plot)

    # Also plot a smaller N for comparison
    if len(all_results) > 1:
        N_small = sorted(all_results.keys())[0]
        r_s = all_results[N_small]
        pred_s = evaluate_all_fields(r_s['basis_u'], r_s['basis_v'],
                                      r_s['theta_u'], r_s['theta_v'],
                                      physics, n_eval=200)
        plot_displacement_comparison(exact, pred_s, OUTPUT_DIR, N_small)
        plot_error_fields(exact, pred_s, OUTPUT_DIR, N_small)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    all_results, physics = run_all_experiments()
    generate_all_plots(all_results, physics)

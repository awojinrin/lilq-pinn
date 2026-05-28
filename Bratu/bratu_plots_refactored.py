"""
Bratu Experiment — Refactored Plotting and Checkpointing
=========================================================

Drop-in replacements for the plotting functions in run_bratu_experiments.py.

Changes from original:
  1. Uses pub_style for consistent CMAME-quality styling
  2. Generates combined subplot figures (no LaTeX tabular assembly needed)
  3. Saves LiL coefficients alongside NiL model state_dicts
  4. Drops line-search plots (only iterations)
  5. Figure sizes set to final manuscript dimensions (no LaTeX rescaling)

To integrate:
  - Replace the plotting functions and rcParams block in run_bratu_experiments.py
  - Add checkpoint saving calls after each method in run_single_experiment()
  - Import pub_style at the top of the runner

Author: For CMAME manuscript
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the centralised style (sets all rcParams on import)
import lilq_shared.style as pub_style

# Re-export constants for convenience
FULL_WIDTH = pub_style.FULL_WIDTH
FONT_SIZE = pub_style.FONT_SIZE


# =============================================================================
# CHECKPOINT SAVING (add these calls in run_single_experiment)
# =============================================================================

def save_lil_checkpoint(basis_obj, coeffs, metrics, summary, filepath):
    """
    Save LiL results to disk so plots can be regenerated without re-training.

    Call after each LiL solve:
        save_lil_checkpoint(basis_obj, coeffs, metrics, summary,
                            exp_dir / 'ql_lil_checkpoint.npz')
    """
    # Convert metrics to serialisable format
    if hasattr(metrics, 'data'):
        metrics_data = {k: np.array(v) for k, v in metrics.data.items()}
    elif hasattr(metrics, 'to_dict'):
        metrics_data = {k: np.array(v) for k, v in metrics.to_dict().items()}
    else:
        metrics_data = {k: np.array(v) for k, v in metrics.items()}

    np.savez(filepath,
             coefficients=np.array(coeffs),
             n_basis=basis_obj.n_basis,
             **{f'metric_{k}': v for k, v in metrics_data.items()},
             **{f'summary_{k}': v for k, v in summary.items()
                if isinstance(v, (int, float, bool, str))})
    print(f"  Checkpoint saved: {filepath}")


# =============================================================================
# CONVERGENCE PLOTS — BY METHOD (2×2 grid, all P values per panel)
# =============================================================================

def plot_convergence_by_method(all_results, N_VALUES, TARGET_LOSSES, output_dir,
                                n_colors=None, basis_label=''):
    """
    2×2 grid: one panel per method, each showing all network sizes.
    This is the primary convergence figure for the manuscript.
    """
    if n_colors is None:
        n_colors = pub_style.N_COLORS_DEFAULT

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.72))
    axes = axes.flatten()

    for idx, method_key in enumerate(pub_style.METHOD_ORDER):
        ax = axes[idx]
        config = pub_style.METHOD_CONFIG[method_key]

        for N in N_VALUES:
            if N not in all_results:
                continue
            if all_results[N].get(method_key) is None:
                continue

            result = all_results[N][method_key]
            iterations, _, losses = _get_metrics(result['metrics'])
            if len(losses) == 0:
                continue

            color = pub_style.get_n_color(N, n_colors)
            P = N ** 2
            ax.semilogy(iterations, losses, color=color, linewidth=1.0,
                        label=f'${P}$')

        # Target lines
        for N in N_VALUES:
            if N in TARGET_LOSSES:
                color = pub_style.get_n_color(N, n_colors)
                ax.axhline(y=TARGET_LOSSES[N], color=color,
                           linestyle='--', linewidth=0.6, alpha=0.5)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('MSE')
        ax.set_title(config['label'])
        ax.legend(title=r'$P$', loc='best')
        pub_style.style_axis(ax)

    plt.tight_layout(w_pad=1.5, h_pad=1.5)
    _save_fig(fig, output_dir / 'convergence_by_method.pdf')


# =============================================================================
# CONVERGENCE PLOTS — BY SIZE (one panel per P, all methods)
# =============================================================================

def plot_convergence_by_size(all_results, N_VALUES, TARGET_LOSSES, output_dir):
    """
    One row of panels: one per P value, each showing all 4 methods.
    """
    n_panels = len(N_VALUES)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(FULL_WIDTH, FULL_WIDTH / n_panels * 0.85))
    if n_panels == 1:
        axes = [axes]

    for idx, N in enumerate(N_VALUES):
        ax = axes[idx]
        if N not in all_results:
            continue

        for method_key in pub_style.METHOD_ORDER:
            if all_results[N].get(method_key) is None:
                continue

            result = all_results[N][method_key]
            config = pub_style.METHOD_CONFIG[method_key]
            iterations, _, losses = _get_metrics(result['metrics'])
            if len(losses) == 0:
                continue

            ax.semilogy(iterations, losses, color=config['color'],
                        linewidth=1.0, label=config['label'])

        target = TARGET_LOSSES.get(N)
        if target is not None:
            ax.axhline(y=target, color='0.5', linestyle='--', linewidth=0.6,
                       label=f'Target')

        P = N ** 2
        ax.set_xlabel('Iterations')
        if idx == 0:
            ax.set_ylabel('MSE')
        ax.set_title(f'$P = {P}$')
        ax.legend(loc='best')
        pub_style.style_axis(ax)

    plt.tight_layout(w_pad=1.0)
    _save_fig(fig, output_dir / 'convergence_by_size.pdf')


# =============================================================================
# SOLUTION FIELD PANELS (combined subplot grid)
# =============================================================================

def plot_solution_fields(all_results, N_VALUES_TO_PLOT, output_dir,
                          evaluate_nn_solution, evaluate_lil_solution,
                          cmap='turbo'):
    """
    Combined solution field figure.
    Rows = network sizes (P), Columns = methods.
    """
    methods_with_data = []
    for method_key in pub_style.METHOD_ORDER:
        has_any = any(
            N in all_results and all_results[N].get(method_key) is not None
            for N in N_VALUES_TO_PLOT
        )
        if has_any:
            methods_with_data.append(method_key)

    n_rows = len(N_VALUES_TO_PLOT)
    n_cols = len(methods_with_data)
    if n_rows == 0 or n_cols == 0:
        return

    panel_w = FULL_WIDTH / n_cols
    panel_h = panel_w * 0.85
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(FULL_WIDTH, panel_h * n_rows),
                              squeeze=False)

    # Collect global color limits
    all_u = []
    for N in N_VALUES_TO_PLOT:
        if N not in all_results:
            continue
        for mk in methods_with_data:
            result = all_results[N].get(mk)
            if result is None:
                continue
            physics = all_results[N].get('physics')
            disc = all_results[N].get('discretization')
            if physics is None or disc is None:
                continue
            if mk in ['std_pinn', 'ql_pinn']:
                model = result.get('model')
                if model is None:
                    continue
                _, _, U = evaluate_nn_solution(model, physics, disc)
            else:
                basis = result.get('basis')
                coeffs = result.get('coefficients')
                if basis is None or coeffs is None:
                    continue
                _, _, U = evaluate_lil_solution(basis, coeffs, physics, disc)
            all_u.extend([U.min(), U.max()])

    if not all_u:
        plt.close()
        return
    vmin, vmax = min(all_u), max(all_u)

    for row_idx, N in enumerate(N_VALUES_TO_PLOT):
        if N not in all_results:
            continue
        physics = all_results[N].get('physics')
        disc = all_results[N].get('discretization')

        for col_idx, mk in enumerate(methods_with_data):
            ax = axes[row_idx, col_idx]
            result = all_results[N].get(mk)

            if result is None or physics is None or disc is None:
                ax.set_visible(False)
                continue

            if mk in ['std_pinn', 'ql_pinn']:
                model = result.get('model')
                if model is None:
                    ax.set_visible(False)
                    continue
                X, Y, U = evaluate_nn_solution(model, physics, disc)
            else:
                basis = result.get('basis')
                coeffs = result.get('coefficients')
                if basis is None or coeffs is None:
                    ax.set_visible(False)
                    continue
                X, Y, U = evaluate_lil_solution(basis, coeffs, physics, disc)

            pcm = ax.pcolormesh(X, Y, U, vmin=vmin, vmax=vmax,
                                cmap=cmap, shading='gouraud')
            ax.set_aspect('equal')
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

            if row_idx == n_rows - 1:
                ax.set_xlabel(r'$x$')
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(r'$y$')
            else:
                ax.set_yticklabels([])

            # Column header (method name) on top row only
            if row_idx == 0:
                ax.set_title(pub_style.METHOD_CONFIG[mk]['label'])

            pub_style.style_axis(ax)

    # Single colorbar on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(pcm, cax=cbar_ax)

    _save_fig(fig, output_dir / 'solution_fields.pdf')


def plot_solution_fields_single_N(all_results, N, output_dir,
                                  evaluate_nn_solution, evaluate_lil_solution,
                                  cmap='turbo'):
    """
    Generate solution field plots for a single network size N (one panel per active method).
    """
    methods_with_data = []
    for method_key in pub_style.METHOD_ORDER:
        if N in all_results and all_results[N].get(method_key) is not None:
            methods_with_data.append(method_key)

    n_cols = len(methods_with_data)
    if n_cols == 0:
        return

    # Compute figure size using pub_style width and aspect ratio
    panel_w = FULL_WIDTH / n_cols
    panel_h = panel_w * 0.85
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(FULL_WIDTH, panel_h),
                             squeeze=False)

    # Collect colorbar limits across all methods for this specific N
    all_u = []
    for mk in methods_with_data:
        result = all_results[N].get(mk)
        if result is None:
            continue
        physics = all_results[N].get('physics')
        disc = all_results[N].get('discretization')
        if physics is None or disc is None:
            continue
        if mk in ['std_pinn', 'ql_pinn']:
            model = result.get('model')
            if model is None:
                continue
            _, _, U = evaluate_nn_solution(model, physics, disc)
        else:
            basis = result.get('basis')
            coeffs = result.get('coefficients')
            if basis is None or coeffs is None:
                continue
            _, _, U = evaluate_lil_solution(basis, coeffs, physics, disc)
        all_u.extend([U.min(), U.max()])

    if not all_u:
        plt.close()
        return
    vmin, vmax = min(all_u), max(all_u)

    physics = all_results[N].get('physics')
    disc = all_results[N].get('discretization')

    for col_idx, mk in enumerate(methods_with_data):
        ax = axes[0, col_idx]
        result = all_results[N].get(mk)

        if result is None or physics is None or disc is None:
            ax.set_visible(False)
            continue

        if mk in ['std_pinn', 'ql_pinn']:
            model = result.get('model')
            if model is None:
                ax.set_visible(False)
                continue
            X, Y, U = evaluate_nn_solution(model, physics, disc)
        else:
            basis = result.get('basis')
            coeffs = result.get('coefficients')
            if basis is None or coeffs is None:
                ax.set_visible(False)
                continue
            X, Y, U = evaluate_lil_solution(basis, coeffs, physics, disc)

        pcm = ax.pcolormesh(X, Y, U, vmin=vmin, vmax=vmax,
                            cmap=cmap, shading='gouraud')
        ax.set_aspect('equal')
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())

        ax.set_xlabel(r'$x$')
        if col_idx == 0:
            ax.set_ylabel(r'$y$')
        else:
            ax.set_yticklabels([])

        ax.set_title(pub_style.METHOD_CONFIG[mk]['label'])
        pub_style.style_axis(ax)

    # Add a single colorbar on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(pcm, cax=cbar_ax)

    # Save to a size-specific filename in both PDF and PNG format
    _save_fig(fig, output_dir / f'solution_fields_N{N}.pdf')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_metrics(metrics):
    """Extract arrays from metrics (handles both object and dict formats)."""
    if hasattr(metrics, 'data'):
        data = metrics.data
    else:
        data = metrics
    iterations = np.array(data.get('iteration', []))
    func_evals = np.array(data.get('n_func_evals', []))
    losses = np.array(data.get('loss', []))
    return iterations, func_evals, losses


def _save_fig(fig, filepath):
    """Save figure as both PDF and PNG."""
    filepath = Path(filepath)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    fig.savefig(filepath.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath.name}")

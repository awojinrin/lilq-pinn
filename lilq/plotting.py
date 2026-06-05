"""
Unified Plotting Module for LiL-Q
====================================

Provides convergence, solution, and comparison plotting functions that
work across all problems. Uses the CMAME publication style from ``lilq.style``.

Usage::

    from lilq.plotting import (
        plot_convergence, plot_solution_2d, plot_method_comparison,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from typing import Dict, List, Optional

from lilq.style import (
    FULL_WIDTH, HALF_WIDTH, FONT_SIZE,
    METHOD_CONFIG, METHOD_ORDER, METHOD_COLORS,
    N_COLORS_DEFAULT, style_axis, get_n_color, add_target_lines,
)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    metrics_dict: Dict[str, dict],
    x_key: str = 'iteration',
    y_key: str = 'loss',
    title: str = 'Convergence',
    xlabel: str = 'Iterations',
    ylabel: str = 'Loss',
    log_y: bool = True,
    target_loss: float = None,
    save_path: str = None,
    figsize: tuple = None,
):
    """Plot convergence curves for multiple methods.

    Parameters
    ----------
    metrics_dict : dict
        Mapping method_name -> metrics dict (from MetricsTracker.to_dict()).
    x_key : str
        Key for x-axis data in metrics dict.
    y_key : str
        Key for y-axis data in metrics dict.
    """
    if figsize is None:
        figsize = (FULL_WIDTH, 3.5)

    fig, ax = plt.subplots(figsize=figsize)
    style_axis(ax)

    for method_name, metrics in metrics_dict.items():
        if x_key not in metrics or y_key not in metrics:
            continue

        cfg = METHOD_CONFIG.get(method_name, {})
        label = cfg.get('label', method_name)
        color = cfg.get('color', '#333333')
        ls = cfg.get('linestyle', '-')

        ax.plot(metrics[x_key], metrics[y_key],
                label=label, color=color, linestyle=ls)

    if target_loss is not None:
        ax.axhline(y=target_loss, color='gray', linestyle='--',
                   linewidth=0.7, alpha=0.6, label=f'Target ({target_loss:.1e})')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale('log')
    ax.legend()

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")

    return fig, ax


def plot_convergence_by_N(
    all_results: Dict[int, Dict[str, dict]],
    method_key: str = 'LiL-Q',
    x_key: str = 'iteration',
    y_key: str = 'loss',
    title: str = 'Convergence by N',
    target_losses: dict = None,
    save_path: str = None,
):
    """Plot convergence for one method across multiple N values."""
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))
    style_axis(ax)

    for N, results in sorted(all_results.items()):
        if method_key not in results:
            continue
        metrics = results[method_key].get('metrics_history', {})
        if x_key not in metrics or y_key not in metrics:
            continue

        color = get_n_color(N)
        ax.plot(metrics[x_key], metrics[y_key],
                label=f'N={N}', color=color)

    if target_losses:
        add_target_lines(ax, target_losses)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Solution Plots (2D)
# ─────────────────────────────────────────────────────────────────────────────

def plot_solution_2d(
    X, Y, U,
    title: str = 'Solution',
    colorbar_label: str = 'u',
    cmap: str = 'jet',
    save_path: str = None,
    figsize: tuple = None,
    use_diverging: bool = False,
):
    """Plot a 2D scalar field as a filled contour."""
    if figsize is None:
        figsize = (HALF_WIDTH + 0.5, HALF_WIDTH)

    fig, ax = plt.subplots(figsize=figsize)
    style_axis(ax)

    if use_diverging:
        vabs = max(abs(np.nanmin(U)), abs(np.nanmax(U)))
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        cf = ax.contourf(X, Y, U, levels=50, cmap=cmap, norm=norm)
    else:
        cf = ax.contourf(X, Y, U, levels=50, cmap=cmap)

    plt.colorbar(cf, ax=ax, label=colorbar_label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_solution_comparison(
    X, Y, U_pred, U_exact, U_error=None,
    labels=('Predicted', 'Exact', 'Error'),
    suptitle: str = '',
    cmap: str = 'jet',
    save_path: str = None,
):
    """3-panel comparison: predicted, exact, pointwise error."""
    ncols = 3 if U_error is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(FULL_WIDTH, 2.5))
    style_axis(axes[0])
    style_axis(axes[1])

    for i, (U, label) in enumerate([(U_pred, labels[0]), (U_exact, labels[1])]):
        cf = axes[i].contourf(X, Y, U, levels=50, cmap=cmap)
        plt.colorbar(cf, ax=axes[i])
        axes[i].set_title(label)
        axes[i].set_aspect('equal')

    if U_error is not None and ncols == 3:
        style_axis(axes[2])
        vabs = max(abs(np.nanmin(U_error)), abs(np.nanmax(U_error)))
        if vabs > 0:
            norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
            cf = axes[2].contourf(X, Y, U_error, levels=50, cmap='seismic', norm=norm)
        else:
            cf = axes[2].contourf(X, Y, U_error, levels=50, cmap='seismic')
        plt.colorbar(cf, ax=axes[2])
        axes[2].set_title(labels[2])
        axes[2].set_aspect('equal')

    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT_SIZE + 1)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Method Comparison Bar Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_comparison_bars(
    results: Dict[str, dict],
    metric_key: str = 'final_loss',
    title: str = 'Method Comparison',
    ylabel: str = 'Loss',
    log_y: bool = True,
    save_path: str = None,
):
    """Bar chart comparing methods by a single metric."""
    methods = []
    values = []
    colors = []

    for method_name in METHOD_ORDER:
        if method_name in results and metric_key in results[method_name]:
            cfg = METHOD_CONFIG.get(method_name, {})
            methods.append(cfg.get('label', method_name))
            values.append(results[method_name][metric_key])
            colors.append(cfg.get('color', '#333333'))

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 3.0))
    style_axis(ax)

    ax.bar(methods, values, color=colors, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale('log')

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# SVD / Condition Number Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_svd_spectrum(
    svd_results: List[dict],
    title: str = 'Singular Value Spectrum',
    save_path: str = None,
):
    """Plot singular value spectra from SVD analysis results."""
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))
    style_axis(ax)

    for i, result in enumerate(svd_results):
        sigma = result.get('sigma', [])
        label = f"Iter {result.get('quasi_iter', i)}"
        ax.semilogy(sigma, label=label, alpha=0.7)

    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_condition_vs_N(
    condition_data: dict,
    title: str = 'Condition Number vs. N',
    save_path: str = None,
):
    """Plot condition number growth with basis size.

    Parameters
    ----------
    condition_data : dict mapping N -> list of SVD results
    """
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))
    style_axis(ax)

    N_values = sorted(condition_data.keys())
    final_kappa = [condition_data[N][-1]['kappa'] for N in N_values]

    ax.semilogy(N_values, final_kappa, 'o-', color=METHOD_COLORS['ql_lil'])

    ax.set_xlabel('N (basis size per dimension)')
    ax.set_ylabel('Condition Number')
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
    return fig, ax

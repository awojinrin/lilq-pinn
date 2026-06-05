"""
Shared Experiment Utilities
==============================

Common helpers for experiment runners: directory management, result saving,
checkpoint I/O, convergence plotting, and solution field visualisation.
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Directory & I/O
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path(__file__).resolve().parent.parent / 'results'


def make_experiment_dir(problem: str, basis: str, N: int,
                        root: Path = None) -> Path:
    """Create ``results/<problem>_experiments_<basis>/N_<N>/`` directory tree.

    Returns the N-specific subdirectory.
    """
    if root is None:
        root = RESULTS_ROOT
    exp_dir = root / f"{problem}_experiments_{basis}" / f"N_{N}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def make_figures_dir(problem: str, basis: str, root: Path = None) -> Path:
    """Create ``results/<problem>_experiments_<basis>/figures/``."""
    if root is None:
        root = RESULTS_ROOT
    fig_dir = root / f"{problem}_experiments_{basis}" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def experiment_base_dir(problem: str, basis: str, root: Path = None) -> Path:
    if root is None:
        root = RESULTS_ROOT
    d = root / f"{problem}_experiments_{basis}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Save/Load
# ─────────────────────────────────────────────────────────────────────────────

def save_lil_checkpoint(path: Path, basis, coefficients, config_dict: dict):
    """Save LiL model: basis object + coefficients + config."""
    np.savez(path,
             coefficients=coefficients,
             config=json.dumps(config_dict))
    print(f"    Saved checkpoint: {path}")


def save_nn_checkpoint(path: Path, model, config_dict: dict):
    """Save NN model state dict + config."""
    import torch
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict,
    }, path)
    print(f"    Saved checkpoint: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(path: Path, metrics):
    """Save metrics object (with .to_dict()) as CSV."""
    if hasattr(metrics, 'to_dict'):
        data = metrics.to_dict()
    elif isinstance(metrics, dict):
        data = metrics
    else:
        return

    if not data:
        return

    keys = list(data.keys())
    n_rows = len(data[keys[0]]) if isinstance(data[keys[0]], list) else 0
    if n_rows == 0:
        return

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n_rows):
            writer.writerow([data[k][i] if isinstance(data[k], list) and i < len(data[k])
                             else data[k] for k in keys])
    print(f"    Saved metrics: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Results JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_master_results(path: Path, all_results: dict, config_info: dict):
    """Save master experiment results as JSON."""
    payload = {
        'config': {**config_info, 'timestamp': datetime.now().isoformat()},
        'results': {},
    }
    for key, val in all_results.items():
        payload['results'][str(key)] = val

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\nMaster results saved: {path}")


def save_summary_json(path: Path, summary: dict):
    """Save per-method summary as JSON."""
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence Plots
# ─────────────────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    'NiL-N': '#E63946',
    'NiL-Q': '#457B9D',
    'LiL-N': '#2A9D8F',
    'LiL-Q': '#F4A261',
}
METHOD_MARKERS = {'NiL-N': 'o', 'NiL-Q': 's', 'LiL-N': '^', 'LiL-Q': 'D'}


def plot_convergence_by_method(all_results: dict, N_values: list,
                               problem_label: str, fig_dir: Path,
                               x_key: str = 'iteration'):
    """2x2 panel: one subplot per method, all N overlaid.

    x_key can be 'iteration' or 'n_func_evals' (line searches).
    """
    try:
        import lilq.style  # noqa: F401
    except ImportError:
        pass

    methods = ['NiL-N', 'NiL-Q', 'LiL-N', 'LiL-Q']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, method in zip(axes.ravel(), methods):
        ax.set_title(method, fontsize=10, fontweight='bold')
        ax.set_xlabel('Iteration' if x_key == 'iteration' else 'Function Evaluations')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3, which='both')

        for N in N_values:
            res = all_results.get(N, {})
            method_data = res.get(method, {})
            hist = method_data.get('metrics_history', {})

            x_data = hist.get(x_key,
                             hist.get('iteration',
                             hist.get('quasi_iter', [])))
            y_data = hist.get('loss',
                             hist.get('total_loss',
                             hist.get('nonlinear_residual', [])))
            if x_data and y_data:
                ax.semilogy(x_data, y_data, label=f'N={N}', lw=1.5, ms=3)

        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8)

    plt.suptitle(f'{problem_label} — Convergence by Method', fontsize=12)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(fig_dir / f'{problem_label}_convergence_by_method.{ext}',
                    bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved convergence_by_method plots")


def plot_convergence_by_size(all_results: dict, N_values: list,
                             problem_label: str, fig_dir: Path):
    """One subplot per N, all methods overlaid."""
    try:
        import lilq.style  # noqa: F401
    except ImportError:
        pass

    n_panels = len(N_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    methods = ['NiL-N', 'NiL-Q', 'LiL-N', 'LiL-Q']

    for ax, N in zip(axes, N_values):
        ax.set_title(f'N = {N}', fontsize=10)
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3, which='both')

        res = all_results.get(N, {})
        for method in methods:
            method_data = res.get(method, {})
            hist = method_data.get('metrics_history', {})
            x_data = hist.get('n_func_evals',
                             hist.get('iteration',
                             hist.get('quasi_iter', [])))
            y_data = hist.get('loss',
                             hist.get('total_loss',
                             hist.get('nonlinear_residual', [])))
            if x_data and y_data:
                color = METHOD_COLORS.get(method, None)
                ax.semilogy(x_data, y_data, label=method,
                            color=color, lw=1.5, ms=3)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=7)

    plt.suptitle(f'{problem_label} — Convergence by Size', fontsize=12)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(fig_dir / f'{problem_label}_convergence_by_size.{ext}',
                    bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved convergence_by_size plots")


# ─────────────────────────────────────────────────────────────────────────────
# Solution Field Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_solution_field(X, Y, U, title: str, fig_dir: Path,
                        filename: str, cmap='jet',
                        xlabel='x', ylabel='y', clabel='u'):
    """Single 2D contour plot saved as PDF + PNG."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cf = ax.contourf(X, Y, U, levels=50, cmap=cmap)
    plt.colorbar(cf, ax=ax, label=clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(fig_dir / f'{filename}.{ext}', bbox_inches='tight', dpi=150)
    plt.close()


def plot_solution_panel(fields_dict: dict, title: str, fig_dir: Path,
                        filename: str, cmap='jet'):
    """Multi-panel solution plot from a dict of {label: (X, Y, U)}."""
    n = len(fields_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, (X, Y, U)) in zip(axes, fields_dict.items()):
        cf = ax.contourf(X, Y, U, levels=50, cmap=cmap)
        plt.colorbar(cf, ax=ax, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(label)
        ax.set_aspect('equal')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(fig_dir / f'{filename}.{ext}', bbox_inches='tight', dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Beltrami 3D Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_fields_2d_beltrami(exact, pred, fig_dir: Path, label: str,
                            t_val: float, z_val: float):
    """3-row x 4-col 2D field panel: exact / predicted / |error| for u,v,w,p."""
    fig, axes = plt.subplots(3, 4, figsize=(10, 5.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.05)
    X, Y = exact['X'], exact['Y']
    flds = ['u', 'v', 'w', 'p']

    for j, fld in enumerate(flds):
        vmin = min(exact[fld].min(), pred[fld].min())
        vmax = max(exact[fld].max(), pred[fld].max())
        for row, (data, ttl) in enumerate([(exact[fld], f'${fld}^*$'),
                                            (pred[fld], f'${fld}$')]):
            ax = axes[row, j]
            im = ax.pcolormesh(X, Y, data, cmap='jet', shading='gouraud',
                               vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_aspect('equal')
            ax.set_title(ttl, fontsize=7, pad=2)
            ax.tick_params(labelsize=5, length=2)
            if row < 2:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=15).ax.tick_params(
                labelsize=4, length=1.5)
        ax = axes[2, j]
        err = np.abs(pred[fld] - exact[fld])
        im_e = ax.pcolormesh(X, Y, err, cmap='hot', shading='gouraud', rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'$|\\Delta {fld}|$', fontsize=7, pad=2)
        ax.tick_params(labelsize=5, length=2)
        if j > 0:
            ax.set_yticklabels([])
        fig.colorbar(im_e, ax=ax, shrink=0.8, pad=0.02, aspect=15).ax.tick_params(
            labelsize=4, length=1.5)
    for ax in axes[2, :]:
        ax.set_xlabel('$x$', fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('$y$', fontsize=7)
    fig.suptitle(f'$z={z_val}$, $t={t_val}$, {label}', fontsize=8, y=1.01)
    stem = f'fields_{label}_t{t_val}_z{z_val}'
    for ext in ('pdf', 'png'):
        fig.savefig(fig_dir / f'{stem}.{ext}', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {stem}")


def plot_3d_volumes_beltrami(exact, pred, fig_dir: Path, label: str, t_val: float):
    """Publication-quality 3D cube rendering: 4 rows (u,v,w,p) x 3 cols (Exact, LiL-Q, |Error|).

    Each cube is rendered by painting its 6 outer faces with smoothed colormaps.
    Matches the original codebase vol3d_composite output exactly.
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    from scipy.ndimage import zoom

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    X, Y, Z = exact['X'], exact['Y'], exact['Z']
    flds = ['u', 'v', 'w', 'p']
    col_titles = ['Exact', 'LiL-Q', 'Error']
    nrow, ncol = 4, 3

    FIG_W = 6.5
    FIG_H = FIG_W * 1.20
    ELEV, AZIM = 25, 45
    SMOOTH = 2
    CB_SHRINK, CB_PAD, CB_ASPECT, CB_N_TICKS = 0.55, 0.02, 14, 4

    def _smooth_face(sx, sy, sz, sv, factor):
        return (zoom(sx, factor, order=3), zoom(sy, factor, order=3),
                zoom(sz, factor, order=3), zoom(sv, factor, order=3))

    def _face_specs(V):
        return [
            (X[0,:,:], Y[0,:,:], Z[0,:,:], V[0,:,:]),
            (X[-1,:,:], Y[-1,:,:], Z[-1,:,:], V[-1,:,:]),
            (X[:,0,:], Y[:,0,:], Z[:,0,:], V[:,0,:]),
            (X[:,-1,:], Y[:,-1,:], Z[:,-1,:], V[:,-1,:]),
            (X[:,:,0], Y[:,:,0], Z[:,:,0], V[:,:,0]),
            (X[:,:,-1], Y[:,:,-1], Z[:,:,-1], V[:,:,-1]),
        ]

    def _paint_cube(ax, V, cmap_obj, norm):
        for sx, sy, sz, sv in _face_specs(V):
            sxs, sys, szs, svs = _smooth_face(sx, sy, sz, sv, SMOOTH)
            ax.plot_surface(sxs, sys, szs, facecolors=cmap_obj(norm(svs)),
                            shade=False, antialiased=True, rasterized=True)

    def _strip_ax(ax):
        ax.set_axis_off()
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(Z.min(), Z.max())
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=ELEV, azim=AZIM)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig,
                           wspace=0.10, hspace=0.04,
                           left=0.05, right=0.93, top=0.96, bottom=0.01)

    for i, fld in enumerate(flds):
        ex_v, pr_v = exact[fld], pred[fld]
        er_v = np.abs(pr_v - ex_v)

        vmin = float(min(ex_v.min(), pr_v.min()))
        vmax = float(max(ex_v.max(), pr_v.max()))
        emin, emax = float(er_v.min()), float(er_v.max())
        if emax - emin < 1e-15:
            emax = emin + 1e-15

        cmap_fld = plt.colormaps['jet']
        cmap_err = plt.colormaps['hot']
        norm_fld = mcolors.Normalize(vmin=vmin, vmax=vmax)
        norm_err = mcolors.Normalize(vmin=emin, vmax=emax)

        views = [
            (ex_v, cmap_fld, norm_fld),
            (pr_v, cmap_fld, norm_fld),
            (er_v, cmap_err, norm_err),
        ]

        for j, (data, cmap_obj, norm) in enumerate(views):
            ax = fig.add_subplot(gs[i, j], projection='3d')
            _paint_cube(ax, data, cmap_obj, norm)
            _strip_ax(ax)

            if i == 0:
                ax.set_title(col_titles[j], fontsize=9, pad=4)
            if j == 0:
                ax.text2D(-0.06, 0.5, f'${fld}$', transform=ax.transAxes,
                          fontsize=9, ha='center', va='center', rotation=90)

            sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=CB_SHRINK, pad=CB_PAD, aspect=CB_ASPECT)

            if j == 0:
                cb.ax.set_visible(False)
                cb.outline.set_visible(False)
                continue

            cb.outline.set_linewidth(0.6)
            cb.ax.tick_params(labelsize=8, length=2, width=0.5, pad=2, direction='out')
            cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=CB_N_TICKS))

            if j == 2:
                fmt = ticker.ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((-2, 2))
                cb.ax.yaxis.set_major_formatter(fmt)
                cb.ax.yaxis.get_offset_text().set_fontsize(8)
            else:
                span = vmax - vmin
                if span < 1:
                    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                else:
                    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    stem = f'vol3d_composite_{label}_t{t_val}'
    fig.savefig(fig_dir / f'{stem}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {stem}.pdf/png")


def print_summary_table(all_results: dict, N_values: list,
                        methods: list, problem_label: str):
    """Print a formatted summary table to console."""
    print(f"\n{'=' * 80}")
    print(f"{problem_label} RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'N':>4s} | {'Method':>8s} | {'Loss':>12s} | {'Iters':>7s} | "
          f"{'Time':>8s} | {'Status':>10s}")
    print('-' * 80)

    for N in N_values:
        res = all_results.get(N, {})
        for method in methods:
            if method in res and 'error' not in res[method]:
                s = res[method]
                status = "CONVERGED" if s.get('converged') else "ran"
                iters = s.get('total_iterations', s.get('n_quasi_iters',
                              s.get('n_outer_iters', '-')))
                print(f"{N:4d} | {method:>8s} | {s['final_loss']:12.4e} | "
                      f"{str(iters):>7s} | {s.get('training_time', 0):7.2f}s | "
                      f"{status:>10s}")
            elif method in res:
                print(f"{N:4d} | {method:>8s} | {'FAILED':>12s}")

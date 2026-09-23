"""
Residual-band Figures (Computational_Package_1_v2.md Section 3.5)
======================================================================

Regenerates the four figures that compare ||R_lin^(k)||_h and
||R^(k+1)||_h against k for each P (Bratu, Burgers, Buckley-Leverett
viscous, Buckley-Leverett gravity), from real LiL-Q reruns logged with
the shared Section 3.1 instrumentation (``lilq.iteration_log``). Per the
spec: y-axis labeled ||.||_h (not "MSE" -- the present repository has no
such figures at all, corrected or otherwise), log scale, one panel per
P, both curves, with chi_k in a second row of panels (the spec's stated
alternative to a secondary axis -- chosen here because chi_k's dynamic
range, log-scale by nature, does not share an axis cleanly with the two
already-log-scale residual curves).

Every problem/size/stopping-rule choice below is imported directly from
that problem's own ``experiments/run_*.py`` (not duplicated as separate
constants here) so this script is guaranteed to use the exact same
"paper settings" (Section 3.3) as the rest of this codebase already
does, with no risk of silently drifting from them.

Usage::

    python experiments/residual_band_figures.py
    python experiments/residual_band_figures.py --quick   # small smoke-test sizes
"""

import sys
import os
import argparse
import dataclasses
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np
import matplotlib.pyplot as plt

from lilq.iteration_log import IterationLogger
from problems.bratu import BratuConfig, BratuOptConfig, run_lil_q as run_bratu_lil_q
from problems.burgers import BurgersConfig, BurgersOptConfig, run_lil_q as run_burgers_lil_q
from problems.buckley_leverett import BLConfig, BLOptConfig, run_lil_q as run_bl_lil_q


RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'residual_band_figures'


# ─────────────────────────────────────────────────────────────────────────────
# Per-problem run configuration -- sizes/targets/budgets pulled straight
# from the existing experiment scripts (Section 3.3's "paper settings").
# Neither LiL-Q pretraining step (``pretrain_lil``, shared by all three
# problems) takes an epoch count -- it's a least-squares fit, not an
# epoch-based optimizer -- so ``pretrain_epochs`` from those scripts is
# irrelevant here and intentionally not imported.
# ─────────────────────────────────────────────────────────────────────────────

def _bratu_runs(quick=False):
    from experiments.run_bratu import (
        DEFAULT_N_VALUES, TARGET_LOSSES, MAX_QUASI_ITERS,
        DEFAULT_LAMBDA, DEFAULT_BASIS, DEFAULT_K_RATIO,
    )
    N_values = DEFAULT_N_VALUES[:2] if quick else DEFAULT_N_VALUES
    runs = []
    for N in N_values:
        config = BratuConfig(lambda_=DEFAULT_LAMBDA, N_x=N, N_y=N,
                              k_ratio=DEFAULT_K_RATIO, basis_type=DEFAULT_BASIS)
        opt = BratuOptConfig(R_tol=TARGET_LOSSES[N], max_quasi_iters_lil=MAX_QUASI_ITERS)
        runs.append((N * N, config, opt))
    return runs


def _burgers_runs(quick=False):
    from experiments.run_burgers import (
        DEFAULT_N_VALUES, TARGET_LOSSES, MAX_QUASI_ITERS,
        DEFAULT_BASIS, VISCOSITY, T_FINAL, K_RATIO,
    )
    N_values = DEFAULT_N_VALUES[:2] if quick else DEFAULT_N_VALUES
    runs = []
    for N in N_values:
        config = BurgersConfig(N_x=N, N_t=N, viscosity=VISCOSITY, T_final=T_FINAL,
                                basis_type=DEFAULT_BASIS, k_ratio=K_RATIO)
        opt = BurgersOptConfig(R_tol=TARGET_LOSSES[N], max_quasi_iters_lil=MAX_QUASI_ITERS)
        runs.append((N * N, config, opt))
    return runs


def _bl_runs(gravity, quick=False):
    from experiments.run_bl import (
        DEFAULT_N_VALUES, TARGET_LOSSES, MAX_QUASI_ITERS,
        GRAVITY_TARGET_LOSSES, GRAVITY_MAX_QUASI_ITERS,
        DEFAULT_BASIS, GRAVITY_BASIS, K_RATIO,
    )
    N_values = DEFAULT_N_VALUES[:2] if quick else DEFAULT_N_VALUES
    targets = GRAVITY_TARGET_LOSSES if gravity else TARGET_LOSSES
    quasi_iters = GRAVITY_MAX_QUASI_ITERS if gravity else MAX_QUASI_ITERS
    basis_type = GRAVITY_BASIS if gravity else DEFAULT_BASIS
    runs = []
    for N in N_values:
        base = BLConfig.with_gravity() if gravity else BLConfig()
        config = dataclasses.replace(base, N_x=N, N_t=N,
                                      basis_type=basis_type, k_ratio=K_RATIO)
        opt = BLOptConfig(R_tol=targets[N], max_quasi_iters_lil=quasi_iters)
        runs.append((N * N, config, opt))
    return runs


# label, title, run-list builder, LiL-Q entry point
PROBLEMS = [
    ('bratu', 'Bratu', _bratu_runs, run_bratu_lil_q),
    ('burgers', 'Burgers', _burgers_runs, run_burgers_lil_q),
    ('bl', 'Buckley–Leverett (viscous)', lambda quick: _bl_runs(False, quick), run_bl_lil_q),
    ('bl_gravity', 'Buckley–Leverett (gravity)', lambda quick: _bl_runs(True, quick), run_bl_lil_q),
]


# ─────────────────────────────────────────────────────────────────────────────
# Run + log + plot
# ─────────────────────────────────────────────────────────────────────────────

def run_and_log(label, runs_fn, run_fn, out_dir, quick=False, verbose=True):
    """Run LiL-Q at every configured P for one problem, with full Section
    3.1 instrumentation, saving each run's own iterations.csv (the "CSV
    behind each panel" the spec asks for) and returning {P: IterationLogger}.
    """
    loggers = {}
    for P, config, opt in runs_fn(quick):
        logger = IterationLogger()
        if verbose:
            print(f"  P={P} ...", end=' ', flush=True)
        run_fn(config, opt, verbose=False, iteration_logger=logger)
        loggers[P] = logger
        logger.to_csv(out_dir / f"{label}_P{P}_iterations.csv")
        if verbose:
            n_rows = len(logger)
            final_norm = logger.rows[-1]['norm_R_h'] if n_rows else float('nan')
            print(f"{n_rows} iterations, final ||R||_h={final_norm:.3e}")
    return loggers


def plot_residual_bands(label, title, loggers, out_dir):
    """One figure per problem: top row = ||R_lin^(k)||_h & ||R^(k+1)||_h
    (log scale) vs k, one panel per P; bottom row = chi_k, same columns.
    """
    Ps = sorted(loggers.keys())
    n = len(Ps)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * max(n, 1), 6), squeeze=False)

    for col, P in enumerate(Ps):
        rows = loggers[P].rows
        k = [r['k'] for r in rows]
        r_lin = [r['norm_Rlin_h'] for r in rows]
        r_next = [r['norm_R_h'] for r in rows]
        chi = [r['chi'] for r in rows]

        ax_top = axes[0, col]
        ax_top.semilogy(k, r_lin, 'o-', ms=3, lw=1.2, color='#457B9D',
                         label=r'$\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$')
        ax_top.semilogy(k, r_next, 's-', ms=3, lw=1.2, color='#E63946',
                         label=r'$\|\mathbf{R}^{(k+1)}\|_h$')
        ax_top.set_title(f'$P={P}$', fontsize=10)
        ax_top.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax_top.set_ylabel(r'$\|\cdot\|_h$')
            ax_top.legend(fontsize=7, loc='best')

        ax_bot = axes[1, col]
        ax_bot.semilogy(k, chi, 'd-', ms=3, lw=1.2, color='#6a3d9a')
        ax_bot.set_xlabel('$k$')
        ax_bot.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax_bot.set_ylabel(r'$\chi_k$')

    fig.suptitle(f'{title} — residual bands', fontsize=12)
    fig.tight_layout()

    pdf_path = out_dir / f"{label}_residual_bands.pdf"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(out_dir / f"{label}_residual_bands.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Section 3.5 residual-band figures")
    parser.add_argument('--quick', action='store_true',
                        help='Smoke-test sizes (first 2 P per problem) instead of the full paper sweep')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: results/residual_band_figures/)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import lilq.style  # noqa: F401
    except ImportError:
        pass

    for label, title, runs_fn, run_fn in PROBLEMS:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        loggers = run_and_log(label, runs_fn, run_fn, out_dir, quick=args.quick)
        pdf_path = plot_residual_bands(label, title, loggers, out_dir)
        print(f"  Saved {pdf_path}")

    print(f"\nAll figures and CSVs written to {out_dir}")


if __name__ == '__main__':
    main()
